import argparse
from pathlib import Path
import yaml
from tqdm.auto import tqdm
from transformers import get_scheduler, GPT2Tokenizer, GPTNeoForCausalLM
from topicrm.dataloader import FinetuneDataset, CorpusLoader, ConcatDataset, TopicLoaderLda, TopicDataset, MaxTokenLoader
import topicrm.dataloader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from topicrm.losses import log_prob, weighted_log_prob
from accelerate import Accelerator
import logging
import time
from topicrm.utils import get_timestamp_str, show_gpu, CheckpointManager
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from contextlib import nullcontext

logger = logging.getLogger(__file__)

def load_dataset(dataset_type, tokenizer, all_data, topic_data, topic_probs, topic_ids,
                    threshold, include_set, exclude_set, max_length, ldamodel, dictionary):
    if dataset_type == "TopicLoaderLda":
        topic_corpus = CorpusLoader(all_data, include=include_set, exclude=exclude_set, return_dict=True)
        non_topic_corpus = CorpusLoader(all_data, return_dict=True)
        if max_length != tokenizer.model_max_length:
            topic_corpus = MaxTokenLoader(topic_corpus, tokenizer, max_length)
            non_topic_corpus = MaxTokenLoader(non_topic_corpus, tokenizer, max_length)
        dictionary = Dictionary.load(dictionary)
        ldamodel = LdaModel.load(ldamodel)
        topic_data = TopicLoaderLda(topic_corpus, dictionary, ldamodel, topic_ids, threshold=threshold)
        non_topic_data = TopicLoaderLda(non_topic_corpus, dictionary, ldamodel, topic_ids, threshold=threshold, keep=False)
    else:
        non_topic_data = TopicDataset(all_data, topic_probs, topic_ids, threshold=threshold, keep=False)
        if dataset_type == "CorpusLoader":
            topic_data = CorpusLoader(topic_data, include=include_set, exclude=exclude_set, return_dict=True)
        elif dataset_type == "ConcatDataset":
            topic_data = ConcatDataset.corpus_from_dir(topic_data, include=include_set, exclude=exclude_set)
        else:
            raise NotImplementedError("Not implemented %s"%(dataset_type))
        if max_length != tokenizer.model_max_length:
            topic_data = MaxTokenLoader(topic_data, tokenizer, max_length)
            non_topic_data = MaxTokenLoader(non_topic_data, tokenizer, max_length)
    return topic_data, non_topic_data

def prepare_dataloader(config, tokenizer, accelerator=None):
    topic_probs = config['probpath']
    all_data = config['alldata']
    topic_data_path = config["topicdata"]
    val_topic_probs = config['val_probpath']
    val_all_data = config['val_alldata']
    val_topic_data_path = config['val_topicdata']
    topic_ids = config['topics']
    threshold = config['thresh']
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    include_set = set(config["include_set"]) if "include_set" in config else None
    exclude_set = set(config["exclude_set"]) if "exclude_set" in config else None
    if config["truncate_len"] is not None:
        max_length = config["truncate_len"]
    else:
        max_length = tokenizer.model_max_length
    logger.info(f'Max input lenght {max_length}')
    ldamodel = config.get('ldamodel', None)
    dictionary = config.get('dictionary', None)

    if include_set is not None:
        logger.info("Including data only from sets %s"%(include_set))
    if exclude_set is not None:
        logger.info("Excluding data from sets %s"%(exclude_set))

    topic_data, non_topic_data = load_dataset(config["topicdata_type"], tokenizer, all_data, topic_data_path, topic_probs, topic_ids,
                    threshold, include_set, exclude_set, max_length, ldamodel, dictionary)
    val_topic_data, val_non_topic_data = load_dataset(config["val_topicdata_type"], tokenizer, val_all_data, val_topic_data_path, val_topic_probs, topic_ids,
                    threshold, include_set, exclude_set, max_length, ldamodel, dictionary)

    dataset = FinetuneDataset(topic_data, non_topic_data)
    val_dataset = FinetuneDataset(val_topic_data, val_non_topic_data)

    def tokenize_function(examples):
        samples1, samples2 = [*zip(*examples)]
        texts1 = [x[topicrm.dataloader.TEXT_KEY] for x in samples1]
        texts2 = [x[topicrm.dataloader.TEXT_KEY] for x in samples2]
        if accelerator is not None and accelerator.num_processes > 1:
            padding = 'max_length'
        else:
            padding = 'longest'
        outputs1 = tokenizer(texts1, return_tensors="pt", padding=padding, truncation=True, max_length=max_length)
        outputs2 = tokenizer(texts2, return_tensors="pt", padding=padding, truncation=True, max_length=max_length)
        if topicrm.dataloader.PROB_KEY in samples1[0]:
            outputs1[topicrm.dataloader.PROB_KEY] = torch.tensor([x[topicrm.dataloader.PROB_KEY] for x in samples1])
            outputs2[topicrm.dataloader.PROB_KEY] = torch.tensor([x[topicrm.dataloader.PROB_KEY] for x in samples2])
        return outputs1, outputs2
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=tokenize_function)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=tokenize_function)

    return dataloader, val_dataloader

def prepare_optimizer_scheduler(config, model):
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = get_scheduler(
        config['scheduler'],
        optimizer,
        num_training_steps=config.get('num_training_steps', None), 
        num_warmup_steps=config.get('num_warmup_steps', None)
    )
    return optimizer, scheduler

def log_train(config, summary_writer: SummaryWriter, n_step, loss_topic, loss_no_topic, steptime, scheduler):
    logger.info('Step {}, loss {}, topic {}, no topic {}'.format(n_step, loss_topic+loss_no_topic, loss_topic, loss_no_topic))
    summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], n_step)
    summary_writer.add_scalar('train/topicloss', loss_topic, n_step)
    summary_writer.add_scalar('train/notopicloss', loss_no_topic, n_step)
    summary_writer.add_scalar('train/loss', loss_topic+loss_no_topic, n_step)
    summary_writer.add_scalar('train/time', steptime, n_step)
    if config["device"] == "cuda":
        pct = show_gpu(f'{n_step}: ')
        for i, p in enumerate(pct):
            summary_writer.add_scalar(f'train/cuda:{i}-mem', p, n_step)

def log_epoch(summary_writer: SummaryWriter, epoch_time):
    summary_writer.add_scalar('train/time', epoch_time)

def forward_model(config, batch_topic, batch_notopic, tokenizer, modelf, model0=None):
    if not config["accelerate"]:
        batch_topic = batch_topic.to(config['device'])
        batch_notopic = batch_notopic.to(config['device'])
    
    input_ids_topic = batch_topic.input_ids
    input_ids_notopic = batch_notopic.input_ids
    att_mask_topic = batch_topic.attention_mask
    att_mask_notopic = batch_notopic.attention_mask
    logitsf_topic   = modelf(input_ids_topic,   attention_mask=att_mask_topic.long()).logits
    logitsf_notopic = modelf(input_ids_notopic, attention_mask=att_mask_notopic.long()).logits
    
    if config.get('loss', 'log_prob') == 'log_prob':
        loss_topic = log_prob(input_ids_topic, att_mask_topic, logitsf_topic)
        loss_notopic = -log_prob(input_ids_notopic, att_mask_notopic, logitsf_notopic)
    elif config['loss'] == 'weighted_log_prob':
        probs = batch_topic.prob
        probs_notopic = 1 - batch_notopic.prob
        loss_topic = weighted_log_prob(input_ids_topic, att_mask_topic, probs, logitsf_topic)
        loss_notopic = -weighted_log_prob(input_ids_notopic, att_mask_notopic, probs_notopic, logitsf_notopic)

    return loss_topic, loss_notopic, (logitsf_topic, logitsf_notopic)

def log_val(summary_writer: SummaryWriter, config, n_step, tokenizer, modelf, val_dataloader, model0=None, accelerator=None):
    n_batches = 0
    total_loss_topic = []
    total_loss_notopic = []
    progress_bar = None
    try:
        progress_bar = tqdm(total=len(val_dataloader))
    except TypeError:
        pass
    start = time.time()
    for batch in val_dataloader:
        batch_topic, batch_notopic = batch
        loss_topic, loss_notopic, _ = forward_model(
            config, batch_topic, batch_notopic, tokenizer, modelf, model0=model0)
        if accelerator is not None:
            all_loss_topic, all_loss_notopic = accelerator.gather_for_metrics((loss_topic, loss_notopic))
            all_loss_topic = all_loss_topic.mean()
            all_loss_notopic = all_loss_notopic.mean()
        else:
            all_loss_topic, all_loss_notopic = loss_topic, loss_notopic
        total_loss_topic.append(all_loss_topic.detach().float())
        total_loss_notopic.append(all_loss_notopic.detach().float())
        n_batches += 1
        if progress_bar is not None:
            progress_bar.update(n_batches)
    if progress_bar is not None:
        progress_bar.close()
    total_time = time.time() - start
    summary_writer.add_scalar("val/topicloss", torch.tensor(total_loss_topic).mean(), n_step)
    summary_writer.add_scalar("val/notopicloss", torch.tensor(total_loss_notopic).mean(), n_step)
    summary_writer.add_scalar('val/loss', (torch.tensor(total_loss_topic) + torch.tensor(total_loss_notopic)).mean() , n_step)
    summary_writer.add_scalar('val/time', total_time, n_step)

def train(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument('--log', type=str, default='debug')
    # parser.add_argument("--tqdm", action='store_true')

    args = parser.parse_args(cmd)
    config = yaml.load(open(args.config,'r'), Loader=yaml.Loader)
    if config["device"] == "cuda": show_gpu('GPU memory usage')
    num_epochs  = config["epochs"]
    num_training_steps = config.get("num_training_steps", None)
    model_name  = config["model"]
    cache_dir   = config["cache"]
    train_log_freq  = config["train_log_freq"]
    val_log_freq    = config["val_log_freq"]
    save_freq = config["save_freq"]
    output_dir  = Path(config["output_dir"]) / (config["tag"] + get_timestamp_str())
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(output_dir)
    print('output directory:', output_dir)
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, filename=str(output_dir / 'logs.txt'), filemode='a')
    yaml.dump(config, open(output_dir/'config.yaml', 'w'), Dumper=yaml.Dumper)

    accelerator = None
    if config["accelerate"]:
        accelerator = Accelerator(gradient_accumulation_steps=config["gradient_acc"])
        ('Accelerator: num_processes', accelerator.num_processes)
    
    device = config['device']
    PAD_TOKEN = '<|endoftext|>'
    tokenizer = GPT2Tokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir, pad_token=PAD_TOKEN)
    dataloader, val_dataloader = prepare_dataloader(config, tokenizer, accelerator=accelerator)
    print('Loading model %s'%model_name)

    modelf = GPTNeoForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    if not config["accelerate"]:
        modelf = modelf.to(device)

    optimizer, lr_scheduler = prepare_optimizer_scheduler(config, modelf)
    summary_writer = SummaryWriter(output_dir)
    n_step = 0
    if config["accelerate"]:
        modelf, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            modelf, optimizer, dataloader, val_dataloader, lr_scheduler)

    print('Starting training')
    for epoch in range(num_epochs):
        start = time.time()
        for batch in dataloader:
            if num_training_steps is not None and n_step >= num_training_steps:
                break
            with accelerator.accumulate(modelf) if accelerator is not None else nullcontext():
                modelf.train()
                batch_topic, batch_notopic = batch
                starttimer = time.time()
                loss_topic, loss_notopic, (logt, lognt) = forward_model(
                    config, batch_topic, batch_notopic, tokenizer, modelf
                )
                loss = 0.1* loss_topic + loss_notopic
                if not config["accelerate"]:
                    loss.backward()
                else:
                    accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if n_step % train_log_freq == 0:
                if config["accelerate"]:
                    all_loss_topic, all_loss_notopic = accelerator.gather_for_metrics((loss_topic, loss_notopic))
                    all_loss_topic = all_loss_topic.mean()
                    all_loss_notopic = all_loss_notopic.mean()
                else:
                    all_loss_topic, all_loss_notopic = loss_topic, loss_notopic
                all_loss_topic = all_loss_topic.detach().float()
                all_loss_notopic = all_loss_notopic.detach().float()
                log_train(config, summary_writer, n_step, all_loss_topic, all_loss_notopic, time.time()-starttimer, lr_scheduler)
            
            if n_step % save_freq == 0 and n_step > 0:
                #save_state(accelerator, modelf, output_dir / 'checkpoint_%d'%n_step)
                checkpoint.save_checkpoint(n_step, modelf, accelerator)
            #torch.cuda.empty_cache()
            n_step += 1
            if n_step % val_log_freq == 0 and n_step > 0:
                modelf.eval()
                with torch.no_grad():
                    log_val(
                        summary_writer, config, n_step, tokenizer, modelf, val_dataloader
                    )
        if num_training_steps is not None and n_step >= num_training_steps:
            break
        total_time = time.time() - start
        log_epoch(summary_writer, total_time)
    #save_state(accelerator, modelf, output_dir / 'checkpoint_%d'%n_step)
    checkpoint.save_checkpoint(n_step, modelf, accelerator)
    checkpoint.save_json()

if __name__ == "__main__":
    train()
import argparse
from pathlib import Path
import yaml
from tqdm.auto import tqdm
from transformers import get_scheduler, GPT2Tokenizer, GPTNeoForCausalLM
from lm_dataformat import Archive
from topicrm.dataloader import FinetuneDataset, CorpusLoader, ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from topicrm.losses import log_prob, kl_loss
from datetime import datetime
from accelerate import Accelerator
from datasets import Dataset
import numpy as np
import shutil
import logging
import time
import subprocess
from utils import get_timestamp_str, show_gpu

logger = logging.getLogger(__file__)

def prepare_dataloader(config, tokenizer, accelerator=None):
    topic_probs = config['probpath']
    all_data = config['alldata']
    val_topic_probs = config['val_probpath']
    val_all_data = config['val_alldata']
    topic_ids = config['topics']
    threshold = config['thresh']
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    include_set = set(config["include_set"]) if "include_set" in config else None
    exclude_set = set(config["exclude_set"]) if "exclude_set" in config else None
    if include_set is not None:
        logger.info("Including data only from sets %s"%(include_set))
    if exclude_set is not None:
        logger.info("Excluding data from sets %s"%(exclude_set))
    if config["topicdata_type"] == "CorpusLoader":
        topic_data = CorpusLoader(config["topicdata"], include=include_set, exclude=exclude_set)
    elif config["topicdata_type"] == "ConcatDataset":
        topic_data = ConcatDataset.corpus_from_dir(config["topicdata"], include=include_set, exclude=exclude_set)
    else:
        raise NotImplementedError("Not implemented %s"%(config["topicdata_type"]))
    if config["val_topicdata_type"] == "CorpusLoader":
        val_topic_data = CorpusLoader(config["val_topicdata"], include=include_set, exclude=exclude_set)
    else:
        raise NotImplementedError("Not implemented %s"%(config["val_topicdata_type"]))
    dataset = FinetuneDataset(all_data, topic_probs, topic_data,
                                topic_ids, threshold=threshold)
    val_dataset = FinetuneDataset(val_all_data, val_topic_probs, val_topic_data,
                                topic_ids, threshold=threshold)

    max_length = None
    if config["truncate_len"] is not None:
        max_length = config["truncate_len"]

    def tokenize_function(examples):
        samples1, samples2 = [*zip(*examples)]
        if accelerator is not None and accelerator.num_processes > 1:
            padding = 'max_length'
        else:
            padding = 'longest'
        outputs1 = tokenizer(samples1, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).input_ids
        outputs2 = tokenizer(samples2, return_tensors="pt", padding=padding, truncation=True, max_length=max_length).input_ids
        att_mask_1      = (outputs1      != tokenizer.pad_token_id).long()
        att_mask_2    = (outputs2    != tokenizer.pad_token_id).long()
        return (outputs1, att_mask_1), (outputs2, att_mask_2)
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

def log_train(summary_writer: SummaryWriter, n_step, loss_topic, loss_no_topic, scheduler):
    logger.info('Step {}, loss {}, topic {}, no topic {}'.format(n_step, loss_topic+loss_no_topic, loss_topic, loss_no_topic))
    summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], n_step)
    summary_writer.add_scalar('train/topicloss', loss_topic, n_step)
    summary_writer.add_scalar('train/notopicloss', loss_no_topic, n_step)
    summary_writer.add_scalar('train/loss', loss_topic+loss_no_topic, n_step)
    pct = show_gpu(f'{n_step}: ')
    for i, p in enumerate(pct):
        summary_writer.add_scalar(f'train/cuda:{i}-mem', p, n_step)

def log_epoch(summary_writer: SummaryWriter, epoch_time):
    summary_writer.add_scalar('train/time', epoch_time)

def forward_model(config, batch_topic, batch_notopic, tokenizer, modelf, model0=None):
    input_ids_topic, att_mask_topic = batch_topic
    input_ids_notopic, att_mask_notopic = batch_notopic

    if not config["accelerate"]:
        input_ids_topic = input_ids_topic.to(config['device'])
        input_ids_notopic = input_ids_notopic.to(config['device'])
        att_mask_topic = att_mask_topic.to(config['device'])
        att_mask_notopic = att_mask_notopic.to(config['device'])
    logitsf_topic   = modelf(input_ids_topic,   attention_mask=att_mask_topic.long()).logits
    logitsf_notopic = modelf(input_ids_notopic, attention_mask=att_mask_notopic.long()).logits
    
    mask_topic = att_mask_topic
    mask_notopic = att_mask_notopic

    if config["klloss"]:
        with torch.no_grad():
            logits0_notopic = model0(input_ids_notopic, attention_mask=att_mask_notopic)

        loss_topic = log_prob(logitsf_topic, input_ids_topic, mask_topic)
        loss_notopic = kl_loss(logits0_notopic.detach(), logitsf_notopic, mask_notopic)
    else:
        loss_topic = log_prob(logitsf_topic, input_ids_topic, mask_topic)
        loss_notopic = -log_prob(logitsf_notopic, input_ids_notopic, mask_notopic)

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
    summary_writer.add_scalar('val/time', total_time)

def save_state(accelerator: Accelerator, model, output_dir):
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir)
    if accelerator.is_main_process:
        model.module.config.save_pretrained(output_dir)
    # unwrapped_model = accelerator.unwrap_model(model)
    # for k, v in unwrapped_model.state_dict().items():
    #     print(k, v.shape)
    # unwrapped_model.save_pretrained(output_dir/'hfmodel', save_function=accelerator.save)

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
    use_kl      = config["klloss"]
    output_dir  = Path(config["output_dir"]) / (config["tag"] + get_timestamp_str())
    output_model_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if use_kl:
        raise NotImplementedError()
        model0 = modelf.deepcopy()
        model0.eval()

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
            with accelerator.accumulate(modelf):
                modelf.train()
                batch_topic, batch_notopic = batch
                loss_topic, loss_notopic, (logt, lognt) = forward_model(
                    config, batch_topic, batch_notopic, tokenizer, modelf,
                    model0= model0 if use_kl else None
                )
                loss = loss_topic + loss_notopic
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
                log_train(summary_writer, n_step, all_loss_topic, all_loss_notopic, lr_scheduler)
            
            if n_step % val_log_freq == 0 and n_step > 0:
                modelf.eval()
                with torch.no_grad():
                    log_val(
                        summary_writer, config, n_step, tokenizer, modelf, val_dataloader,
                        model0= model0 if use_kl else None
                    )
            n_step += 1
            torch.cuda.empty_cache()
            if num_training_steps is not None and n_step >= num_training_steps:
                break
        if num_training_steps is not None and n_step >= num_training_steps:
            break
        total_time = time.time() - start
        log_epoch(summary_writer, total_time)
        if epoch % save_freq == 0 and epoch > 0:
            save_state(accelerator, modelf, output_model_dir)
    save_state(accelerator, modelf, output_model_dir)

if __name__ == "__main__":
    train()
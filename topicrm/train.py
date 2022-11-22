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

def prepare_dataloader(config, tokenizer):
    topic_probs = config['probpath']
    all_data = config['alldata']
    val_topic_probs = config['val_probpath']
    val_all_data = config['val_alldata']
    topic_ids = config['topics']
    threshold = config['thresh']
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    if config["topicdata_type"] == "CorpusLoader":
        topic_data = CorpusLoader(config["topicdata"])
    else:
        raise NotImplementedError("Not implemented %s"%(config["topicdata_type"]))
    if config["val_topicdata_type"] == "CorpusLoader":
        val_topic_data = CorpusLoader(config["val_topicdata"])
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
        outputs1 = tokenizer(samples1, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids
        outputs2 = tokenizer(samples2, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids
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
    #scheduler = get_scheduler(
    #    config['scheduler'],
    #    optimizer,
    #    num_training_steps=config.get('num_training_steps', None), 
    #    num_warmup_steps=config.get('num_warmup_steps', None)
    #)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return optimizer, scheduler

def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def log_train(summary_writer: SummaryWriter, n_step, loss_topic, loss_no_topic, scheduler):
    print('Step {}, loss {}'.format(n_step, loss_topic+loss_no_topic))
    summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], n_step)
    summary_writer.add_scalar('train/topicloss', loss_topic, n_step)
    summary_writer.add_scalar('train/notopicloss', loss_no_topic, n_step)
    summary_writer.add_scalar('train/loss', loss_topic+loss_no_topic, n_step)

def forward_model(config, batch_topic, batch_notopic, tokenizer, modelf, model0=None):
    input_ids_topic, att_mask_topic = batch_topic
    input_ids_notopic, att_mask_notopic = batch_notopic

    if not config["accelerate"]:
        input_ids_topic = input_ids_topic.to(config['device'])
        input_ids_notopic = input_ids_notopic.to(config['device'])
        att_mask_topic = att_mask_topic.to(config['device'])
        att_mask_notopic = att_mask_notopic.to(config['device'])
    print(input_ids_topic.dtype, att_mask_topic.dtype, att_mask_topic.long().dtype, att_mask_topic.long().device)
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

    return loss_topic, loss_notopic

def log_val(summary_writer: SummaryWriter, accelerator, config, n_step, tokenizer, modelf, val_dataloader, model0=None):
    n_batches = 0
    total_loss_topic = torch.tensor((0,)) 
    total_loss_notopic = torch.tensor((0,)) 
    for batch in val_dataloader:
        batch_topic, batch_notopic = batch
        loss_topic, loss_notopic = forward_model(
            config, batch_topic, batch_notopic, tokenizer, modelf, model0=model0)
        all_loss_topic, all_loss_notopic = accelerator.gather_for_metric((loss_topic, loss_notopic))
        total_loss_topic += all_loss_topic
        total_loss_notopic += all_loss_notopic
        n_batches += 1
    
    summary_writer.add_scalar("val/topicloss", total_loss_topic / n_batches, n_step)
    summary_writer.add_scalar("val/notopicloss", total_loss_notopic / n_batches, n_step)
    summary_writer.add_scalar('val/loss', (total_loss_topic + total_loss_notopic) / n_batches, n_step)

import subprocess

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return np.array(list(map(int,result.strip().split('\n'))))
    print(query('memory.used'))
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    for i, p in enumerate(pct):
        print(msg, f'cuda:{i}. {100*p:2.1f}% ({used[i]} out of {total[i]})')    

def train(cmd=None):
    show_gpu('GPU memory usage')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # parser.add_argument("--tqdm", action='store_true')

    args = parser.parse_args(cmd)
    config = yaml.load(open(args.config,'r'), Loader=yaml.Loader)
    num_epochs  = config["epochs"]
    num_training_steps = config.get("num_training_steps", None)
    model_name  = config["model"]
    cache_dir   = config["cache"]
    train_log_freq  = config["train_log_freq"]
    val_log_freq    = config["val_log_freq"]
    use_kl      = config["klloss"]
    output_dir  = Path(config["output_dir"]) / (config["tag"] + get_timestamp_str())
    
    device = config['device']
    PAD_TOKEN = '<|endoftext|>'
    tokenizer = GPT2Tokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir, pad_token=PAD_TOKEN)
    dataloader, val_dataloader = prepare_dataloader(config, tokenizer)
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
    accelerator = None
    if config["accelerate"]:
        accelerator = Accelerator()
        modelf, optimizer, dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            modelf, optimizer, dataloader, val_dataloader, lr_scheduler)

    print('Starting training')
    for epoch in range(num_epochs):
        for batch in dataloader:
            # print(len(batch[0]),(len(batch[1])))
            modelf.train()
            batch_topic, batch_notopic = batch
            loss_topic, loss_notopic = forward_model(
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
                all_loss_topic, all_loss_notopic = accelerator.gether_metric((loss_topic, loss_notopic))
                log_train(summary_writer, n_step, all_loss_topic, all_loss_notopic, lr_scheduler)
            if n_step % val_log_freq == 0 and n_step > 0:
                modelf.eval()
                log_val(
                    summary_writer, config, n_step, tokenizer, modelf, val_dataloader,
                    model0= model0 if use_kl else None
                )
            n_step += 1
            accelerator.print(f'{n_step}')
            #del batch_topic, batch_notopic, loss_topic, loss_notopic
            #torch.cuda.empty_cache()
            show_gpu(f'{n_step}: GPU memory usage:')
            if num_training_steps is not None and n_step >= num_training_steps:
                break
        if num_training_steps is not None and n_step >= num_training_steps:
            break

if __name__ == "__main__":
    train()
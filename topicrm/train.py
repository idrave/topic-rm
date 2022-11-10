import argparse
from pathlib import Path
import yaml
from tqdm.auto import tqdm
from transformers import get_scheduler, GPT2Tokenizer, GPTNeoForCausalLM
from lm_dataformat import Archive
from dataloader import FinetuneDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from losses import log_prob, kl_loss
import datetime

def prepare_dataloader(config):
    topic_data = config['topicdata']
    topic_probs = config['probpath']
    all_data = config['alldata']
    val_topic_data = config['val_topicdata']
    val_topic_probs = config['val_probpath']
    val_all_data = config['val_alldata']
    topic_ids = config['topics']
    threshold = config['thresh']
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    dataset = FinetuneDataset(all_data, topic_probs, topic_data,
                                topic_ids, threshold=threshold)
    val_dataset = FinetuneDataset(val_all_data, val_topic_probs, val_topic_data,
                                topic_ids, threshold=threshold)
    def collate_fn(samples):
        return [zip(*samples)]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=collate_fn)
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

def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def log_train(summary_writer: SummaryWriter, n_step, loss_topic, loss_no_topic, scheduler):
    summary_writer.add_scalar('lr', scheduler.get_last_lr(), n_step)
    summary_writer.add_scalar('train/topicloss', loss_topic, n_step)
    summary_writer.add_scalar('train/notopicloss', loss_no_topic, n_step)
    summary_writer.add_scalar('train/loss', loss_topic+loss_no_topic, n_step)

def forward_model(config, batch_topic, batch_notopic, tokenizer, modelf, model0=None):
    input_ids_topic =\
        tokenizer(batch_topic, return_tensors="pt", padding=True).input_ids
    input_ids_notopic =\
        tokenizer(batch_notopic, return_tensors="pt", padding=True).input_ids
    att_mask_topic      = (input_ids_topic      != tokenizer.pad_token_id).long()
    att_mask_notopic    = (input_ids_notopic    != tokenizer.pad_token_id).long()

    logitsf_topic   = modelf(input_ids_topic,   attention_mask=att_mask_topic).logits
    logitsf_notopic = modelf(input_ids_notopic, attention_mask=att_mask_notopic).logits
    
    mask_topic = att_mask_topic
    mask_notopic = att_mask_notopic

    if config["klloss"]:
        with torch.no_grad():
            logits0_notopic = model0(input_ids_notopic, attention_mask=att_mask_notopic)

        loss_topic = log_prob(logitsf_topic, input_ids_topic, mask_topic)
        loss_notopic = kl_loss(logits0_notopic.detach(), logitsf_notopic, mask_notopic)
    else:
        loss_topic = log_prob(logitsf_topic, input_ids_topic, mask_topic)
        loss_notopic = log_prob(logitsf_notopic, input_ids_notopic, mask_notopic)

    return loss_topic, loss_notopic

def log_val(summary_writer: SummaryWriter, config, n_step, tokenizer, model0, modelf, val_dataloader):
    n_batches = 0
    total_loss_topic = torch.tensor((0,)) 
    total_loss_notopic = torch.tensor((0,)) 
    for batch in val_dataloader:
        batch_topic, batch_notopic = batch
        loss_topic, loss_notopic = forward_model(
            config, batch_topic, batch_notopic, tokenizer, modelf, model0=model0)
        total_loss_topic += loss_topic
        total_loss_notopic += loss_notopic
        n_batches += 1
    
    summary_writer.add_scalar("val/topicloss", total_loss_topic / n_batches, n_step)
    summary_writer.add_scalar("val/notopicloss", total_loss_notopic / n_batches, n_step)
    summary_writer.add_scalar('val/loss', (total_loss_topic + total_loss_notopic) / n_batches, n_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader())
    num_epochs  = config["epochs"]
    model_name  = config["model"]
    cache_dir   = config["cache"]
    train_log_freq  = config["train_log_freq"]
    val_log_freq    = config["val_log_freq"]
    use_kl      = config["klloss"]
    output_dir  = Path(config["output_dir"]) / config["tag"] + get_timestamp_str()
    
    device = "cuda"
    dataloader, val_dataloader = prepare_dataloader(config)
    PAD_TOKEN = '<|endoftext|>'
    tokenizer = GPT2Tokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir, pad_token=PAD_TOKEN)
    
    modelf = GPTNeoForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    if use_kl:
        model0 = modelf.deepcopy()
        model0.eval()

    #progress_bar = tqdm(range(num_training_steps))
    modelf.train()
    optimizer, lr_scheduler = prepare_optimizer_scheduler(config, modelf)
    summary_writer = SummaryWriter(output_dir)
    n_step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            modelf.train()
            batch_topic, batch_notopic = batch
            loss_topic, loss_notopic = forward_model(
                config, batch_topic, batch_notopic, tokenizer, modelf,
                model0= model0 if use_kl else None
            )
            loss = loss_topic + loss_notopic
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            if n_step % train_log_freq == 0:
                log_train(summary_writer, n_step, loss_topic, loss_notopic, lr_scheduler)
            if n_step % val_log_freq == 0:
                modelf.eval()
                log_val(
                    summary_writer, config, n_step, tokenizer, modelf, val_dataloader,
                    model0= model0 if use_kl else None
                )
            n_step += 1
            # progress_bar.update(1)

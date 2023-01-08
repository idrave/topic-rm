import argparse
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import yaml
from yaml import Loader, Dumper
from lm_dataformat import Archive
from pathlib import Path
import time
from datetime import datetime
from dataloader import CorpusLoader
import random
import torch

random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--config', required=True)
parser.add_argument('--jobid', default=None)

args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=Loader)

model_name = config['model']
cache = config['cache']
tok_name = config['tokenizer'] if config.get('tokenizer', None) is not None else model_name

tokenizer = GPT2Tokenizer.from_pretrained(tok_name, cache_dir=cache)
model = GPTNeoForCausalLM.from_pretrained(model_name, cache_dir=cache).to('cuda')

def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

n = config['num_samples']
batch = config['batch_size']
max_length = config['max_length']
top_p = config['top_p']
temp = config['temp']
top_k = config['top_k']
init_phase = config['init_phase']
init_top_p = config['init_top_p']
init_temp = config['init_temp']
init_top_k = config['init_top_k']
prompt_corpus = config.get('prompt_corpus', None)
corpus_offset = config.get('corpus_offset', None)
if args.jobid is None:
    output = Path(config['output'])/('%s_%s' % (config['id'], get_timestamp_str()))
else:
    output = Path(config['output'])/('%s_%s_%s' % (config['id'], get_timestamp_str(), args.jobid))
    if prompt_corpus is not None:
        prompt_corpus = prompt_corpus.format(id=args.jobid)

#workers = config['workers']

print(output)
output.mkdir(parents=True, exist_ok=True)
archive = Archive(str(output))

start = time.time()
generate_time = 0.
add_time = 0.

eot = tokenizer.encode("<|endoftext|>", return_tensors="pt")[0].to('cuda')
input_ids = eot.repeat(batch, 1)

corpus_iter = None
if init_phase is not None and prompt_corpus is not None:
    if corpus_offset is None:
        corpus_offset = 0
    corpus_iter = iter(CorpusLoader(prompt_corpus).iter_offset(corpus_offset))
    corpus_index = corpus_offset

log_dict = {}
# TODO: could fail if run out of data
for i in range((n+batch-1)//batch):
    start_gen = time.time()
    prefixes = None
    if init_phase is not None:
        if prompt_corpus is not None:
            prompts = []
            while len(prompts) < batch:
                text = next(corpus_iter)
                corpus_index += 1
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=init_phase).input_ids
                if tokens.shape[1] == init_phase:
                    prompts.append(tokens)
            warmup = torch.concat(prompts, dim=0).to('cuda')
            prefixes = [tokenizer.decode(sentence) for sentence in warmup]
        else:
            attention_mask = (input_ids != tokenizer.eos_token_id).long()
            warmup = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, do_sample=True, min_length=init_phase,
                                    max_length=init_phase, temperature=init_temp,
                                    top_p=init_top_p, top_k=init_top_k)[:,1:] # exclude first padding token
        attention_mask = (warmup != tokenizer.eos_token_id).long()
        sample_output = model.generate(warmup, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_length=max_length,
                                temperature=temp, top_p=top_p, top_k=top_k)
    else:
        attention_mask = (input_ids != tokenizer.eos_token_id).long()
        sample_output = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_length=max_length+1, # add one to count for padding token
                                temperature=temp, top_p=top_p, top_k=top_k)
    start_decode = time.time()
    generate_time += start_decode - start_gen
    for i, data in enumerate(sample_output):
        data = tokenizer.decode(data, skip_special_tokens=True)
        meta = {
            'prefix': None if prefixes is None else prefixes[i]
        }
        archive.add_data(data, meta=meta)
    add_time += time.time() - start_decode

start_commit = time.time()
archive.commit()
yaml.dump(config, open(output/'config.yaml', 'w'), Dumper=Dumper)

if init_phase is not None and prompt_corpus is not None:
    open(output/'corpus_index', 'w').write('%d' % corpus_index)

print('Commit', time.time() - start_commit)
print('Generation', generate_time)
print('Add/decode', add_time)
print('Time', time.time() - start)

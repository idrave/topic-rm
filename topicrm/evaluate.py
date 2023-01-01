import time
import argparse
import yaml
import json
from typing import List, Union
from lm_eval.evaluator import simple_evaluate
from pathlib import Path
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from utils import get_timestamp_str, TaskGroup, get_task, show_gpu
import logging

def eval(model: str, tokenizer: str, tasks: List[Union[str,TaskGroup]], output_dir, batch_size=4, device='0'):
    task_names = set()
    for task in tasks:
        if isinstance(task, TaskGroup):
            task_names = task_names.union(task.get_tasks())
        else:
            task_names.add(task)
    results = simple_evaluate(
        'gpt2',
        f'pretrained={model},tokenizer={tokenizer}',
        tasks=task_names,
        batch_size=batch_size,
        device=device,
    )
    for task in tasks:
        if isinstance(task, TaskGroup):
            task.add_results(results["results"])
    json.dump(results, open(output_dir/'results.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    logging.basicConfig()
    config = yaml.load(open(args.config,'r'),Loader=yaml.Loader)
    if args.model is not None:
        config['model'] = args.model
    
    output_dir = Path(config['output'])/(config["tag"]+'_'+get_timestamp_str())
    output_dir.mkdir()
    yaml.dump(config, open(str(output_dir/'config.yaml'), 'w'), Dumper=yaml.Dumper)
    start = time.time()
    eval(
        config['model'],
        config['tokenizer'],
        [get_task(s) for s in config['tasks']],
        output_dir,
        batch_size=config['batch_size']
    )
    print('Time:', time.time()-start)
    show_gpu('Eval:')
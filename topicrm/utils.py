import threading
from queue import Queue
from datetime import datetime
import lm_eval.tasks
import logging
import subprocess
import numpy as np
from pytablewriter import LatexTableWriter
import csv

logger = logging.getLogger(__file__)

def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

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
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    for i, p in enumerate(pct):
        logger.debug(msg+f'Memory cuda:{i}. {100*p:2.1f}% ({used[i]} out of {total[i]})')    
    return pct

class AverageMetric:
    def __init__(self, task, metrics) -> None:
        assert len(task.tasks) == len(metrics)
        self.name = 'avg'
        self.task = task
        self.metrics = metrics

    def aggregate(self, results):
        total = 0.
        for task, metric in zip(self.task.get_tasks(), self.metrics):
            total += results[task][metric]
        return total / len(self.metrics)

class TaskGroup:
    def __init__(self, name, tasks, metrics=None) -> None:
        self.name = name
        self.tasks = tasks
        metrics = metrics if metrics is not None else []
        self.metrics = [metric(self, *args) for metric, args in metrics]

    def get_tasks(self):
        return self.tasks
    
    def add_results(self, results):
        results[self.name] = {}
        for metric in self.metrics:
            results[self.name][metric.name] = metric.aggregate(results)
    
    def download(self):
        for task in self.tasks:
            lm_eval.tasks.get_task(task)().download()
        

TASKS_GROUPS = {
    'superglue': TaskGroup('superglue',
        ['cb', 'copa', 'multirc', 'rte', 'wic', 'wsc', 'boolq', 'record'],
        [(AverageMetric, [['acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'acc', 'f1'],])]),
    'gpt-neo': TaskGroup('gpt-neo', 
        ['lambada_openai', 'wikitext', 'winogrande', 'hellaswag'])
}

def get_task(task_name):
    if task_name in TASKS_GROUPS:
        return TASKS_GROUPS[task_name]
    return task_name

def _submit_queue(q: Queue, items, num_consumers):
    for item in items:
        q.put(item)
    for i in range(num_consumers):
        q.put(None)

def _process_queue(in_q: Queue, out_q: Queue, task):
    item = in_q.get()
    while item is not None:
        out_q.put(task(item))
        item = in_q.get()
    out_q.put(None)

def thread_pool_iter(items, task, num_workers, maxsize=100):
    input_queue = Queue(maxsize=maxsize)
    output_queue = Queue(maxsize=maxsize)
    producer = threading.Thread(target=_submit_queue, args=(input_queue, items, num_workers), daemon=True)
    workers = [threading.Thread(target=_process_queue, args=(input_queue, output_queue, task), daemon=True) for _ in range(num_workers)]
    producer.start()
    for worker in workers:
        worker.start()
    finished = 0
    while finished < num_workers:
        result = output_queue.get()
        if result is None:
            finished += 1
        else:
            yield result
    producer.join()
    for worker in workers:
        worker.join()

def generator_to_queue(foo):
    def wrapper(*args, **kwargs):
        # Create the queue and start the generator thread
        q = Queue(maxsize=100)
        t = threading.Thread(target=_enqueue_generator, args=(foo, q, args, kwargs), daemon=True)
        t.start()

        # Yield values from the queue
        item = None
        while item != StopIteration:
            item = q.get()
            if item == StopIteration:
                t.join()
            yield item
    return wrapper

def _enqueue_generator(gen, q, args, kwargs):
    # Iterate through the generator and put the values in the queue
    for value in gen(*args, **kwargs):
        q.put(value)
    q.put(StopIteration)

def make_table(names, results, output_file, decimals=2):
    writer = csv.writer(open(output_file, 'w'), delimiter='|')

    task_metric = []
    for task, metrics in results[0]["results"].items():
        for metric, _ in metrics.items():
            task_metric.append((task, metric))
    headers = ['name'] + [f'{task} {metric}'.replace('_',' ') for (task, metric) in task_metric]
    writer.writerow(headers)
    for name, result in zip(names, results):
        row = [name]
        for task, metric in task_metric:
            value = result["results"][task][metric]
            try:
                value = (f'%.{decimals}f') % value
            except ValueError:
                pass
            row.append(value)
        writer.writerow(row)


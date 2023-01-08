from datetime import datetime
import logging
import subprocess
import numpy as np
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


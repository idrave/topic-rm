import lm_eval.tasks

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
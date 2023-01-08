import threading
from queue import Queue

def _submit_queue(q: Queue, items, num_consumers):
    try:
        for item in items:
            q.put(item)
        for i in range(num_consumers):
            q.put(None)
    except Exception as e:
        for i in range(num_consumers):
            q.put(e)

def _process_queue(in_q: Queue, out_q: Queue, task):
    try:
        item = in_q.get()
        while item is not None:
            if isinstance(item, Exception):
                raise e
            out_q.put(task(item))
            item = in_q.get()
        out_q.put(None)
    except Exception as e:
        out_q.put(e)

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
        elif isinstance(result, Exception):
            raise result
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
            elif isinstance(item, Exception):
                raise item
            yield item
    return wrapper

def _enqueue_generator(gen, q, args, kwargs):
    # Iterate through the generator and put the values in the queue
    try:
        for value in gen(*args, **kwargs):
            q.put(value)
        q.put(StopIteration)
    except Exception as e:
        q.put(e)
import threading
from queue import Queue

from execution.runner import Runner


class IterationsWorker(threading.Thread):

    def __init__(self, queue: Queue, queue_result: Queue, runner: Runner, start_time: float):
        threading.Thread.__init__(self)
        self.queue = queue
        self.queue_result = queue_result
        self.runner = runner
        self.start_time = start_time

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            current_iteration, search_suffix, current_env_variables = self.queue.get()
            env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                current_iteration=current_iteration,
                search_suffix=search_suffix,
                current_env_variables=current_env_variables,
                _start_time=self.start_time,
            )

            self.queue_result.put_nowait((env_predicate_pair, execution_time, regression_time))
            self.queue.task_done()

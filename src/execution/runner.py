import datetime
import multiprocessing
import threading
import time
from queue import Queue
from typing import Tuple

import numpy as np

from abstract_agent import AbstractAgent
from algo.env_predicate_pair import EnvPredicatePair
from envs.env_variables import EnvVariables
from execution.execution_result import ExecutionResult
from log import Log


def execute_train(agent: AbstractAgent,
                  current_iteration: int,
                  search_suffix: str,
                  current_env_variables: EnvVariables,
                  _start_time: float,
                  random_search: bool = False) -> Tuple[EnvPredicatePair, float, float]:

    env_predicate_pairs = []
    communication_queue = Queue()
    logger = Log("execute_train")

    # agent.train sets seed globally (for tf, np and random)
    seed = np.random.randint(2 ** 32 - 1)

    # order of argument matters in the args param; must match the order of args in the train method of agent
    thread = threading.Thread(
        target=agent.train,
        args=(seed, communication_queue, current_iteration, search_suffix, current_env_variables, random_search,),
    )
    thread.start()
    sum_training_time = 0.0
    sum_regression_time = 0.0
    while True:
        data: ExecutionResult = communication_queue.get()  # blocking code
        logger.debug(
            "Env: {}, evaluates to {}".format(current_env_variables.get_params_string(),
                                              data.is_adequate_performance(), )
        )
        logger.debug("Info: {}".format(data.get_info()))
        env_predicate_pairs.append(
            EnvPredicatePair(
                env_variables=current_env_variables,
                predicate=data.is_adequate_performance(),
                regression=data.is_regression(),
                execution_info=data.get_info(),
                model_dirs=[search_suffix]
            )
        )
        sum_regression_time += data.get_regression_time()
        sum_training_time += data.get_training_time()
        if data.is_task_completed():
            break

    while thread.is_alive():
        time.sleep(1.0)

    logger.info("TIME ELAPSED: {}".format(str(datetime.timedelta(seconds=(time.time() - _start_time)))))

    return env_predicate_pairs[-1], sum_training_time, sum_regression_time


class ProbabilityEstimationWorker(threading.Thread):

    def __init__(self, queue: Queue, queue_result: Queue, agent: AbstractAgent, start_time: float, random_search: bool):
        threading.Thread.__init__(self)
        self.queue = queue
        self.queue_result = queue_result
        self.agent = agent
        self.start_time = start_time
        self.random_search = random_search

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            current_iteration, search_suffix, current_env_variables = self.queue.get()
            env_predicate_pair, execution_time, regression_time = \
                execute_train(agent=self.agent, current_iteration=current_iteration, search_suffix=search_suffix,
                              current_env_variables=current_env_variables, _start_time=self.start_time,
                              random_search=self.random_search)
            self.queue_result.put_nowait((env_predicate_pair, execution_time, regression_time))
            self.queue.task_done()


class Runner:
    def __init__(self, agent: AbstractAgent, runs_for_probability_estimation: int = 1):
        self.logger = Log("Runner")
        self.agent = agent
        self.runs_for_probability_estimation = runs_for_probability_estimation

    def execute_train(
        self, current_iteration: int, search_suffix: str, current_env_variables: EnvVariables, _start_time: float,
            random_search: bool = False
    ) -> Tuple[EnvPredicatePair, float, float]:
        if self.runs_for_probability_estimation == 1:
            env_predicate_pair, execution_time, regression_time = execute_train(
                agent=self.agent,
                current_iteration=current_iteration,
                search_suffix=search_suffix,
                current_env_variables=current_env_variables,
                _start_time=_start_time,
                random_search=random_search,
            )
            self.logger.debug('--------------------------------------------------: end runner execution')
            return env_predicate_pair, execution_time, regression_time
        execution_start_time = time.time()

        num_of_cpus = multiprocessing.cpu_count()
        num_of_processes_to_spawn = self.runs_for_probability_estimation \
            if num_of_cpus >= self.runs_for_probability_estimation else num_of_cpus - 1
        self.logger.debug('num of processes to spawn: {}'.format(num_of_processes_to_spawn))

        search_suffixes = [search_suffix + '_run_' + str(i) for i in range(self.runs_for_probability_estimation)]

        queue = Queue()
        queue_result = Queue()
        # Create worker threads
        for _ in range(num_of_processes_to_spawn):
            worker = ProbabilityEstimationWorker(
                queue=queue, queue_result=queue_result, agent=self.agent, start_time=_start_time,
                random_search=random_search
            )
            # Setting daemon to True will let the main thread exit even though the workers are blocking
            worker.daemon = True
            worker.start()
        # Put the tasks into the queue as a tuple
        for search_suffix in search_suffixes:
            work_to_pass = (current_iteration, search_suffix, current_env_variables)
            queue.put(work_to_pass)
        # Causes the main thread to wait for the queue to finish processing all the tasks
        queue.join()

        env_predicate_pairs = []
        execution_times = []
        regression_times = []
        while not queue_result.empty():
            env_predicate_pair, execution_time, regression_time = queue_result.get_nowait()
            env_predicate_pairs.append(env_predicate_pair)
            execution_times.append(execution_time)
            regression_times.append(regression_time)

        execution_end_time = time.time()
        # execution_time = execution_end_time - execution_start_time
        execution_time = np.asarray(execution_times).sum()
        regression_time = np.asarray(regression_times).sum()
        adequate_performance_list = []
        regression_list = []

        for env_predicate_pair in env_predicate_pairs:
            adequate_performance_list.append(env_predicate_pair.is_predicate())
            if env_predicate_pair.is_predicate():
                regression_list.append(env_predicate_pair.is_regression())

        env_predicate_pair = EnvPredicatePair(
            env_variables=current_env_variables,
            probability_estimation_runs=adequate_performance_list,
            regression_estimation_runs=regression_list,
            model_dirs=search_suffixes
        )
        self.logger.debug('--------------------------------------------------: end runner execution')
        return env_predicate_pair, execution_time, regression_time

    def execute_test_with_callback(self, current_env_variables: EnvVariables, n_eval_episodes: int = None)\
            -> EnvPredicatePair:
        seed = np.random.randint(2 ** 32 - 1)
        return self.agent.test_with_callback(seed=seed, env_variables=current_env_variables,
                                             n_eval_episodes=n_eval_episodes)

    def execute_test_without_callback(self, n_eval_episodes: int, model_path: str) -> Tuple[float, float]:
        seed = np.random.randint(2 ** 32 - 1)
        return self.agent.test_without_callback(seed=seed, n_eval_episodes=n_eval_episodes, model_path=model_path)

    def execute_train_without_evaluation(self, current_iteration: int,
                                         current_env_variables: EnvVariables,
                                         search_suffix: str = "1") -> None:
        # agent.train sets seed globally (for tf, np and random)
        seed = np.random.randint(2 ** 32 - 1)
        self.agent.train(seed=seed, current_iteration=current_iteration, search_suffix=search_suffix,
                         env_variables=current_env_variables)

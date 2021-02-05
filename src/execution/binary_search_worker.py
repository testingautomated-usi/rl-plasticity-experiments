import threading
from queue import Queue

from algo.get_binary_search_candidate import get_binary_search_candidate
from envs.env_variables import EnvVariables
from execution.runner import Runner
import copy

from log import Log
from utilities import norm
import time
import random


class BinarySearchWorker(threading.Thread):

    def __init__(
            self,
            queue: Queue,
            queue_result: Queue,
            runner: Runner,
            start_time: float,
            init_env_variables: EnvVariables,
            param_names,
            param_names_string: str,
            algo_name: str,
            env_name: str,
            binary_search_epsilon: float,
            discrete_action_space: bool,
    ):
        threading.Thread.__init__(self)
        self.queue = queue
        self.queue_result = queue_result
        self.runner = runner
        self.start_time = start_time
        self.algo_name = algo_name
        self.env_name = env_name
        self.logger = Log('binary_search_worker')
        self.binary_search_epsilon = binary_search_epsilon
        self.init_env_variables = init_env_variables
        self.param_names = param_names
        self.param_names_string = param_names_string
        self.discrete_action_space = discrete_action_space

    def run(self):

        while True:
            # Get the work from the queue and expand the tuple
            current_env_variables, current_iteration, buffer_env_predicate_pairs = self.queue.get()
            self.logger.info('Env variables for binary search: {}, Iteration: {}'.format(
                        current_env_variables.get_params_string(), current_iteration))

            t_env_variables = copy.deepcopy(self.init_env_variables)
            f_env_variables = copy.deepcopy(current_env_variables)
            binary_search_counter = 0
            max_binary_search_iterations = 20

            search_suffix = "binary_search_"
            if self.param_names:
                search_suffix += self.param_names_string + "_"
            search_suffix += str(current_iteration) + "_" + str(binary_search_counter)

            dist = norm(env_vars_1=f_env_variables, env_vars_2=t_env_variables) / norm(env_vars_1=t_env_variables)

            execution_times = []
            regression_times = []
            env_predicate_pairs = []

            while dist > self.binary_search_epsilon:
                new_env_variables = get_binary_search_candidate(
                    t_env_variables=t_env_variables,
                    f_env_variables=f_env_variables,
                    algo_name=self.algo_name,
                    env_name=self.env_name,
                    param_names=self.param_names,
                    discrete_action_space=self.discrete_action_space,
                    buffer_env_predicate_pairs=buffer_env_predicate_pairs
                )

                self.logger.debug("New env after binary search: {}".format(new_env_variables.get_params_string()))
                env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                    current_iteration=current_iteration,
                    search_suffix=search_suffix,
                    current_env_variables=new_env_variables,
                    _start_time=self.start_time,
                )

                execution_times.append(execution_time)
                regression_times.append(regression_time)
                env_predicate_pairs.append(env_predicate_pair)
                if env_predicate_pair.is_predicate():
                    self.logger.debug(
                        "New t_env found: {}".format(env_predicate_pair.get_env_variables().get_params_string()))
                    t_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                else:
                    self.logger.debug(
                        "New f_env found: {}".format(env_predicate_pair.get_env_variables().get_params_string()))
                    f_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())

                dist = norm(env_vars_1=f_env_variables, env_vars_2=t_env_variables) / norm(env_vars_1=t_env_variables)

                if binary_search_counter == max_binary_search_iterations:
                    break

                binary_search_counter += 1

                search_suffix = "binary_search_"
                if self.param_names:
                    search_suffix += self.param_names_string + "_"
                search_suffix += str(current_iteration) + "_" + str(binary_search_counter)

            self.logger.info('dist {} <= binary_search_epsilon {}'.format(dist, self.binary_search_epsilon))
            self.queue_result.put_nowait((t_env_variables, f_env_variables,
                                          env_predicate_pairs, max_binary_search_iterations,
                                          binary_search_counter, current_iteration, execution_times,
                                          regression_times))
            self.queue.task_done()

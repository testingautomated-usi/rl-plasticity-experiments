import copy
import datetime
import glob
import itertools
import multiprocessing
import os
import random
import shutil
import time
from queue import Queue
from typing import List, Tuple, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from abstract_agent import AbstractAgent
from agent_stub import AgentStub
from algo.archive import (Archive, compute_dist, is_frontier_pair,
                          read_saved_archive)
from algo.env_exec_details import EnvExecDetails
from algo.env_predicate_pair import (BufferEnvPredicatePairs, EnvPredicatePair,
                                     read_saved_buffer)
from algo.exp_search_exhausted_error import ExpSearchExhaustedError
from algo.get_binary_search_candidate import get_binary_search_candidate
from algo.search_utils import (BufferExecutionsSkipped, ExecutionSkipped,
                               read_saved_buffer_executions_skipped)
from algo.time_elapsed_util import save_time_elapsed
from env_utils import instantiate_env_variables, standardize_env_name
from envs.env_variables import EnvVariables
from execution.binary_search_worker import BinarySearchWorker
from execution.iterations_worker import IterationsWorker
from execution.runner import Runner
from log import Log
from send_email import MonitorProgress
from utilities import (HOME, NUM_OF_THREADS, PREFIX_DIR_MODELS_SAVE,
                       get_result_file_iteration_number, norm)


def _set_random_seed():
    seed = np.random.randint(2 ** 32 - 1)
    random.seed(seed)


def _translate_to_values(env_variables: EnvVariables) -> List[float]:
    result = []
    for param in env_variables.get_params():
        result.append(param.get_current_value())
    return result


class AlphaTest:
    def __init__(
        self,
        agent: AbstractAgent,
        algo_name: str,
        env_name: str,
        tb_log_name: str,
        continue_learning_suffix: str,
        env_variables: EnvVariables,
        num_iterations: int = 20,
        exp_search_guidance: bool = False,
        binary_search_guidance: bool = False,
        binary_search_epsilon: float = 0.05,
        decision_tree_guidance: bool = False,
        only_exp_search: bool = False,
        buffer_file: str = None,
        archive_file: str = None,
        executions_skipped_file: str = None,
        param_names=None,
        runs_for_probability_estimation: int = 1,
        stop_at_min_max_num_iterations: bool = False,
        parallelize_search: bool = False,
        monitor_search_every: int = -1,
        start_search_time: float = None,
        starting_progress_report_number: int = 0,
        stop_at_first_iteration: bool = False,
        determine_multipliers: bool = False,
        model_suffix: str = None,
        exp_suffix: str = None,
    ):
        assert agent, "agent should have a value: {}".format(agent)
        assert algo_name, "algo_name should have a value: {}".format(algo_name)
        assert env_name, "env_name should have a value: {}".format(env_name)

        self.agent = agent
        self.start_time = time.time()
        self.logger = Log("AlphaTest")
        self.num_iterations = num_iterations
        self.previous_num_iterations = None
        self.param_names = param_names
        self.param_names_string = None
        self.parallelize_search = parallelize_search
        self.env_name = env_name
        self.algo_name = algo_name
        self.stop_at_first_iteration = stop_at_first_iteration
        self.runs_for_probability_estimation = runs_for_probability_estimation
        self.model_suffix = model_suffix
        self.exp_suffix = exp_suffix

        if param_names:
            self.param_names_string = "_".join(param_names)

        self.all_params = env_variables.instantiate_env()
        self.stop_at_min_max_num_iterations = stop_at_min_max_num_iterations

        if param_names:
            self.logger.info("Param names: {}".format(param_names))

        self.runner = Runner(agent=self.agent, runs_for_probability_estimation=self.runs_for_probability_estimation,)
        self.init_env_variables = env_variables

        if not determine_multipliers:

            # TODO: refactor buffer restoring in abstract class extended by search algo
            #  (for now only random search and alphatest)
            if buffer_file:
                previously_saved_buffer = read_saved_buffer(buffer_file=buffer_file)
                index_last_slash = buffer_file.rindex("/")

                self.algo_save_dir = buffer_file[:index_last_slash]
                self.logger.debug("Algo save dir from restored execution: {}".format(self.algo_save_dir))
                self.buffer_env_predicate_pairs = BufferEnvPredicatePairs(save_dir=self.algo_save_dir)
                self.buffer_executions_skipped = BufferExecutionsSkipped(save_dir=self.algo_save_dir)
                self.archive = Archive(save_dir=self.algo_save_dir, epsilon=binary_search_epsilon)

                # restore buffer
                for buffer_item in previously_saved_buffer:
                    previous_env_variables = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params["discrete_action_space"],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=buffer_item.get_env_values(),
                    )
                    self.buffer_env_predicate_pairs.append(
                        EnvPredicatePair(
                            env_variables=previous_env_variables,
                            pass_probability=buffer_item.get_pass_probability(),
                            predicate=buffer_item.is_predicate(),
                            regression_probability=buffer_item.get_regression_probability(),
                            probability_estimation_runs=buffer_item.get_probability_estimation_runs(),
                            regression_estimation_runs=buffer_item.get_regression_estimation_runs(),
                            model_dirs=buffer_item.get_model_dirs(),
                        )
                    )
                assert archive_file, (
                    "when buffer file is available so needs to be the archive file to " "restore a previous execution"
                )
                assert executions_skipped_file, (
                    "when buffer file is available so needs to be the executions skipped file to "
                    "restore a previous execution"
                )
                try:
                    previous_num_iterations_buffer = get_result_file_iteration_number(filename=buffer_file)
                    previous_num_iterations_archive = get_result_file_iteration_number(filename=archive_file)
                    previous_num_iterations_executions_skipped = get_result_file_iteration_number(filename=archive_file)
                    assert (
                        previous_num_iterations_buffer == previous_num_iterations_archive
                    ), "The two nums must coincide: {}, {}".format(
                        previous_num_iterations_buffer, previous_num_iterations_archive
                    )
                    assert (
                        previous_num_iterations_buffer == previous_num_iterations_executions_skipped
                    ), "The two nums must coincide: {}, {}".format(
                        previous_num_iterations_buffer, previous_num_iterations_executions_skipped
                    )
                    assert (
                        previous_num_iterations_archive == previous_num_iterations_executions_skipped
                    ), "The two nums must coincide: {}, {}".format(
                        previous_num_iterations_archive, previous_num_iterations_executions_skipped
                    )
                    previous_num_iterations = previous_num_iterations_buffer + 1
                except ValueError as e:
                    raise ValueError(e)

                self.num_iterations -= previous_num_iterations
                self.previous_num_iterations = previous_num_iterations
                self.logger.info(
                    "Restore previous execution of {} iterations. New num iterations: {}".format(
                        previous_num_iterations, self.num_iterations
                    )
                )

                # restore archive
                previously_saved_archive = read_saved_archive(archive_file=archive_file)
                t_env_variables = None
                f_env_variables = None
                for env_values, predicate in previously_saved_archive:
                    previous_env_variables = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params["discrete_action_space"],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=env_values,
                    )
                    if predicate:
                        t_env_variables = previous_env_variables
                    else:
                        f_env_variables = previous_env_variables

                    if t_env_variables and f_env_variables:
                        self.archive.append(t_env_variables=t_env_variables, f_env_variables=f_env_variables)
                        t_env_variables = None
                        f_env_variables = None

                # restore executions skipped
                previously_saved_executions_skipped = read_saved_buffer_executions_skipped(
                    buffer_executions_skipped_file=executions_skipped_file
                )
                for buffer_executions_skipped_item in previously_saved_executions_skipped:
                    previous_env_variables_skipped = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params["discrete_action_space"],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=buffer_executions_skipped_item.env_values_skipped,
                    )
                    env_predicate_pair_skipped = EnvPredicatePair(
                        env_variables=previous_env_variables_skipped, predicate=buffer_executions_skipped_item.predicate
                    )
                    previous_env_variables_executed = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params["discrete_action_space"],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=buffer_executions_skipped_item.env_values_executed,
                    )
                    env_predicate_pair_executed = EnvPredicatePair(
                        env_variables=previous_env_variables_executed, predicate=buffer_executions_skipped_item.predicate
                    )
                    self.buffer_executions_skipped.append(
                        ExecutionSkipped(
                            env_predicate_pair_skipped=env_predicate_pair_skipped,
                            env_predicate_pair_executed=env_predicate_pair_executed,
                            search_component=buffer_executions_skipped_item.search_component,
                        )
                    )
            else:
                attempt = 0

                suffix = "n_iterations_"
                if self.model_suffix:
                    suffix += self.model_suffix + "_"
                if self.param_names:
                    suffix += self.param_names_string + "_"
                if self.exp_suffix:
                    suffix += self.exp_suffix + "_"
                suffix += str(num_iterations)

                algo_save_dir = os.path.abspath(
                    HOME + "/alphatest/" + env_name + "/" + algo_name + "/" + suffix + "_" + str(attempt)
                )
                _algo_save_dir = algo_save_dir
                while os.path.exists(_algo_save_dir):
                    attempt += 1
                    _algo_save_dir = algo_save_dir[:-1] + str(attempt)
                self.algo_save_dir = _algo_save_dir
                if not self.stop_at_min_max_num_iterations:
                    os.makedirs(self.algo_save_dir)
                self.buffer_env_predicate_pairs = BufferEnvPredicatePairs(save_dir=self.algo_save_dir)
                self.buffer_executions_skipped = BufferExecutionsSkipped(save_dir=self.algo_save_dir)
                self.archive = Archive(save_dir=self.algo_save_dir, epsilon=binary_search_epsilon)

            self.tb_log_name = tb_log_name
            self.continue_learning_suffix = continue_learning_suffix
            self.binary_search_epsilon = binary_search_epsilon
            self.exp_search_guidance = exp_search_guidance
            self.binary_search_guidance = binary_search_guidance
            self.decision_tree_guidance = decision_tree_guidance
            self.only_exp_search = only_exp_search
            self.dt = DecisionTreeClassifier()

            self.buffer_file = buffer_file
            self.archive_file = buffer_file

            if not buffer_file:
                # assuming initial env_variables satisfies the predicate of adequate performance
                if self.runs_for_probability_estimation:
                    env_predicate_pair = EnvPredicatePair(
                        env_variables=self.init_env_variables,
                        predicate=True,
                        probability_estimation_runs=[True] * self.runs_for_probability_estimation,
                    )
                else:
                    env_predicate_pair = EnvPredicatePair(env_variables=self.init_env_variables, predicate=True)
                self.buffer_env_predicate_pairs.append(env_predicate_pair)

            if self.decision_tree_guidance:
                self._train_decision_tree()

            self.max_num_iterations = None
            self.min_num_iterations = None

            current_env_variables = copy.deepcopy(self.init_env_variables)
            possible_params_dict = dict()
            for index in range(len(current_env_variables.get_params())):
                param = current_env_variables.get_params()[index]
                possible_params_dict[param.get_name()] = []
                possible_params_dict[param.get_name()].append(param.get_current_value())
                while True:
                    mutated = param.mutate()
                    if not mutated:
                        # limit (high or low) reached
                        break
                    possible_params_dict[param.get_name()].append(param.get_current_value())

            # discard first element since it is the original env
            # TODO: to optimize; remove list and use generator to discard first value
            possible_envs = list(itertools.product(*list(possible_params_dict.values())))[1:]
            possible_envs_dict = dict()
            count = 0
            for env_values in possible_envs:
                diff = np.array(env_values) - np.array(self.init_env_variables.get_values())
                num_value_changed_pos = np.sum(diff > 0)
                num_value_changed_neg = np.sum(diff < 0)
                num_value_changed = num_value_changed_pos + num_value_changed_neg
                assert num_value_changed != 0, (
                    "By applying multipliers we should have changed at least one value wrt the"
                    " original environment. Here we changed none: {}, {}".format(
                        env_values, self.init_env_variables.get_values()
                    )
                )
                if num_value_changed not in possible_envs_dict:
                    possible_envs_dict[num_value_changed] = []
                if num_value_changed == 1:
                    param_changed_index = list(np.where(diff > 0)[0])
                    if len(param_changed_index) == 0:
                        param_changed_index = list(np.where(diff < 0)[0])
                    assert len(param_changed_index) == 1, "Only one value must be > 0 or < 0. Found: {}, Diff: {}".format(
                        param_changed_index, diff
                    )
                    per_drop = self.init_env_variables.get_param(index=param_changed_index[0]).get_percentage_drop()
                else:
                    per_drop = None

                env_values_evaluated = False
                for env_predicate_pair in self.buffer_env_predicate_pairs.get_buffer():
                    if np.allclose(list(env_values), env_predicate_pair.get_env_variables().get_values()):
                        env_values_evaluated = True
                        break
                if not env_values_evaluated:
                    possible_envs_dict[num_value_changed].append(
                        EnvExecDetails(
                            env_values=list(env_values),
                            executed=False,
                            per_drop=per_drop,
                            env_id=count,
                            init_env_variables=self.init_env_variables,
                            predicate=False,
                            pass_probability=-1.0,
                            regression_probability=-1.0,
                        )
                    )
                count += 1

            self.max_num_iterations = len(possible_envs)
            self.min_num_iterations = len(
                possible_envs_dict[1]
            )  # num of possible environments considering only one param changed
            self.possible_envs_dict = possible_envs_dict

            self.monitor_search_every = monitor_search_every
            self.monitor_progress = None
            if self.monitor_search_every != -1 and self.monitor_search_every > 0:
                self.monitor_progress = MonitorProgress(
                    algo_name=self.algo_name,
                    env_name=standardize_env_name(env_name=self.env_name),
                    results_dir=self.algo_save_dir,
                    param_names_string=self.param_names_string,
                    search_type="alphatest",
                    start_search_time=start_search_time,
                    starting_progress_report_number=starting_progress_report_number,
                )

            self.logger.info("Max num of iterations considering multipliers is: {}".format(self.max_num_iterations))
            self.logger.info("Min num of iterations considering multipliers is: {}".format(self.min_num_iterations))
            assert num_iterations >= self.min_num_iterations, "Num iterations {} is too low. It should be at least {}".format(
                num_iterations, self.min_num_iterations
            )

    def search(self) -> Union[Archive, None]:
        if self.stop_at_min_max_num_iterations:
            return None

        # assumes training with default params is already done
        self.logger.debug("Original env: {}".format(self.init_env_variables.get_params_string()))
        self.logger.debug("\n")
        range_fn = (
            range(0, self.num_iterations)
            if not self.previous_num_iterations
            else range(self.previous_num_iterations, self.num_iterations)
        )

        self.logger.info("Num of cpu threads: {}".format(multiprocessing.cpu_count()))

        start_search_time = time.time()

        if self.parallelize_search:

            # exp_search
            candidates_exp_search = list(itertools.chain(*self.possible_envs_dict.values()))

            exp_search_suffixes = []
            for i in range(len(candidates_exp_search)):
                search_suffix = "exp_search_"
                if self.param_names:
                    search_suffix += self.param_names_string + "_"
                if self.exp_suffix:
                    search_suffix += self.exp_suffix + "_"
                search_suffix += str(i)
                exp_search_suffixes.append(search_suffix)

            num_of_processes_to_spawn = NUM_OF_THREADS // self.runs_for_probability_estimation
            if num_of_processes_to_spawn <= len(candidates_exp_search):
                num_of_processes_to_spawn -= 2

            self.logger.info("max num of processes to spawn: {}".format(num_of_processes_to_spawn))
            queue = Queue()
            queue_result = Queue()
            # Create worker threads
            for _ in range(num_of_processes_to_spawn):
                worker = IterationsWorker(
                    queue=queue, queue_result=queue_result, runner=self.runner, start_time=self.start_time
                )
                # Setting daemon to True will let the main thread exit even though the workers are blocking
                worker.daemon = True
                worker.start()
            # Put the tasks into the queue as a tuple
            for i, search_suffix in enumerate(exp_search_suffixes):
                candidate_new_env_variables = copy.deepcopy(self.init_env_variables)
                env_values_to_run = candidates_exp_search[i].get_env_values()
                for index, value in enumerate(env_values_to_run):
                    candidate_new_env_variables.set_param(index=index, new_value=value)
                work_to_pass = (i, search_suffix, candidate_new_env_variables)
                queue.put(work_to_pass)
            # Causes the main thread to wait for the queue to finish processing all the tasks
            queue.join()  # sync point

            if self.monitor_progress:
                # in this case I do not take into account the monitor_every param but I print at every sync point
                self.monitor_progress.send_progress_report(time_elapsed=(time.time() - start_search_time))

            execution_results_stop = []
            execution_results_for_binary_search = []
            iteration = 0
            while not queue_result.empty():
                env_predicate_pair, execution_time, regression_time = queue_result.get_nowait()
                if env_predicate_pair.is_predicate():
                    execution_results_stop.append((env_predicate_pair, execution_time, iteration, regression_time))
                else:
                    execution_results_for_binary_search.append((env_predicate_pair, execution_time, iteration))
                # it should not raise an exception since we know that all environments
                self.buffer_env_predicate_pairs.append(env_predicate_pair=env_predicate_pair)
                iteration += 1

            # store in directories exp_searches which did not invalidate the environment
            for execution_result_stop in execution_results_stop:
                current_iteration = execution_result_stop[2]
                self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                self.archive.save(current_iteration=current_iteration)
                self._move_output_directories(current_iteration=current_iteration)
                save_time_elapsed(
                    save_dir=self.algo_save_dir,
                    current_iteration=current_iteration,
                    execution_times=[execution_result_stop[1]],
                )
                save_time_elapsed(
                    save_dir=self.algo_save_dir,
                    current_iteration=current_iteration,
                    execution_times=[execution_result_stop[3]],
                    regression=True,
                )

            # binary_search
            queue = Queue()
            queue_result = Queue()
            # Create worker threads
            for _ in range(num_of_processes_to_spawn):
                worker = BinarySearchWorker(
                    queue=queue,
                    queue_result=queue_result,
                    runner=self.runner,
                    start_time=self.start_time,
                    init_env_variables=self.init_env_variables,
                    param_names=self.param_names,
                    param_names_string=self.param_names_string,
                    algo_name=self.algo_name,
                    env_name=self.env_name,
                    binary_search_epsilon=self.binary_search_epsilon,
                    discrete_action_space=self.all_params["discrete_action_space"],
                )
                # Setting daemon to True will let the main thread exit even though the workers are blocking
                worker.daemon = True
                worker.start()
            # Put the tasks into the queue as a tuple
            for i in range(len(execution_results_for_binary_search)):
                current_env_variables = execution_results_for_binary_search[i][0].get_env_variables()
                current_iteration = execution_results_for_binary_search[i][2]
                self.logger.info(
                    "Env variables for binary search: {}, Iteration: {}".format(
                        current_env_variables.get_params_string(), current_iteration
                    )
                )
                work_to_pass = (current_env_variables, current_iteration, self.buffer_env_predicate_pairs)
                queue.put(work_to_pass)
            # Causes the main thread to wait for the queue to finish processing all the tasks
            queue.join()  # sync point

            if self.monitor_progress:
                # in this case I do not take into account the monitor_every param but I print at every sync point
                self.monitor_progress.send_progress_report(time_elapsed=(time.time() - start_search_time))

            while not queue_result.empty():
                (
                    t_env_variables,
                    f_env_variables,
                    env_predicate_pairs,
                    max_binary_search_iterations,
                    binary_search_counter,
                    current_iteration,
                    execution_times,
                    regression_times,
                ) = queue_result.get_nowait()

                for env_predicate_pair in env_predicate_pairs:
                    if not self.buffer_env_predicate_pairs.is_already_evaluated(
                        candidate_env_variables=env_predicate_pair.get_env_variables()
                    ):
                        self.buffer_env_predicate_pairs.append(env_predicate_pair=env_predicate_pair)

                if binary_search_counter == max_binary_search_iterations:
                    self.logger.info("Binary search did not converge in " + str(max_binary_search_iterations) + " steps")
                    self.logger.info("Reporting the closest pair found")
                    self.archive.append(t_env_variables=t_env_variables[0], f_env_variables=f_env_variables[1])
                    self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                    self.archive.save(current_iteration=current_iteration)
                    self._move_output_directories(current_iteration=current_iteration)

                self.archive.append(t_env_variables=t_env_variables, f_env_variables=f_env_variables)
                self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                self._move_output_directories(current_iteration=current_iteration)
                self.archive.save(current_iteration=current_iteration)
                save_time_elapsed(
                    save_dir=self.algo_save_dir, current_iteration=current_iteration, execution_times=execution_times
                )
                save_time_elapsed(
                    save_dir=self.algo_save_dir,
                    current_iteration=current_iteration,
                    execution_times=regression_times,
                    regression=True,
                )
        else:
            for current_iteration in range_fn:

                if self.monitor_progress and current_iteration % self.monitor_search_every == 0:
                    self.monitor_progress.send_progress_report(time_elapsed=(time.time() - start_search_time))

                execution_times = []
                regression_times = []
                self._log_separator("start_iteration", iteration=current_iteration)
                self.logger.info("TIME ELAPSED: {}".format(str(datetime.timedelta(seconds=(time.time() - self.start_time)))))
                current_env_variables = copy.deepcopy(self.init_env_variables)
                max_exp_search_iterations = 20
                exp_search_counter = 0

                search_suffix = "exp_search_"
                if self.param_names:
                    search_suffix += self.param_names_string + "_"
                if self.exp_suffix:
                    search_suffix += self.exp_suffix + "_"
                search_suffix += str(current_iteration) + "_" + str(exp_search_counter)

                self._log_separator("exp_search", iteration=current_iteration)
                exp_search_exception = False
                exp_search_exhausted = False
                while True:
                    try:
                        env_predicate_pair, execution_time, regression_time = self._exp_search(
                            current_iteration=current_iteration,
                            search_suffix=search_suffix,
                            current_env_variables=current_env_variables,
                        )
                        execution_times.append(execution_time)
                        regression_times.append(regression_time)
                    except OverflowError as e:
                        self.logger.info(e)
                        exp_search_exception = True
                        break
                    except ExpSearchExhaustedError as e:
                        self.logger.info(e)
                        exp_search_exhausted = True
                        break

                    current_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                    self.buffer_env_predicate_pairs.append(env_predicate_pair)
                    if self.decision_tree_guidance:
                        self._train_decision_tree()
                    if not env_predicate_pair.is_predicate() or exp_search_counter >= max_exp_search_iterations:
                        break

                    exp_search_counter += 1

                    search_suffix = "exp_search_"
                    if self.param_names:
                        search_suffix += self.param_names_string + "_"
                    if self.exp_suffix:
                        search_suffix += self.exp_suffix + "_"
                    search_suffix += str(current_iteration) + "_" + str(exp_search_counter)

                if exp_search_counter == max_exp_search_iterations or exp_search_exception:
                    if not exp_search_exception:
                        self.logger.info(
                            "Exponential search could not invalidate env "
                            + self.init_env_variables.get_params_string()
                            + " in "
                            + str(exp_search_counter)
                            + " steps"
                        )
                    self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                    self.buffer_executions_skipped.save(current_iteration=current_iteration)
                    self.archive.save(current_iteration=current_iteration)
                    self._move_output_directories(current_iteration=current_iteration)
                    continue

                if exp_search_exhausted:
                    self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                    self.buffer_executions_skipped.save(current_iteration=current_iteration)
                    # self.archive.save(current_iteration=current_iteration)
                    self._move_output_directories(current_iteration=current_iteration)
                    break

                self.logger.info("Env found by exp search: {}".format(current_env_variables.get_params_string()))

                if self.only_exp_search:
                    self.logger.debug("Skipping binary_search")
                    self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                    self.buffer_executions_skipped.save(current_iteration=current_iteration)
                    self._move_output_directories(current_iteration=current_iteration)
                    continue

                self._log_separator("binary_search", iteration=current_iteration)
                # TODO maybe look in the buffer at the t_env that is closest to the new f_env and does not belong
                #  to any frontier point verify if it is worth it by looking at if the avg number of binary_search
                #  runs increases with #iterations when analyzing if dominance idea in binary_search is effective
                #  in reducing # runs.
                t_env_variables = copy.deepcopy(self.init_env_variables)
                t_env_predicate_pair = copy.deepcopy(self.buffer_env_predicate_pairs.get_buffer()[0])
                f_env_variables = current_env_variables
                f_env_predicate_pair = copy.deepcopy(self.buffer_env_predicate_pairs.get_buffer()[-1])
                binary_search_counter = 0
                max_binary_search_iterations = 20

                search_suffix = "binary_search_"
                if self.param_names:
                    search_suffix += self.param_names_string + "_"
                if self.exp_suffix:
                    search_suffix += self.exp_suffix + "_"
                search_suffix += str(current_iteration) + "_" + str(binary_search_counter)

                dist = compute_dist(t_env_variables=t_env_variables, f_env_variables=f_env_variables)
                best_t_f_env_variables = (t_env_variables, f_env_variables, dist)
                pass_probabilities = [(1.0, 0.0, dist)]

                while not is_frontier_pair(
                    t_env_variables=t_env_variables,
                    f_env_variables=f_env_variables,
                    epsilon=self.binary_search_epsilon,
                    dist=dist,
                ):
                    new_env_variables = self._binary_search(t_env_variables=t_env_variables, f_env_variables=f_env_variables,)

                    self.logger.debug("New env after binary search: {}".format(new_env_variables.get_params_string()))
                    # is this candidate env dominated by another executed env that evaluates to True?
                    # If yes then no need to execute, we already know that it will evaluate to true

                    # TODO: refactor
                    # there could be more than one executed env that dominates the candidate but for simplicity now
                    # I only take one; otherwise take the mean in the values of pass and regression probability
                    executed_env_dominate_true = self.buffer_env_predicate_pairs.dominance_analysis(
                        candidate_env_variables=new_env_variables
                    )
                    executed_env_dominate_false = self.buffer_env_predicate_pairs.dominance_analysis(
                        candidate_env_variables=new_env_variables, predicate_to_consider=False
                    )
                    if self.binary_search_guidance and executed_env_dominate_false:
                        assert (
                            executed_env_dominate_true is None
                        ), "it can't be that env {} dominates False env {} and is dominated by True env {}".format(
                            new_env_variables.get_params_string(),
                            executed_env_dominate_false.get_env_variables().get_params_string(),
                            executed_env_dominate_true.get_env_variables().get_params_string(),
                        )
                        self.logger.debug(
                            "binary_search_guidance: env {} not executed because dominates False env {}".format(
                                new_env_variables.get_params_string(),
                                executed_env_dominate_false.get_env_variables().get_params_string(),
                            )
                        )
                        env_predicate_pair = EnvPredicatePair(
                            env_variables=new_env_variables,
                            probability_estimation_runs=executed_env_dominate_false.get_probability_estimation_runs(),
                            predicate=False,
                            regression_probability=1.0,
                            model_dirs=executed_env_dominate_false.get_model_dirs(),
                        )

                        self.buffer_executions_skipped.append(
                            ExecutionSkipped(
                                env_predicate_pair_skipped=env_predicate_pair,
                                env_predicate_pair_executed=executed_env_dominate_false,
                                search_component="binary_search",
                            )
                        )
                    elif self.binary_search_guidance and executed_env_dominate_true:
                        assert (
                            executed_env_dominate_false is None
                        ), "it can't be that env {} is dominated by True env {} and dominates False env {}".format(
                            new_env_variables.get_params_string(),
                            executed_env_dominate_true.get_env_variables().get_params_string(),
                            executed_env_dominate_false.get_env_variables().get_params_string(),
                        )
                        self.logger.debug(
                            "binary_search_guidance: env {} not executed because dominated by True env {}".format(
                                new_env_variables.get_params_string(),
                                executed_env_dominate_true.get_env_variables().get_params_string(),
                            )
                        )
                        env_predicate_pair = EnvPredicatePair(
                            env_variables=new_env_variables,
                            probability_estimation_runs=executed_env_dominate_true.get_probability_estimation_runs(),
                            predicate=True,
                            regression_probability=executed_env_dominate_true.get_regression_probability(),
                            regression_estimation_runs=executed_env_dominate_true.get_regression_estimation_runs(),
                            model_dirs=executed_env_dominate_true.get_model_dirs(),
                        )
                        self.buffer_executions_skipped.append(
                            ExecutionSkipped(
                                env_predicate_pair_skipped=env_predicate_pair,
                                env_predicate_pair_executed=executed_env_dominate_true,
                                search_component="binary_search",
                            )
                        )
                    else:
                        env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                            current_iteration=current_iteration,
                            search_suffix=search_suffix,
                            current_env_variables=new_env_variables,
                            _start_time=self.start_time,
                        )
                        execution_times.append(execution_time)
                        regression_times.append(regression_time)

                    self.buffer_env_predicate_pairs.append(env_predicate_pair)
                    if self.decision_tree_guidance:
                        self._train_decision_tree()
                    if env_predicate_pair.is_predicate():
                        self.logger.debug(
                            "New t_env found: {}".format(env_predicate_pair.get_env_variables().get_params_string())
                        )
                        t_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        t_env_predicate_pair = copy.deepcopy(env_predicate_pair)
                    else:
                        self.logger.debug(
                            "New f_env found: {}".format(env_predicate_pair.get_env_variables().get_params_string())
                        )
                        f_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        f_env_predicate_pair = copy.deepcopy(env_predicate_pair)

                    dist = compute_dist(t_env_variables=t_env_variables, f_env_variables=f_env_variables)

                    t_pass_probability = (
                        t_env_predicate_pair.get_pass_probability()
                        if t_env_predicate_pair.get_pass_probability()
                        else t_env_predicate_pair.compute_pass_probability()
                    )
                    f_pass_probability = (
                        f_env_predicate_pair.get_pass_probability()
                        if f_env_predicate_pair.get_pass_probability()
                        else f_env_predicate_pair.compute_pass_probability()
                    )
                    pass_probabilities.append((t_pass_probability, f_pass_probability, dist))
                    self.logger.debug("#{} pass probabilities: {}".format(binary_search_counter, pass_probabilities))

                    binary_search_counter += 1

                    search_suffix = "binary_search_"
                    if self.param_names:
                        search_suffix += self.param_names_string + "_"
                    if self.exp_suffix:
                        search_suffix += self.exp_suffix + "_"
                    search_suffix += str(current_iteration) + "_" + str(binary_search_counter)

                    if dist < best_t_f_env_variables[2]:
                        best_t_f_env_variables = (t_env_variables, f_env_variables, dist)

                    if binary_search_counter == max_binary_search_iterations:
                        break

                if binary_search_counter == max_binary_search_iterations:
                    self.logger.info("Binary search did not converge in " + str(max_binary_search_iterations) + " steps")
                    self.logger.info(
                        "Reporting the closest pair found, t_env: {}, f_env: {}, dist: {}, epsilon: {}".format(
                            best_t_f_env_variables[0].get_params_string(),
                            best_t_f_env_variables[1].get_params_string(),
                            dist,
                            self.binary_search_epsilon,
                        )
                    )
                    self.archive.append_best_frontier_pair(
                        t_env_variables=best_t_f_env_variables[0],
                        f_env_variables=best_t_f_env_variables[1],
                        best_distance=dist,
                    )
                    self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                    self.buffer_executions_skipped.save(current_iteration=current_iteration)
                    self.archive.save(current_iteration=current_iteration)
                    self._move_output_directories(current_iteration=current_iteration)
                    continue
                else:
                    assert is_frontier_pair(
                        t_env_variables=t_env_variables, f_env_variables=f_env_variables, epsilon=self.binary_search_epsilon
                    ), "Dist {} should be <= epsilon {}".format(dist, self.binary_search_epsilon)

                self.archive.append(t_env_variables=t_env_variables, f_env_variables=f_env_variables)
                self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
                self.buffer_executions_skipped.save(current_iteration=current_iteration)
                self._move_output_directories(current_iteration=current_iteration)
                self.archive.save(current_iteration=current_iteration)
                self.logger.debug("\n")
                self.logger.debug("\n")

                save_time_elapsed(
                    save_dir=self.algo_save_dir, current_iteration=current_iteration, execution_times=execution_times
                )
                save_time_elapsed(
                    save_dir=self.algo_save_dir,
                    current_iteration=current_iteration,
                    execution_times=regression_times,
                    regression=True,
                )

                if self.stop_at_first_iteration:
                    break

        if not isinstance(self.agent, AgentStub):
            self.resample()

        return self.archive

    def _move_output_directories(self, current_iteration, resampling: bool = False) -> None:
        # moving directories for better organization
        dirs_prefix = os.path.abspath(PREFIX_DIR_MODELS_SAVE)
        exp_search_suffix = "_exp_search_" if not self.param_names else "_exp_search_" + self.param_names_string + "_"
        exp_search_suffix += self.exp_suffix + "_" if self.exp_suffix else ""
        exp_search_suffix += str(current_iteration) + "_*"
        binary_search_suffix = "_binary_search_" if not self.param_names else "_binary_search_" + self.param_names_string + "_"
        binary_search_suffix += self.exp_suffix + "_" if self.exp_suffix else ""
        binary_search_suffix += str(current_iteration) + "_*"
        exp_search_saved_dirs = glob.glob(
            dirs_prefix
            + "/"
            + self.algo_name
            + "/logs_"
            + self.tb_log_name
            + ("_" + self.model_suffix if self.model_suffix else "")
            + "_"
            + self.continue_learning_suffix
            + exp_search_suffix
        )
        binary_search_saved_dirs = glob.glob(
            dirs_prefix
            + "/"
            + self.algo_name
            + "/logs_"
            + self.tb_log_name
            + ("_" + self.model_suffix if self.model_suffix else "")
            + "_"
            + self.continue_learning_suffix
            + binary_search_suffix
        )
        self.logger.debug("Moving files and directories......")
        self.logger.debug("exp_search_saved_dirs: {}".format(exp_search_saved_dirs))
        self.logger.debug("binary_search_saved_dirs: {}".format(binary_search_saved_dirs))
        self.logger.debug("exp_search_suffix: {}".format(exp_search_suffix))
        self.logger.debug("binary_search_suffix: {}".format(binary_search_suffix))
        all_saved_dirs = [exp_search_saved_dirs, binary_search_saved_dirs]
        dir_to_create = (
            self.algo_save_dir + "/iteration_" + str(current_iteration)
            if not resampling
            else self.algo_save_dir + "/iteration_" + str(current_iteration) + "_resampling"
        )
        for saved_dirs in all_saved_dirs:
            if len(saved_dirs) > 0:
                os.makedirs(dir_to_create, exist_ok=True)
                for saved_dir in saved_dirs:
                    shutil.move(saved_dir, dir_to_create)

    def _exp_search(
        self, current_iteration: int, search_suffix: str, current_env_variables: EnvVariables,
    ) -> Tuple[EnvPredicatePair, float, float]:

        candidate_new_env_variables = copy.deepcopy(current_env_variables)
        candidates: List[EnvExecDetails] = []
        num_param_changed_selected = 0

        for num_param_changed in self.possible_envs_dict.keys():
            candidates = list(
                filter(lambda env_exec_details: not env_exec_details.is_executed(), self.possible_envs_dict[num_param_changed])
            )
            # filter also based on the env that may be already evaluated by binary search
            new_candidates = []
            for i, candidate in enumerate(candidates):
                candidate_new_env_variables = copy.deepcopy(current_env_variables)
                _ = candidate_new_env_variables.mutate_params(candidates=[candidate])
                if self.buffer_env_predicate_pairs.is_already_evaluated(candidate_env_variables=candidate_new_env_variables):
                    self.possible_envs_dict[num_param_changed][i].set_executed(executed=True)
                else:
                    new_candidates.append(candidate)

            candidates = copy.deepcopy(new_candidates)
            self.logger.debug("All candidates exp search: {}".format([candidate.__str__() for candidate in candidates]))
            if len(candidates) > 0:
                num_param_changed_selected = num_param_changed
                self.logger.debug("Selecting env with num param changed = {}".format(num_param_changed))
                break

        if num_param_changed_selected == 0:
            raise ExpSearchExhaustedError("All environments have been tried, stopping search")

        index = candidate_new_env_variables.mutate_params(candidates=candidates)
        candidate_selected: EnvExecDetails = candidates[index]
        self.logger.debug("Candidate selected: {}".format(candidate_selected.__str__()))

        # TODO: refactor
        executed_env_dominate_false = self.buffer_env_predicate_pairs.dominance_analysis(
            candidate_env_variables=candidate_new_env_variables, predicate_to_consider=False
        )
        executed_env_dominate_true = self.buffer_env_predicate_pairs.dominance_analysis(
            candidate_env_variables=candidate_new_env_variables
        )

        index_dict = self.possible_envs_dict[num_param_changed_selected].index(candidate_selected)
        self.possible_envs_dict[num_param_changed_selected][index_dict].set_executed(executed=True)
        self.logger.debug(
            "Setting candidate {} as executed {}".format(
                self.possible_envs_dict[num_param_changed_selected][index_dict].__str__(),
                self.possible_envs_dict[num_param_changed_selected][index_dict].is_executed(),
            )
        )

        if self.exp_search_guidance and executed_env_dominate_false:
            assert (
                executed_env_dominate_true is None
            ), "it can't be that env {} dominates False env {} and is dominated by True env {}".format(
                candidate_new_env_variables.get_params_string(),
                executed_env_dominate_false.get_env_variables().get_params_string(),
                executed_env_dominate_true.get_env_variables().get_params_string(),
            )
            self.logger.debug(
                "exp_search_guidance: env {} not executed because it dominates False env {}".format(
                    candidate_new_env_variables.get_params_string(),
                    executed_env_dominate_false.get_env_variables().get_params_string(),
                )
            )

            env_predicate_pair = EnvPredicatePair(
                env_variables=candidate_new_env_variables,
                probability_estimation_runs=executed_env_dominate_false.get_probability_estimation_runs(),
                predicate=False,
                regression_probability=1.0,
                model_dirs=executed_env_dominate_false.get_model_dirs(),
            )
            execution_time = 0.0
            regression_time = 0.0
            self.buffer_executions_skipped.append(
                ExecutionSkipped(
                    env_predicate_pair_skipped=env_predicate_pair,
                    env_predicate_pair_executed=executed_env_dominate_false,
                    search_component="exp_search",
                )
            )
        elif self.exp_search_guidance and executed_env_dominate_true:
            assert (
                executed_env_dominate_false is None
            ), "it can't be that env {} is dominated by True env {} and dominates False env {}".format(
                candidate_new_env_variables.get_params_string(),
                executed_env_dominate_true.get_env_variables().get_params_string(),
                executed_env_dominate_false.get_env_variables().get_params_string(),
            )

            self.logger.debug(
                "exp_search_guidance: env {} not executed because dominated by True env {}".format(
                    candidate_new_env_variables.get_params_string(),
                    executed_env_dominate_true.get_env_variables().get_params_string(),
                )
            )

            # we can assume that in all runs the algorithm adapted
            env_predicate_pair = EnvPredicatePair(
                env_variables=candidate_new_env_variables,
                probability_estimation_runs=executed_env_dominate_true.get_probability_estimation_runs(),
                predicate=True,
                regression_probability=executed_env_dominate_true.get_regression_probability(),
                regression_estimation_runs=executed_env_dominate_true.get_regression_estimation_runs(),
                model_dirs=executed_env_dominate_true.get_model_dirs(),
            )
            execution_time = 0.0
            regression_time = 0.0
            self.buffer_executions_skipped.append(
                ExecutionSkipped(
                    env_predicate_pair_skipped=env_predicate_pair,
                    env_predicate_pair_executed=executed_env_dominate_true,
                    search_component="exp_search",
                )
            )
        else:
            env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                current_iteration=current_iteration,
                search_suffix=search_suffix,
                current_env_variables=candidate_new_env_variables,
                _start_time=self.start_time,
            )

        self.possible_envs_dict[num_param_changed_selected][index_dict].set_predicate(
            predicate=env_predicate_pair.is_predicate()
        )
        self.possible_envs_dict[num_param_changed_selected][index_dict].set_pass_probability(
            pass_probability=env_predicate_pair.get_pass_probability()
        )
        self.possible_envs_dict[num_param_changed_selected][index_dict].set_regression_probability(
            regression_probability=env_predicate_pair.get_regression_probability()
        )
        return env_predicate_pair, execution_time, regression_time

    def _train_decision_tree(self) -> None:
        lists_of_values = []
        list_of_predicates = []

        for env_predicate_pair in self.buffer_env_predicate_pairs.get_buffer():
            lists_of_values.append(_translate_to_values(env_predicate_pair.get_env_variables()))
            list_of_predicates.append(env_predicate_pair.is_predicate())

        X = np.array(lists_of_values)
        y = np.array(list_of_predicates)

        self.dt.fit(X, y)

    def _predict_with_decision_tree(self, env_variables: EnvVariables) -> bool:
        X_test = np.array([_translate_to_values(env_variables=env_variables)])
        return self.dt.predict(X_test)

    def _binary_search(self, t_env_variables: EnvVariables, f_env_variables: EnvVariables,) -> EnvVariables:

        original_max_iterations = 50
        max_number_iterations = original_max_iterations

        while True:

            # compute all possible combinations of environments
            candidates_dict = dict()
            t_f_env_variables = random.choice([(t_env_variables, True), (f_env_variables, False)])

            for i in range(len(t_env_variables.get_params())):
                new_value = (
                    t_env_variables.get_param(index=i).get_current_value()
                    + f_env_variables.get_param(index=i).get_current_value()
                ) / 2
                if i not in candidates_dict:
                    candidates_dict[i] = []
                if (
                    t_env_variables.get_param(index=i).get_current_value()
                    != f_env_variables.get_param(index=i).get_current_value()
                ):
                    candidates_dict[i].append(new_value)
                for index in range(len(t_env_variables.get_params())):
                    if index not in candidates_dict:
                        candidates_dict[index] = []
                    if index != i:
                        candidates_dict[index].append(t_f_env_variables[0].get_values()[index])

            all_candidates = list(itertools.product(*list(candidates_dict.values())))
            self.logger.debug(
                "t_env: {}, f_env: {}".format(t_env_variables.get_params_string(), f_env_variables.get_params_string())
            )
            self.logger.debug("all candidates binary search: {}".format(all_candidates))
            all_candidates_env_variables_filtered = []
            all_candidates_env_variables = []
            for candidate_values in all_candidates:
                env_values = dict()
                for i in range(len(t_f_env_variables[0].get_params())):
                    param_name = t_f_env_variables[0].get_param(index=i).get_name()
                    env_values[param_name] = candidate_values[i]
                candidate_env_variables = instantiate_env_variables(
                    algo_name=self.algo_name,
                    discrete_action_space=self.all_params["discrete_action_space"],
                    env_name=self.env_name,
                    param_names=self.param_names,
                    env_values=env_values,
                )
                if not self.buffer_env_predicate_pairs.is_already_evaluated(candidate_env_variables=candidate_env_variables):
                    all_candidates_env_variables_filtered.append(candidate_env_variables)
                all_candidates_env_variables.append(candidate_env_variables)
            if len(all_candidates_env_variables_filtered) > 0:
                candidate_new_env_variables = random.choice(all_candidates_env_variables_filtered)
                break
            else:
                # remove candidate = t_f_env_variables
                all_candidates_env_variables_filtered = []
                for candidate_env_variables in all_candidates_env_variables:
                    if not candidate_env_variables.is_equal(t_f_env_variables[0]):
                        all_candidates_env_variables_filtered.append(candidate_env_variables)
                assert (
                    len(all_candidates_env_variables_filtered) > 0
                ), "there must be at least one candidate env for binary search"
                candidate_env_variables_already_evaluated = random.choice(all_candidates_env_variables_filtered)
                if t_f_env_variables[1]:
                    t_env_variables = copy.deepcopy(candidate_env_variables_already_evaluated)
                else:
                    f_env_variables = copy.deepcopy(candidate_env_variables_already_evaluated)

            max_number_iterations -= 1

            if max_number_iterations == 0:
                break

        assert max_number_iterations > 0, "Could not binary mutate any param of envs {} and {} in {} steps".format(
            t_env_variables.get_params_string(), f_env_variables.get_params_string(), str(original_max_iterations)
        )

        return candidate_new_env_variables

    def _log_separator(self, phase: str, iteration: int) -> None:
        if phase == "start_iteration":
            self.logger.info("*********************** ITERATION {} ***********************".format(iteration))
        elif phase == "exp_search":
            self.logger.info("####################### EXP SEARCH {} #######################".format(iteration))
        elif phase == "binary_search":
            self.logger.info("%%%%%%%%%%%%%%%%%%%%%%% BINARY SEARCH {} %%%%%%%%%%%%%%%%%%%%%%%".format(iteration))

    def _choose_based_on_percentage_drops(self, indexes: List[int], env_variables: EnvVariables) -> int:
        per_drops = []
        for i in indexes:
            per_drop = env_variables.get_param(index=i).get_percentage_drop()
            per_drops.append(per_drop)

        self.logger.debug("Percentage drops: {}".format(per_drops))
        return random.choices(population=indexes, weights=per_drops)[0]

    def resample(self):
        self.logger.info("Resampling...")
        env_variables_to_execute_per_param = []
        for i in range(len(self.init_env_variables.get_params())):
            env_variables_to_execute = []
            param = self.init_env_variables.get_params()[i]
            assert (
                param.get_direction() == "positive" or param.get_direction() == "negative"
            ), "Direction either positive or negative. Found {} for param {}".format(param.get_direction(), param.get_name())
            limit = None
            if param.get_direction() == "positive":
                limit = param.get_high_limit() if param.get_starting_multiplier() > 1.0 else param.get_low_limit()
            elif param.get_direction() == "negative":
                limit = param.get_low_limit() if param.get_starting_multiplier() > 1.0 else param.get_high_limit()
            for env_predicate_pair in self.buffer_env_predicate_pairs.get_buffer():
                other_param = env_predicate_pair.get_env_variables().get_param(index=i)
                if env_predicate_pair.is_predicate() and other_param.get_current_value() == limit:
                    self.logger.info(
                        "Env to execute for param {} with limit {}: {}".format(
                            other_param.get_name(), limit, env_predicate_pair.get_env_variables().get_params_string()
                        )
                    )
                    env_variables_to_execute.append(env_predicate_pair.get_env_variables())
            env_variables_to_execute_per_param.append((i, env_variables_to_execute))

        current_iteration = self.previous_num_iterations if self.previous_num_iterations else self.num_iterations - 1
        for index_param, list_of_env_variables_to_execute in env_variables_to_execute_per_param:
            count_executions_per_param = 0
            for env_variables_to_execute in list_of_env_variables_to_execute:
                env_variables_to_execute_modified = copy.deepcopy(env_variables_to_execute)
                param_to_modify = env_variables_to_execute_modified.get_param(index=index_param)
                multiplier = 2.0 if param_to_modify.get_starting_multiplier() > 1.0 else 0.5
                self.logger.info(
                    "Current param value for param {}: {}".format(
                        env_variables_to_execute_modified.get_param(index=index_param).get_name(),
                        env_variables_to_execute_modified.get_param(index=index_param).get_current_value(),
                    )
                )
                predicate = True
                while predicate:
                    search_suffix = "exp_search_"
                    if self.param_names:
                        search_suffix += self.param_names_string + "_"
                    if self.exp_suffix:
                        search_suffix += self.exp_suffix + "_"
                    search_suffix += str(current_iteration) + "_resampling_" + str(count_executions_per_param)
                    env_variables_to_execute_modified.set_param(
                        index=index_param,
                        new_value=env_variables_to_execute_modified.get_param(index=index_param).get_current_value()
                        * multiplier,
                        do_not_check_for_limits=True,
                    )
                    self.logger.info(
                        "New param value for param {}: {}".format(
                            env_variables_to_execute_modified.get_param(index=index_param).get_name(),
                            env_variables_to_execute_modified.get_param(index=index_param).get_current_value(),
                        )
                    )
                    env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                        current_iteration=current_iteration,
                        search_suffix=search_suffix,
                        current_env_variables=env_variables_to_execute_modified,
                        _start_time=self.start_time,
                    )
                    count_executions_per_param += 1
                    self.buffer_env_predicate_pairs.append(env_predicate_pair)
                    if env_predicate_pair.compute_pass_probability() == 0.0:
                        self.logger.info(
                            "Stop resampling. Found value {} of param {} that invalidates the env".format(
                                env_variables_to_execute_modified.get_param(index=index_param).get_current_value(),
                                env_variables_to_execute_modified.get_param(index=index_param).get_name(),
                            )
                        )
                        predicate = env_predicate_pair.is_predicate()
            if len(list_of_env_variables_to_execute) > 0:
                self.buffer_env_predicate_pairs.save(current_iteration=current_iteration, resampling=True)
                self.archive.save(current_iteration=current_iteration, resampling=True)
                self._move_output_directories(current_iteration=current_iteration, resampling=True)
                current_iteration += 1

    def determine_multipliers(self, num_of_runs: int, halve_or_double: str = "double") -> Tuple[List[float], List[float]]:
        assert halve_or_double == "double" or halve_or_double == "halve", "halve_or_double == double or halve, {}".format(
            halve_or_double
        )
        # MAYBE TO CALL IN INIT TO SET STARTING_MULTIPLIERS
        multipliers_aggregator = []
        percentage_drops_aggregator = []
        for i in range(num_of_runs):
            multipliers = []
            percentage_drops = []
            current_env_variables = copy.deepcopy(self.init_env_variables)
            # starting from first index determine multiplier
            for index in range(len(current_env_variables.get_params())):
                param = current_env_variables.get_param(index)
                if param.get_starting_multiplier() > 0.0:
                    self.logger.debug(
                        "No need to search for multiplier for param {} since it is already set {}".format(
                            param.get_name(), param.get_starting_multiplier()
                        )
                    )
                    multipliers.append(param.get_starting_multiplier())
                    percentage_drops.append(param.get_percentage_drop())
                    continue

                assert (
                    param.get_direction() == "positive" or param.get_direction() == "negative"
                ), "unknown direction not supported yet"
                if param.get_direction() == "positive":
                    halve_or_double = "double" if param.get_high_limit() > param.get_current_value() else "halve"
                else:
                    halve_or_double = "double" if param.get_low_limit() < param.get_current_value() else "halve"

                env_predicate_pair = None
                while True:
                    before_mutation_param_value = param.get_current_value()
                    mutated = param.mutate(halve_or_double=halve_or_double)
                    if not mutated:
                        # limit (high or low) reached
                        break
                    env_predicate_pair = self.runner.execute_test_with_callback(current_env_variables=current_env_variables)
                    if not env_predicate_pair.is_predicate():
                        # env invalidated
                        break

                if not env_predicate_pair.is_predicate():
                    percentage_drop = env_predicate_pair.get_execution_info()["percentage_drop"]
                    mean_reward = env_predicate_pair.get_execution_info()["mean_reward"]
                    if percentage_drop >= 100.0 or mean_reward <= current_env_variables.get_lower_limit_reward():
                        # starting_multipliers is meant to challenge the algo in the new env but not so much that
                        # that it cannot recover
                        # doubling/halving is a lot -> try going with baby steps
                        current_env_variables.set_param(index=index, new_value=before_mutation_param_value)
                        mul = 1.5 if halve_or_double == "double" else 0.8
                        while True:
                            self.logger.debug(
                                "Try changing the param with baby steps. Mul {}, new value: {}".format(
                                    mul, param.get_current_value() * mul
                                )
                            )
                            current_env_variables.set_param(index=index, new_value=param.get_current_value() * mul)
                            env_predicate_pair = self.runner.execute_test_with_callback(
                                current_env_variables=current_env_variables
                            )
                            if not env_predicate_pair.is_predicate():
                                percentage_drop = env_predicate_pair.get_execution_info()["percentage_drop"]
                                mean_reward = env_predicate_pair.get_execution_info()["mean_reward"]
                                if percentage_drop >= 100.0 or mean_reward <= current_env_variables.get_lower_limit_reward():
                                    current_env_variables.set_param(index=index, new_value=before_mutation_param_value)
                                    previous_mul = mul
                                    mul = mul - 0.05 if halve_or_double == "double" else mul + 0.05
                                    if halve_or_double == "double":
                                        assert mul > 1.0, "step size too big"
                                    elif halve_or_double == "halve":
                                        assert mul < 1.0, "step size too big"
                                    self.logger.debug("Decreasing step amount: {} -> {}".format(previous_mul, mul))
                                else:
                                    break

                if not mutated:
                    # FIXME do something to cope with it: maybe increase limits automatically and retry
                    # as for now doing it manually to see if it works
                    self.logger.debug(
                        "Not able to invalidate env {} by mutating param {} until limit".format(
                            current_env_variables.get_params_string(), param.get_name()
                        )
                    )
                    multipliers.append(0.0)
                    percentage_drops.append(0.0)
                else:
                    multiplier = param.get_current_value() / param.get_default_value(non_zero_value=True)
                    self.logger.debug(
                        "Env {} invalidated by mutating param {}. Current value: {}, "
                        "starting_multiplier to set to value {}".format(
                            current_env_variables.get_params_string(), param.get_name(), param.get_current_value(), multiplier,
                        )
                    )
                    multipliers.append(abs(multiplier))
                    if env_predicate_pair and "percentage_drop" in env_predicate_pair.get_execution_info():
                        percentage_drops.append(env_predicate_pair.get_execution_info()["percentage_drop"])
                    else:
                        percentage_drops.append(0.0)

                self.logger.debug("Restart from original env with next param")
                current_env_variables = copy.deepcopy(self.init_env_variables)

            self.logger.debug("Multipliers iteration {}: {}".format(i, multipliers))
            self.logger.debug("Percentage drops iteration {}: {}".format(i, percentage_drops))

            multipliers_aggregator.append(multipliers)
            percentage_drops_aggregator.append(percentage_drops)

        multipliers_aggregator = np.array(multipliers_aggregator)
        percentage_drops_aggregator = np.array(percentage_drops_aggregator)
        return multipliers_aggregator.mean(axis=0), percentage_drops_aggregator.mean(axis=0)

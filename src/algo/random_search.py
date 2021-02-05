import copy
import datetime
import glob
import numpy as np
import multiprocessing
import os
import shutil
import time
import math
from queue import Queue

from abstract_agent import AbstractAgent
from algo.archive import Archive, read_saved_archive, is_frontier_pair, compute_dist, compute_inverse_dist_random_search
from algo.env_predicate_pair import BufferEnvPredicatePairs, read_saved_buffer, EnvPredicatePair
from algo.search_utils import BufferExecutionsSkipped, read_saved_buffer_executions_skipped, ExecutionSkipped
from algo.time_elapsed_util import save_time_elapsed
from env_utils import instantiate_env_variables, standardize_env_name
from envs.env_variables import EnvVariables
from execution.iterations_worker import IterationsWorker
from execution.runner import Runner
from log import Log
from send_email import MonitorProgress
from utilities import PREFIX_DIR_MODELS_SAVE, HOME, get_result_file_iteration_number, NUM_OF_THREADS, norm


class RandomSearch:
    def __init__(
        self,
        agent: AbstractAgent,
        num_iterations: int,
        algo_name: str,
        env_name: str,
        tb_log_name: str,
        continue_learning_suffix: str,
        env_variables: EnvVariables,
        param_names=None,
        runs_for_probability_estimation: int = 1,
        buffer_file: str = None,
        archive_file: str = None,
        executions_skipped_file: str = None,
        parallelize_search: bool = False,
        monitor_search_every: bool = False,
        binary_search_epsilon: float = 0.05,
        start_search_time: float = None,
        starting_progress_report_number: int = 0,
        stop_at_first_iteration: bool = False,
        exp_suffix: str = None,
    ):
        assert agent, 'agent should have a value: {}'.format(agent)
        assert algo_name, 'algo_name should have a value: {}'.format(algo_name)
        assert env_name, 'env_name should have a value: {}'.format(env_name)

        self.agent = agent
        self.num_iterations = num_iterations
        self.init_env_variables = env_variables
        self.previous_num_iterations = None
        self.start_time = time.time()
        self.logger = Log("Random")
        self.param_names = param_names
        self.all_params = env_variables.instantiate_env()
        self.runs_for_probability_estimation = runs_for_probability_estimation
        self.buffer_file = buffer_file
        self.archive_file = archive_file
        self.parallelize_search = parallelize_search
        self.stop_at_first_iteration = stop_at_first_iteration
        self.exp_suffix = exp_suffix

        if param_names:
            self.param_names_string = '_'.join(param_names)

        # TODO: refactor buffer restoring in abstract class extended by search algo
        #  (for now only random search and alphatest)
        if buffer_file:
            previously_saved_buffer = read_saved_buffer(buffer_file=buffer_file)
            index_last_slash = buffer_file.rindex('/')

            self.algo_save_dir = buffer_file[:index_last_slash]
            self.logger.debug('Algo save dir from restored execution: {}'.format(self.algo_save_dir))
            self.buffer_env_predicate_pairs = BufferEnvPredicatePairs(save_dir=self.algo_save_dir)
            self.archive = Archive(save_dir=self.algo_save_dir, epsilon=binary_search_epsilon)

            # restore buffer
            for buffer_item in previously_saved_buffer:
                previous_env_variables = instantiate_env_variables(
                    algo_name=algo_name,
                    discrete_action_space=self.all_params['discrete_action_space'],
                    env_name=env_name,
                    param_names=param_names,
                    env_values=buffer_item.get_env_values()
                )
                self.buffer_env_predicate_pairs.append(EnvPredicatePair(
                    env_variables=previous_env_variables,
                    pass_probability=buffer_item.get_pass_probability(),
                    predicate=buffer_item.is_predicate(),
                    regression_probability=buffer_item.get_regression_probability(),
                    probability_estimation_runs=buffer_item.get_probability_estimation_runs(),
                    regression_estimation_runs=buffer_item.get_regression_estimation_runs(),
                    model_dirs=buffer_item.get_model_dirs()
                ))
            assert archive_file, 'when buffer file is available so needs to be the archive file to ' \
                                 'restore a previous execution'
            try:
                previous_num_iterations_buffer = get_result_file_iteration_number(filename=buffer_file)
                previous_num_iterations_archive = get_result_file_iteration_number(filename=archive_file)
                assert previous_num_iterations_buffer == previous_num_iterations_archive, \
                    'The two nums must coincide: {}, {}'.format(previous_num_iterations_buffer,
                                                                previous_num_iterations_archive)
                previous_num_iterations = previous_num_iterations_buffer + 1
            except ValueError as e:
                raise ValueError(e)

            self.previous_num_iterations = previous_num_iterations
            self.logger.info("Restore previous execution of {} iterations.".format(previous_num_iterations))

            # restore archive
            previously_saved_archive = read_saved_archive(archive_file=archive_file)
            t_env_variables = None
            f_env_variables = None
            for env_values, predicate in previously_saved_archive:
                all_params = env_variables.instantiate_env()
                previous_env_variables = instantiate_env_variables(
                    algo_name=algo_name,
                    discrete_action_space=all_params['discrete_action_space'],
                    env_name=env_name,
                    param_names=param_names,
                    env_values=env_values
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
                    buffer_executions_skipped_file=executions_skipped_file)
                for buffer_executions_skipped_item in previously_saved_executions_skipped:
                    previous_env_variables_skipped = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params['discrete_action_space'],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=buffer_executions_skipped_item.env_values_skipped
                    )
                    env_predicate_pair_skipped = EnvPredicatePair(
                        env_variables=previous_env_variables_skipped,
                        predicate=buffer_executions_skipped_item.predicate
                    )
                    previous_env_variables_executed = instantiate_env_variables(
                        algo_name=algo_name,
                        discrete_action_space=self.all_params['discrete_action_space'],
                        env_name=env_name,
                        param_names=param_names,
                        env_values=buffer_executions_skipped_item.env_values_executed
                    )
                    env_predicate_pair_executed = EnvPredicatePair(
                        env_variables=previous_env_variables_executed,
                        predicate=buffer_executions_skipped_item.predicate
                    )
                    self.buffer_executions_skipped.append(
                        ExecutionSkipped(env_predicate_pair_skipped=env_predicate_pair_skipped,
                                         env_predicate_pair_executed=env_predicate_pair_executed,
                                         search_component=buffer_executions_skipped_item.search_component)
                    )
        else:
            attempt = 0

            suffix = "n_iterations_"
            if self.param_names:
                suffix += self.param_names_string + "_"
            if self.exp_suffix:
                suffix += self.exp_suffix + "_"
            suffix += str(num_iterations)

            algo_save_dir = os.path.abspath(HOME + "/random/" + env_name + "/" + algo_name + "/" + suffix + "_" + str(attempt))
            _algo_save_dir = algo_save_dir
            while os.path.exists(_algo_save_dir):
                attempt += 1
                _algo_save_dir = algo_save_dir[:-1] + str(attempt)
            self.algo_save_dir = _algo_save_dir
            os.makedirs(self.algo_save_dir)
            self.buffer_env_predicate_pairs = BufferEnvPredicatePairs(save_dir=self.algo_save_dir)
            # assuming initial env_variables satisfies the predicate of adequate performance
            if self.runs_for_probability_estimation:
                env_predicate_pair = EnvPredicatePair(env_variables=self.init_env_variables, predicate=True,
                                                      probability_estimation_runs=[True] * self.runs_for_probability_estimation)
            else:
                env_predicate_pair = EnvPredicatePair(env_variables=self.init_env_variables, predicate=True)
            self.buffer_env_predicate_pairs.append(env_predicate_pair)
            self.buffer_executions_skipped = BufferExecutionsSkipped(save_dir=self.algo_save_dir)
            self.archive = Archive(save_dir=self.algo_save_dir, epsilon=binary_search_epsilon)

        self.env_name = env_name
        self.algo_name = algo_name
        self.tb_log_name = tb_log_name
        self.continue_learning_suffix = continue_learning_suffix
        self.binary_search_epsilon = binary_search_epsilon

        self.runner = Runner(agent=self.agent, runs_for_probability_estimation=self.runs_for_probability_estimation, )

        self.monitor_search_every = monitor_search_every
        self.monitor_progress = None
        if self.monitor_search_every != -1 and self.monitor_search_every > 0:
            self.monitor_progress = MonitorProgress(
                algo_name=self.algo_name,
                env_name=standardize_env_name(env_name=self.env_name),
                results_dir=self.algo_save_dir,
                param_names_string=self.param_names_string,
                search_type='random',
                start_search_time=start_search_time,
                starting_progress_report_number=starting_progress_report_number
            )

    def search(self) -> Archive:

        self.logger.info('Num of cpu threads: {}'.format(multiprocessing.cpu_count()))

        # assumes training with default params is already done
        self.logger.debug("Original env: {}".format(self.init_env_variables.get_params_string()))
        self.logger.debug("\n")

        range_fn = range(0, self.num_iterations) if not self.previous_num_iterations \
            else range(self.previous_num_iterations, self.num_iterations)

        start_search_time = time.time()

        for current_iteration in range_fn:

            if self.monitor_progress and current_iteration % self.monitor_search_every == 0:
                self.monitor_progress.send_progress_report(time_elapsed=(time.time() - start_search_time))

            execution_times = []
            self._log_separator("start_iteration")
            self.logger.debug("Current iteration: {}".format(current_iteration))
            self.logger.info("TIME ELAPSED: {}".format(str(datetime.timedelta(seconds=(time.time() - self.start_time)))))

            search_suffix = "random_search_"
            if self.param_names:
                search_suffix += self.param_names_string + "_"
            if self.exp_suffix:
                search_suffix += self.exp_suffix + "_"
            search_suffix += str(current_iteration)

            max_attempts = 50
            while True:
                current_env_variables = copy.deepcopy(self.init_env_variables)
                current_env_variables.mutate_params_randomly()
                if not self.buffer_env_predicate_pairs.is_already_evaluated(
                        candidate_env_variables=current_env_variables
                ) or max_attempts == 0:
                    break
                max_attempts -= 1

            if max_attempts == 0:
                raise OverflowError('Max attempts threshold reached')

            # TODO: refactor
            # dominance analysis
            executed_env_dominate_true = self.buffer_env_predicate_pairs.dominance_analysis(
                candidate_env_variables=current_env_variables
            )
            executed_env_dominate_false = self.buffer_env_predicate_pairs.dominance_analysis(
                candidate_env_variables=current_env_variables, predicate_to_consider=False
            )
            if executed_env_dominate_false:
                assert executed_env_dominate_true is None, 'it can\'t be that env {} dominates False env {} and is dominated by True env {}'.format(
                    current_env_variables.get_params_string(),
                    executed_env_dominate_false.get_env_variables().get_params_string(),
                    executed_env_dominate_true.get_env_variables().get_params_string()
                )
                self.logger.debug(
                    'dominance analysis: env {} not executed because dominates False env {}'.format(
                        current_env_variables.get_params_string(),
                        executed_env_dominate_false.get_env_variables().get_params_string())
                )
                env_predicate_pair = EnvPredicatePair(
                    env_variables=current_env_variables,
                    probability_estimation_runs=executed_env_dominate_false.get_probability_estimation_runs(),
                    predicate=False,
                    regression_probability=1.0,
                    model_dirs=executed_env_dominate_false.get_model_dirs()
                )

                self.buffer_executions_skipped.append(
                    ExecutionSkipped(
                        env_predicate_pair_skipped=env_predicate_pair,
                        env_predicate_pair_executed=executed_env_dominate_false,
                        search_component='random'
                    )
                )
            elif executed_env_dominate_true:
                assert executed_env_dominate_false is None, 'it can\'t be that env {} is dominated by True env {} and dominates False env {}'.format(
                    current_env_variables.get_params_string(),
                    executed_env_dominate_true.get_env_variables().get_params_string(),
                    executed_env_dominate_false.get_env_variables().get_params_string(),
                )
                self.logger.debug('dominance_analysis: env {} not executed because dominated by True env {}'.format(
                    current_env_variables.get_params_string(),
                    executed_env_dominate_true.get_env_variables().get_params_string())
                )
                env_predicate_pair = EnvPredicatePair(
                    env_variables=current_env_variables,
                    probability_estimation_runs=executed_env_dominate_true.get_probability_estimation_runs(),
                    predicate=True,
                    regression_probability=executed_env_dominate_true.get_regression_probability(),
                    regression_estimation_runs=executed_env_dominate_true.get_regression_estimation_runs(),
                    model_dirs=executed_env_dominate_true.get_model_dirs()
                )
                self.buffer_executions_skipped.append(
                    ExecutionSkipped(
                        env_predicate_pair_skipped=env_predicate_pair,
                        env_predicate_pair_executed=executed_env_dominate_true,
                        search_component='random'
                    )
                )
            else:
                env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                    current_iteration=current_iteration,
                    search_suffix=search_suffix,
                    current_env_variables=current_env_variables,
                    _start_time=self.start_time,
                )
                execution_times.append(execution_time)

            self.buffer_env_predicate_pairs.append(env_predicate_pair=env_predicate_pair)
            self.buffer_executions_skipped.save(current_iteration=current_iteration)

            candidates_frontier_env_variables = []
            for i in range(len(current_env_variables.get_params())):
                first_candidate_frontier_env_variables = copy.deepcopy(current_env_variables)
                try:
                    first_candidate_frontier_env_variables.set_param(
                        index=i,
                        new_value=compute_inverse_dist_random_search(
                            env_variables=current_env_variables,
                            index_param=i,
                            epsilon=self.binary_search_epsilon)[0]
                    )
                    dist = compute_dist(
                        t_env_variables=current_env_variables, f_env_variables=first_candidate_frontier_env_variables
                    )
                    self.logger.info('1) Dist: {}'.format(dist))
                    if not is_frontier_pair(
                            t_env_variables=current_env_variables,
                            f_env_variables=first_candidate_frontier_env_variables,
                            epsilon=self.binary_search_epsilon) \
                            or math.isclose(dist, 0.0):
                        self.logger.warn('1) Discarding the pair env_1: {}, env_2: {} since it is not a potential frontier pair'.format(
                            current_env_variables.get_params_string(),
                            first_candidate_frontier_env_variables.get_params_string()
                        ))
                    else:
                        candidates_frontier_env_variables.append(first_candidate_frontier_env_variables)
                except ValueError:
                    pass
                second_candidate_frontier_env_variables = copy.deepcopy(current_env_variables)
                try:
                    second_candidate_frontier_env_variables.set_param(
                        index=i,
                        new_value=compute_inverse_dist_random_search(
                            env_variables=current_env_variables,
                            index_param=i,
                            epsilon=self.binary_search_epsilon)[1]
                    )
                    dist = compute_dist(
                        t_env_variables=current_env_variables, f_env_variables=second_candidate_frontier_env_variables
                    )
                    self.logger.info('2) Dist: {}'.format(dist))
                    if not is_frontier_pair(
                            t_env_variables=current_env_variables,
                            f_env_variables=second_candidate_frontier_env_variables,
                            epsilon=self.binary_search_epsilon)\
                            or math.isclose(dist, 0.0):
                        self.logger.warn('2) Discarding the pair env_1: {}, env_2: {} since it is not a potential frontier pair'.format(
                            current_env_variables.get_params_string(),
                            second_candidate_frontier_env_variables.get_params_string()
                        ))
                    else:
                        candidates_frontier_env_variables.append(second_candidate_frontier_env_variables)
                except ValueError:
                    pass

            current_env_predicate_pair = copy.deepcopy(env_predicate_pair)

            self.logger.info('Evaluating neighbors of env {}. Len {}, frontier {}'.format(
                current_env_variables.get_params_string(), len(candidates_frontier_env_variables),
                [candidate_frontier_env_variables.get_params_string() for candidate_frontier_env_variables
                 in candidates_frontier_env_variables])
            )

            if self.parallelize_search:

                # TODO: dominance analysis as below in the non parallelized version
                search_suffix = "random_search_"
                if self.param_names:
                    search_suffix += self.param_names_string + "_"
                if self.exp_suffix:
                    search_suffix += self.exp_suffix + "_"
                search_suffix += str(current_iteration)

                search_suffixes = []
                for i in range(len(candidates_frontier_env_variables)):
                    search_suffix = "random_search_"
                    if self.param_names:
                        search_suffix += self.param_names_string + "_"
                    if self.exp_suffix:
                        search_suffix += self.exp_suffix + "_"
                    search_suffix += str(current_iteration) + "_frontier_" + str(i)
                    search_suffixes.append(search_suffix)

                num_of_processes_to_spawn = NUM_OF_THREADS // self.runs_for_probability_estimation
                self.logger.info('max num of processes to spawn: {}'.format(num_of_processes_to_spawn))
                queue = Queue()
                queue_result = Queue()
                # Create worker threads
                for _ in range(num_of_processes_to_spawn):
                    worker = IterationsWorker(queue=queue, queue_result=queue_result,
                                              runner=self.runner, start_time=self.start_time)
                    # Setting daemon to True will let the main thread exit even though the workers are blocking
                    worker.daemon = True
                    worker.start()
                # Put the tasks into the queue as a tuple
                for i, search_suffix in enumerate(search_suffixes):
                    current_env_variables = candidates_frontier_env_variables[i]
                    work_to_pass = (current_iteration, search_suffix, current_env_variables)
                    queue.put(work_to_pass)
                # Causes the main thread to wait for the queue to finish processing all the tasks
                queue.join()

                env_predicate_pairs = []
                execution_times_worker = []
                while not queue_result.empty():
                    env_predicate_pair, execution_time = queue_result.get_nowait()
                    env_predicate_pairs.append(env_predicate_pair)
                    execution_times_worker.append(execution_time)
                for execution_time_worker in execution_times_worker:
                    execution_times.append(execution_time_worker)

                for env_predicate_pair in env_predicate_pairs:
                    self.buffer_env_predicate_pairs.append(env_predicate_pair=env_predicate_pair)
                    if current_env_predicate_pair.is_predicate() and not env_predicate_pair.is_predicate():
                        f_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        self.archive.append(t_env_variables=current_env_predicate_pair.get_env_variables(),
                                            f_env_variables=f_env_variables)

                    if not current_env_predicate_pair.is_predicate() and env_predicate_pair.is_predicate():
                        t_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        self.archive.append(t_env_variables=t_env_variables,
                                            f_env_variables=current_env_predicate_pair.get_env_variables())

            else:
                for i, candidate_frontier_env_variables in enumerate(candidates_frontier_env_variables):
                    search_suffix = "random_search_"
                    if self.param_names:
                        search_suffix += self.param_names_string + "_"
                    if self.exp_suffix:
                        search_suffix += self.exp_suffix + "_"
                    search_suffix += str(current_iteration) + "_frontier_" + str(i)

                    # TODO: refactor
                    # dominance analysis
                    executed_env_dominate_true = self.buffer_env_predicate_pairs.dominance_analysis(
                        candidate_env_variables=candidate_frontier_env_variables
                    )
                    executed_env_dominate_false = self.buffer_env_predicate_pairs.dominance_analysis(
                        candidate_env_variables=candidate_frontier_env_variables, predicate_to_consider=False
                    )

                    if executed_env_dominate_false:
                        assert executed_env_dominate_true is None, 'it can\'t be that env {} dominates False env {} and is dominated by True env {}'.format(
                            candidate_frontier_env_variables.get_params_string(),
                            executed_env_dominate_false.get_env_variables().get_params_string(),
                            executed_env_dominate_true.get_env_variables().get_params_string()
                        )
                        self.logger.debug(
                            'dominance analysis: env {} not executed because dominates False env {}'.format(
                                candidate_frontier_env_variables.get_params_string(),
                                executed_env_dominate_false.get_env_variables().get_params_string())
                        )
                        env_predicate_pair = EnvPredicatePair(
                            env_variables=candidate_frontier_env_variables,
                            probability_estimation_runs=executed_env_dominate_false.get_probability_estimation_runs(),
                            predicate=False,
                            regression_probability=1.0,
                            model_dirs=executed_env_dominate_false.get_model_dirs()
                        )

                        self.buffer_executions_skipped.append(
                            ExecutionSkipped(
                                env_predicate_pair_skipped=env_predicate_pair,
                                env_predicate_pair_executed=executed_env_dominate_false,
                                search_component='random'
                            )
                        )
                    elif executed_env_dominate_true:
                        assert executed_env_dominate_false is None, 'it can\'t be that env {} is dominated by True env {} and dominates False env {}'.format(
                            candidate_frontier_env_variables.get_params_string(),
                            executed_env_dominate_true.get_env_variables().get_params_string(),
                            executed_env_dominate_false.get_env_variables().get_params_string(),
                        )
                        self.logger.debug(
                            'dominance_analysis: env {} not executed because dominated by True env {}'.format(
                                candidate_frontier_env_variables.get_params_string(),
                                executed_env_dominate_true.get_env_variables().get_params_string())
                        )
                        env_predicate_pair = EnvPredicatePair(
                            env_variables=candidate_frontier_env_variables,
                            probability_estimation_runs=executed_env_dominate_true.get_probability_estimation_runs(),
                            predicate=True,
                            regression_probability=executed_env_dominate_true.get_regression_probability(),
                            regression_estimation_runs=executed_env_dominate_true.get_regression_estimation_runs(),
                            model_dirs=executed_env_dominate_true.get_model_dirs()
                        )
                        self.buffer_executions_skipped.append(
                            ExecutionSkipped(
                                env_predicate_pair_skipped=env_predicate_pair,
                                env_predicate_pair_executed=executed_env_dominate_true,
                                search_component='random'
                            )
                        )
                    else:
                        env_predicate_pair, execution_time, regression_time = self.runner.execute_train(
                            current_iteration=current_iteration,
                            search_suffix=search_suffix,
                            current_env_variables=candidate_frontier_env_variables,
                            _start_time=self.start_time,
                        )
                        execution_times.append(execution_time)

                    # env_predicate_pair, execution_time, _ = self.runner.execute_train(
                    #     current_iteration=current_iteration,
                    #     search_suffix=search_suffix,
                    #     current_env_variables=candidate_frontier_env_variables,
                    #     _start_time=self.start_time,
                    #     random_search=True,
                    # )
                    # execution_times.append(execution_time)

                    self.buffer_env_predicate_pairs.append(env_predicate_pair=env_predicate_pair)
                    if current_env_predicate_pair.is_predicate() and not env_predicate_pair.is_predicate():
                        f_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        self.archive.append(t_env_variables=current_env_predicate_pair.get_env_variables(),
                                            f_env_variables=f_env_variables)

                    if not current_env_predicate_pair.is_predicate() and env_predicate_pair.is_predicate():
                        t_env_variables = copy.deepcopy(env_predicate_pair.get_env_variables())
                        self.archive.append(t_env_variables=t_env_variables,
                                            f_env_variables=current_env_predicate_pair.get_env_variables())

                    if self.stop_at_first_iteration:
                        break

            self.archive.save(current_iteration=current_iteration)
            self.buffer_env_predicate_pairs.save(current_iteration=current_iteration)
            self.buffer_executions_skipped.save(current_iteration=current_iteration)

            self._move_output_directories(current_iteration=current_iteration)
            save_time_elapsed(save_dir=self.algo_save_dir, current_iteration=current_iteration,
                              execution_times=execution_times)

        return self.archive

    def _move_output_directories(self, current_iteration) -> None:
        # moving directories for better organization
        dirs_prefix = os.path.abspath(PREFIX_DIR_MODELS_SAVE)
        saved_dirs = glob.glob(
            dirs_prefix
            + "/"
            + self.algo_name
            + "/"
            + "logs_"
            + self.tb_log_name
            + "_"
            + self.continue_learning_suffix
            + ("_random_search_" if not self.param_names_string else "_random_search_" + self.param_names_string + '_')
            + (self.exp_suffix + '_' if self.exp_suffix else '')
            + str(current_iteration) + '_*'
        )
        if len(saved_dirs) > 0:
            os.makedirs(self.algo_save_dir + "/iteration_" + str(current_iteration), exist_ok=True)
            for saved_dir in saved_dirs:
                shutil.move(saved_dir, self.algo_save_dir + "/iteration_" + str(current_iteration))

    def _log_separator(self, phase: str):
        if phase == "start_iteration":
            self.logger.info("*********************** ITERATION ***********************")
        elif phase == "exp_search":
            self.logger.info("####################### EXP SEARCH #######################")
        elif phase == "binary_search":
            self.logger.info("%%%%%%%%%%%%%%%%%%%%%%% BINARY SEARCH %%%%%%%%%%%%%%%%%%%%%%%")
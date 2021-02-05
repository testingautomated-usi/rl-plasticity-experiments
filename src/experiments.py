import argparse
import datetime
import glob
import os
import time

import numpy as np

from agent import Agent
from agent_stub import AgentStub
from algo.alphatest import AlphaTest
from algo.random_search import RandomSearch
from env_utils import (instantiate_env_variables, instantiate_eval_callback,
                       standardize_env_name)
from log import Log
from utilities import (SUPPORTED_ALGOS, SUPPORTED_ENVS, check_file_existence,
                       check_param_names, get_result_file_iteration_number)

if __name__ == "__main__":

    logger = Log("experiments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", choices=SUPPORTED_ALGOS, required=True, default="ppo2")
    parser.add_argument("--env_name", choices=SUPPORTED_ENVS, type=str, required=True, default="CartPole-v1")
    parser.add_argument("--search_guidance", action="store_false", default=True)
    parser.add_argument("--param_names", type=check_param_names, default=None)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--runs_for_probability_estimation", type=int, default=3)
    parser.add_argument("--continue_learning_suffix", type=str, default="cl_search")
    parser.add_argument("--num_search_iterations", type=int, default=1)
    parser.add_argument("--stub_agent", type=bool, default=False)
    parser.add_argument("--buffer_file", type=check_file_existence, default=None)
    parser.add_argument("--archive_file", type=check_file_existence, default=None)
    parser.add_argument("--executions_skipped_file", type=check_file_existence, default=None)
    parser.add_argument("--search_type", choices=["alphatest", "random"], default="alphatest")
    parser.add_argument("--stop_at_min_max_num_iterations", type=bool, default=False)
    parser.add_argument("--parallelize_search", type=bool, default=False)
    parser.add_argument("--monitor_search_every", type=int, default=-1)
    parser.add_argument("--full_training_time", action="store_true")
    parser.add_argument("--binary_search_epsilon", type=float, default=0.05)
    parser.add_argument("--stop_at_first_iteration", type=bool, default=False)
    parser.add_argument("--model_suffix", type=str, default=None)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--dir_experiments", type=check_file_existence, default=None)
    parser.add_argument("--exp_suffix", type=str, default=None)
    parser.add_argument("--max_runtime_h", type=int, default=48)

    args, _ = parser.parse_known_args()

    param_names = args.param_names

    args.tb_log_name = standardize_env_name(env_name=args.env_name)
    args.n_eval_episodes = 20
    args.continue_learning = True
    args.only_exp_search = False

    args.exp_search_guidance = True
    args.binary_search_guidance = True
    if not args.search_guidance:
        args.exp_search_guidance = False
        args.binary_search_guidance = False
    args.eval_callback = False
    args.show_progress_bar = False
    args.model_to_load = "best_model_eval"
    args.num_envs = 1
    args.render = False
    args.train_total_timesteps = None
    if args.full_training_time:
        args.train_total_timesteps = -1
    args.log_to_tensorboard = False
    args.decision_tree_guidance = False
    args.save_model = True
    args.save_replay_buffer = False

    if args.algo_name == "dqn":
        args.discrete_action_space = True
    elif args.algo_name == "sac":
        args.discrete_action_space = False

    # if args.env_name == 'CartPole-v1':
    if args.algo_name == "ppo2" or args.algo_name == "dqn":
        args.discrete_action_space = True
        args.log_every = 10000
        args.sb_version = "sb2"
    elif args.algo_name == "sac":
        args.log_every = 5000
        args.sb_version = "sb2"

    env_variables = instantiate_env_variables(
        algo_name=args.algo_name,
        discrete_action_space=args.discrete_action_space,
        env_name=args.env_name,
        param_names=param_names,
        model_suffix=args.model_suffix,
    )
    env_eval_callback = instantiate_eval_callback(env_name=args.env_name)

    if not args.stub_agent:
        agent = Agent(
            algo_name=args.algo_name,
            env_name=args.env_name,
            log_to_tensorboard=args.log_to_tensorboard,
            tb_log_name=args.tb_log_name,
            train_total_timesteps=args.train_total_timesteps,
            n_eval_episodes=args.n_eval_episodes,
            render=args.render,
            num_envs=args.num_envs,
            model_to_load=args.model_to_load,
            continue_learning=args.continue_learning,
            discrete_action_space=args.discrete_action_space,
            eval_callback=args.eval_callback,
            env_variables=env_variables,
            continue_learning_suffix=args.continue_learning_suffix,
            env_eval_callback=env_eval_callback,
            show_progress_bar=args.show_progress_bar,
            log_every=args.log_every,
            sb_version=args.sb_version,
            save_model=args.save_model,
            save_replay_buffer=args.save_replay_buffer,
            model_suffix=args.model_suffix,
        )
    else:
        agent = AgentStub(
            algo_name=args.algo_name,
            env_name=args.env_name,
            tb_log_name=args.tb_log_name,
            model_to_load=args.model_to_load,
            continue_learning=args.continue_learning,
            env_variables=env_variables,
            continue_learning_suffix=args.continue_learning_suffix,
        )

    if args.only_exp_search and args.binary_search_guidance:
        raise ValueError("only_exp_search and binary_search_guidance cannot be both True")

    start_search_time = time.time()
    counter = 0

    if args.resample and args.dir_experiments:
        experiment_dirs_pattern = (
            "n_iterations_{}_*".format("_".join(param_names))
            if not args.model_suffix
            else "n_iterations_{}_{}_*".format(args.model_suffix, "_".join(param_names))
        )
        list_of_dir_experiments = glob.glob(os.path.join(args.dir_experiments, experiment_dirs_pattern))
        for dir_experiment in list_of_dir_experiments:
            dir_experiment = os.path.join(args.dir_experiments, dir_experiment)
            list_of_buffer_files = glob.glob(os.path.join(dir_experiment, "buffer_predicate_pairs_*.txt"))
            last_buffer_file = max(list_of_buffer_files, key=get_result_file_iteration_number)
            list_of_archive_files = glob.glob(os.path.join(dir_experiment, "frontier_*.txt"))
            last_archive_file = max(list_of_archive_files, key=get_result_file_iteration_number)
            list_of_executions_skipped_files = glob.glob(os.path.join(dir_experiment, "executions_skipped_*.txt"))
            last_executions_skipped_file = max(list_of_executions_skipped_files, key=get_result_file_iteration_number)
            logger.info("########### Resampling dir {} ###########".format(dir_experiment))
            alphatest = AlphaTest(
                num_iterations=args.num_iterations,
                env_variables=env_variables,
                agent=agent,
                algo_name=args.algo_name,
                env_name=args.env_name,
                tb_log_name=args.tb_log_name,
                continue_learning_suffix=args.continue_learning_suffix,
                exp_search_guidance=args.exp_search_guidance,
                binary_search_guidance=args.binary_search_guidance,
                decision_tree_guidance=args.decision_tree_guidance,
                only_exp_search=args.only_exp_search,
                buffer_file=last_buffer_file,
                archive_file=last_archive_file,
                executions_skipped_file=last_executions_skipped_file,
                param_names=param_names,
                runs_for_probability_estimation=args.runs_for_probability_estimation,
                stop_at_min_max_num_iterations=args.stop_at_min_max_num_iterations,
                parallelize_search=args.parallelize_search,
                monitor_search_every=args.monitor_search_every,
                binary_search_epsilon=args.binary_search_epsilon,
                start_search_time=start_search_time,
                starting_progress_report_number=counter,
                stop_at_first_iteration=args.stop_at_first_iteration,
                model_suffix=args.model_suffix,
            )
            alphatest.resample()
    else:
        times_elapsed = []
        for i in range(args.num_search_iterations):
            if len(times_elapsed) > 0:
                # estimate if it is possible to complete the iteration given
                hours_elapsed_mean = np.asarray(times_elapsed).mean() / 3600
                hours_elapsed_sum = np.asarray(times_elapsed).sum() / 3600
                search_iterations_left = args.num_search_iterations - i + 1
                logger.info("Hours elapsed: {}h. Iterations left: {}".format(hours_elapsed_sum, search_iterations_left))
                if hours_elapsed_sum + hours_elapsed_mean > args.max_runtime_h:
                    logger.info(
                        "########### Stop experiments at iteration {} since it is unlikely "
                        "that the next iteration will complete within the runtime {}h. "
                        "Mean runtime: {}h, Time elapsed: {}h ###########".format(
                            i, args.max_runtime_h, hours_elapsed_mean, hours_elapsed_sum
                        )
                    )
                    break
            logger.info("########### Start repetition num {} ###########".format(i))
            if args.monitor_search_every != -1:
                counter = i
            if args.search_type == "alphatest":
                alphatest = AlphaTest(
                    num_iterations=args.num_iterations,
                    env_variables=env_variables,
                    agent=agent,
                    algo_name=args.algo_name,
                    env_name=args.env_name,
                    tb_log_name=args.tb_log_name,
                    continue_learning_suffix=args.continue_learning_suffix,
                    exp_search_guidance=args.exp_search_guidance,
                    binary_search_guidance=args.binary_search_guidance,
                    decision_tree_guidance=args.decision_tree_guidance,
                    only_exp_search=args.only_exp_search,
                    buffer_file=args.buffer_file,
                    archive_file=args.archive_file,
                    executions_skipped_file=args.executions_skipped_file,
                    param_names=param_names,
                    runs_for_probability_estimation=args.runs_for_probability_estimation,
                    stop_at_min_max_num_iterations=args.stop_at_min_max_num_iterations,
                    parallelize_search=args.parallelize_search,
                    monitor_search_every=args.monitor_search_every,
                    binary_search_epsilon=args.binary_search_epsilon,
                    start_search_time=start_search_time,
                    starting_progress_report_number=counter,
                    stop_at_first_iteration=args.stop_at_first_iteration,
                    model_suffix=args.model_suffix,
                    exp_suffix=args.exp_suffix,
                )
                alphatest.search()
            elif args.search_type == "random":
                random_search = RandomSearch(
                    env_variables=env_variables,
                    agent=agent,
                    algo_name=args.algo_name,
                    env_name=args.env_name,
                    tb_log_name=args.tb_log_name,
                    continue_learning_suffix=args.continue_learning_suffix,
                    buffer_file=args.buffer_file,
                    archive_file=args.archive_file,
                    executions_skipped_file=args.executions_skipped_file,
                    num_iterations=args.num_iterations,
                    param_names=param_names,
                    runs_for_probability_estimation=args.runs_for_probability_estimation,
                    parallelize_search=args.parallelize_search,
                    monitor_search_every=args.monitor_search_every,
                    binary_search_epsilon=args.binary_search_epsilon,
                    start_search_time=start_search_time,
                    starting_progress_report_number=counter,
                    stop_at_first_iteration=args.stop_at_first_iteration,
                    exp_suffix=args.exp_suffix,
                )
                random_search.search()
            logger.info(
                "########### End repetition num {}. Time elapsed: {} ###########".format(
                    i, str(datetime.timedelta(seconds=(time.time() - start_search_time)))
                )
            )
            times_elapsed.append(time.time() - start_search_time)

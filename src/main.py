import argparse
import copy
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.ndimage import uniform_filter1d
from stable_baselines.bench import load_results
from stable_baselines.results_plotter import X_EPISODES, X_TIMESTEPS

from agent import DEFAULT_N_EVAL_EPISODES, Agent
from env_utils import instantiate_env_variables, instantiate_eval_callback
from execution.runner import Runner
from log import Log
from log_utils import _ts2xy
from utilities import (PREFIX_DIR_MODELS_SAVE, SUPPORTED_ALGOS, SUPPORTED_ENVS,
                       check_param_names, check_probability_range, str2bool)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "test", "eval_model"], required=True, default="train",
    )
    parser.add_argument(
        "--algo_name", choices=SUPPORTED_ALGOS, required=True, default="ppo2",
    )
    parser.add_argument("--env_name", choices=SUPPORTED_ENVS, required=True, default="CartPole-v1")
    parser.add_argument("--log_to_tensorboard", type=bool, default=False)
    parser.add_argument("--tb_log_name", type=str, default="ppo2")
    parser.add_argument("--train_total_timesteps", type=int, default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--model_to_load", type=str, default=None)
    parser.add_argument("--continue_learning", type=bool, default=False)
    parser.add_argument("--discrete_action_space", type=bool, default=False)
    parser.add_argument("--eval_callback", type=bool, default=False)
    parser.add_argument("--continue_learning_suffix", type=str, default="continue_learning")
    parser.add_argument("--show_progress_bar", type=bool, default=False)
    parser.add_argument("--instantiate_eval_callback", type=bool, default=False)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--env_values", type=str, default=None)
    parser.add_argument("--random_seed", type=bool, default=True)
    parser.add_argument("--param_names", type=check_param_names, default=None)
    parser.add_argument("--env_pairs_pass_probabilities", type=str, default="")
    parser.add_argument("--save_replay_buffer", type=str2bool, default=True)
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--num_training_runs", type=int, default=3)
    parser.add_argument("--smoothing_window", type=check_probability_range, default=0.0)
    parser.add_argument("--execute", type=str2bool, default=True)
    parser.add_argument("--algo_hyperparams", type=str, default=None)
    parser.add_argument("--model_suffix", type=str, default=None)

    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--sb_version", type=str, default="sb2")

    parser.add_argument("--frontier_path", type=str, default=None)
    parser.add_argument("--runs_for_probability_estimation", type=int, default=1)
    args = parser.parse_args()

    if args.random_seed:
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()
    else:
        args.seed = 0

    env_kwargs = instantiate_env_variables(
        algo_name=args.algo_name,
        discrete_action_space=args.discrete_action_space,
        env_name=args.env_name,
        env_values=args.env_values,
        param_names=args.param_names,
    )
    env_eval_callback = None
    if args.instantiate_eval_callback:
        env_eval_callback = instantiate_eval_callback(env_name=args.env_name)

    if args.num_threads:
        print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)
        th.set_num_interop_threads(args.num_threads)

    logger = Log("main")

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
        env_variables=env_kwargs,
        continue_learning_suffix=args.continue_learning_suffix,
        env_eval_callback=env_eval_callback,
        show_progress_bar=args.show_progress_bar,
        log_every=args.log_every,
        save_replay_buffer=args.save_replay_buffer,
        save_model=args.save_model,
        algo_hyperparams=args.algo_hyperparams,
        sb_version=args.sb_version,
        model_suffix=args.model_suffix,
    )

    if args.mode == "train":
        agent.train(seed=args.seed)
    elif args.mode == "test":
        assert args.n_eval_episodes > 0, "The number of evaluation episodes should be > 0. Found: {}".format(
            args.n_eval_episodes
        )
        agent.test(seed=args.seed)
    elif args.mode == "eval_model":
        runner = Runner(agent=agent)
        logger.debug("Training {} different models".format(args.num_training_runs))
        current_env_variables = copy.deepcopy(env_kwargs)
        start_time = time.time()
        model_folders = []
        abs_dir = os.path.abspath(PREFIX_DIR_MODELS_SAVE + "/")
        prefix_dir = abs_dir + "/" + args.algo_name + "/logs_" + args.tb_log_name
        execution_times = []
        mean_rewards = []
        std_rewards = []
        for current_iteration in range(args.num_training_runs):
            start_time = time.time()
            if not args.continue_learning:
                model_save_path = prefix_dir + "_" + str(current_iteration)
            else:
                model_save_path = prefix_dir

            if os.path.exists(model_save_path) and args.execute and not args.continue_learning:
                shutil.rmtree(model_save_path)

            model_folders.append(model_save_path)
            if args.execute:
                if not args.continue_learning:
                    runner.execute_train_without_evaluation(
                        current_iteration=current_iteration, current_env_variables=current_env_variables
                    )
                else:
                    search_suffix = "it_" + str(current_iteration)
                    runner.execute_train_without_evaluation(
                        current_iteration=current_iteration,
                        search_suffix=search_suffix,
                        current_env_variables=current_env_variables,
                    )
                if args.save_model:
                    if not args.continue_learning:
                        mean_reward, std_reward = runner.execute_test_without_callback(
                            n_eval_episodes=100, model_path=model_save_path
                        )
                    else:
                        model_save_path_continue_learning = (
                            prefix_dir + "_" + args.continue_learning_suffix + "_it_" + str(current_iteration)
                        )
                        mean_reward, std_reward = runner.execute_test_without_callback(
                            n_eval_episodes=100, model_path=model_save_path_continue_learning
                        )
                    logger.debug("Mean reward over 100 episodes: {} +- {}".format(mean_reward, std_reward))
                    mean_rewards.append(mean_reward)
                    std_rewards.append(std_reward)

            execution_times.append(time.time() - start_time)

        logger.debug("Execution time average: {}".format(np.mean(np.array(execution_times))))
        if args.save_model:
            logger.debug(
                "Average of the mean rewards: {} +- {}".format(np.mean(np.array(mean_rewards)), np.std(np.array(mean_rewards)))
            )
            logger.debug(
                "Average of the std rewards: {} +- {}".format(np.mean(np.array(std_rewards)), np.std(np.array(std_rewards)))
            )
            distances = [abs(np.mean(mean_rewards) - mean_reward) for mean_reward in mean_rewards]
            logger.debug("Distances from average of the mean rewards: {}".format(distances))

        rewards_a = []
        limits = []
        iteration = 0
        for model_folder in model_folders:

            if args.continue_learning:
                model_folder = prefix_dir + "_" + args.continue_learning_suffix + "_it_" + str(iteration)
                iteration += 1

            # Retrieve training reward
            x_rewards, y_rewards = _ts2xy(load_results(model_folder), X_TIMESTEPS)
            x_episodes, y_episodes = _ts2xy(load_results(model_folder), X_EPISODES)

            rewards_repeated = []
            sum_rewards = 0
            for i in range(len(y_rewards)):
                sum_rewards += y_rewards[i]
                rewards_repeated.append([y_rewards[i]] * y_episodes[i])

            rewards_repeated = [item for sublist in rewards_repeated for item in sublist]
            assert x_rewards[-1] == len(
                rewards_repeated
            ), "x_rewards last element should be equal to the length of rewards repeated: {}, {}".format(
                x_rewards[-1], len(rewards_repeated)
            )
            rewards_a.append(rewards_repeated)

            limits.append(len(rewards_repeated))

        limit = np.min(np.array(limits))

        if args.smoothing_window == 0.0:
            smoothing_window = 1
        else:
            smoothing_window = int(limit * args.smoothing_window)

        rewards_a_preprocessed = []
        for rewards in rewards_a:
            rewards_a_preprocessed.append(rewards[:limit])

        avg_values = np.mean(np.array(rewards_a_preprocessed), axis=0)
        std_dev = np.std(np.array(rewards_a_preprocessed), axis=0)

        rewards_smoothed_1 = uniform_filter1d(avg_values, size=smoothing_window)
        std_smoothed_1 = uniform_filter1d(std_dev, size=smoothing_window)

        trajs = list(range(limit))

        fig = plt.figure()

        plot_title = (
            args.tb_log_name + "_" + args.algo_name + "_" + str(args.num_training_runs) + "_smooth_" + str(smoothing_window)
        )
        fig.canvas.set_window_title(plot_title)

        plt.plot(trajs, rewards_smoothed_1)
        offset = 3
        # plt.errorbar(trajs[::25 + offset], rewards_smoothed_1[::25 + offset], yerr=std_smoothed_1[::25 + offset],
        #              linestyle='None', capsize=5)
        plt.fill_between(trajs, rewards_smoothed_1 + std_smoothed_1, rewards_smoothed_1 - std_smoothed_1, alpha=0.3)
        plt.show()

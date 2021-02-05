import os
import warnings
from queue import Queue
from typing import Optional, Union, Tuple, Dict, Any

import gym
import numpy as np
import tensorflow as tf
from stable_baselines.bench import load_results
from stable_baselines.common.callbacks import BaseCallback, EventCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization, VecNormalize
from stable_baselines.results_plotter import X_EPISODES, X_TIMESTEPS

from envs.env_eval_callback import EnvEvalCallback
from evaluation import custom_evaluate_policy
from execution.execution_result import ExecutionResult
from log import Log
from log_utils import _ts2xy

import time


class EvalBaseCallback:
    def __init__(self):
        pass

    def on_eval_start(self) -> None:
        pass

    def on_eval_episode_step(self) -> bool:
        return True


class FpsCallback(BaseCallback, EvalBaseCallback):

    def __init__(self):
        super(FpsCallback, self).__init__()
        self.start_time = time.time()
        self.fps_array = []

    def _on_step(self) -> bool:
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        self.fps_array.append(fps)
        return True

    def get_fps_average(self):
        return np.array(self.fps_array).mean()


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """
    def __init__(self, log_every: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.log_every = log_every
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, '{}_{}_steps.pkl'.format(self.name_prefix, self.num_timesteps))
            else:
                path = os.path.join(self.save_path, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True


class ProgressBarCallback(BaseCallback, EvalBaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self) -> bool:
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True

    def on_eval_start(self) -> None:
        self.n_eval_episodes = 0

    def on_eval_episode_step(self) -> bool:
        self.n_eval_episodes += 1
        self._pbar.n = self.n_eval_episodes
        self._pbar.update(0)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        self._logger = Log("EvalCallback")

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.num_evaluation_steps = 0

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type" "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = (
                np.mean(episode_lengths),
                np.std(episode_lengths),
            )
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            self.num_evaluation_steps += 1

            mean_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_mean_reward", simple_value=mean_reward)])
            std_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_std_reward", simple_value=std_reward)])
            mean_ep_length_summary = tf.Summary(
                value=[tf.Summary.Value(tag="eval_mean_ep_length", simple_value=mean_ep_length)]
            )
            std_ep_length_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_std_ep_length", simple_value=std_ep_length)])
            self.locals["writer"].add_summary(mean_reward_summary, self.num_evaluation_steps)
            self.locals["writer"].add_summary(std_reward_summary, self.num_evaluation_steps)
            self.locals["writer"].add_summary(mean_ep_length_summary, self.num_evaluation_steps)
            self.locals["writer"].add_summary(std_ep_length_summary, self.num_evaluation_steps)

            if self.verbose > 0:
                self._logger.debug(
                    "Eval num_timesteps={}, "
                    "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward)
                )
                self._logger.debug("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    self._logger.debug("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

class LoggingTrainingMetricsCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param log_every: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(
        self,
        log_every: int,
        log_dir: str,
        verbose=0,
        num_envs=1,
        env_name=None,
        eval_env: Union[gym.Env, VecEnv] = None,
        original_env: Union[gym.Env, VecEnv] = None,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        env_eval_callback: EnvEvalCallback = None,
        continue_learning: bool = False,
        total_timesteps: int = 0,
        communication_queue: Queue = None,
        save_model: bool = True,
        random_search: bool = False
    ):
        super(LoggingTrainingMetricsCallback, self).__init__(verbose)
        self.log_every = log_every // num_envs
        self.total_timesteps = total_timesteps // num_envs
        self.num_envs = num_envs
        self.log_dir = log_dir
        self._logger = Log("LoggingTrainingMetricsCallback")

        self.save_path = os.path.join(log_dir, "best_model")
        self.save_path_eval = os.path.join(log_dir, "best_model_eval")
        # self.save_path_100 = os.path.join(log_dir, 'best_model_100')

        self.best_mean_reward = -np.inf
        self.best_mean_reward_eval = -np.inf
        # self.best_mean_reward_100 = -np.inf

        self.minimize_for_saving_best_model = np.inf
        self.maximize_for_saving_best_model = -self.minimize_for_saving_best_model

        self.env_name = env_name
        self.eval_env = eval_env
        self.original_env = original_env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

        self.communication_queue: Queue = communication_queue
        self.env_eval_callback = env_eval_callback
        self.continue_learning = continue_learning

        self.task_completed = False

        self.save_model = save_model
        self.random_search = random_search

        self.training_time = None

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self.training_time = time.time()

    def _init_callback(self) -> None:

        # Create folder if needed
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)

        with self.model.graph.as_default():
            with tf.variable_scope("reward_info", reuse=False):
                self.rewards_last_100_ph = tf.placeholder(tf.float32, [None], name="rewards_last_100_ph")
                mean_reward_summary = tf.summary.scalar("mean_reward", tf.math.reduce_mean(self.rewards_last_100_ph))
                std_reward_summary = tf.summary.scalar("std_reward", tf.math.reduce_std(self.rewards_last_100_ph))
                min_reward_summary = tf.summary.scalar("min_reward", tf.math.reduce_min(self.rewards_last_100_ph))
                max_reward_summary = tf.summary.scalar("max_reward", tf.math.reduce_max(self.rewards_last_100_ph))
                self.reward_summary = tf.summary.merge(
                    [mean_reward_summary, std_reward_summary, min_reward_summary, max_reward_summary,]
                )

            with tf.variable_scope("episode_length_info", reuse=False):
                self.ep_lengths_last_100_ph = tf.placeholder(tf.float32, [None], name="ep_length_last_100_ph")
                mean_ep_length_summary = tf.summary.scalar("mean_ep_length", tf.math.reduce_mean(self.ep_lengths_last_100_ph))
                std_ep_length_summary = tf.summary.scalar("std_ep_length", tf.math.reduce_std(self.ep_lengths_last_100_ph))
                min_ep_length_summary = tf.summary.scalar("min_ep_length", tf.math.reduce_min(self.ep_lengths_last_100_ph))
                max_ep_length_summary = tf.summary.scalar("max_ep_length", tf.math.reduce_max(self.ep_lengths_last_100_ph))
                self.ep_length_summary = tf.summary.merge(
                    [mean_ep_length_summary, std_ep_length_summary, min_ep_length_summary, max_ep_length_summary,]
                )

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every == 0:

            # Retrieve training reward
            x_rewards, y_rewards = _ts2xy(load_results(self.log_dir), X_TIMESTEPS)
            # Retrieve episode lengths
            x_ep_lengths, y_ep_lengths = _ts2xy(load_results(self.log_dir), X_EPISODES)

            if self.eval_env is not None and not self.env_eval_callback:
                mean_reward, std_reward = self._save_best_model_using_eval_callback()

                mean_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_mean_reward", simple_value=mean_reward)])
                if self.locals and self.locals["writer"]:
                    self.locals["writer"].add_summary(mean_reward_summary, self.num_timesteps)
                if self.n_eval_episodes > 1:
                    std_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_std_reward", simple_value=std_reward)])
                    if self.locals and self.locals["writer"]:
                        self.locals["writer"].add_summary(std_reward_summary, self.num_timesteps)
            elif len(x_rewards) > 0 and not self.continue_learning:
                self._save_best_model_using_mean(x_rewards=x_rewards, y_rewards=y_rewards)

            if self.env_eval_callback and self.communication_queue:
                self._logger.debug("Starting evaluation...")

                self._save_normalization_artifacts()

                self._logger.debug('Computing adaptation on current env: {}'.format(self.eval_env.env))
                adequate_performance, info = self.env_eval_callback.evaluate_env(
                    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
                )

                if self.n_calls == self.total_timesteps or adequate_performance:
                    regression = True
                    regression_time = 0.0
                    if adequate_performance:
                        self._logger.debug("Stopping training since performance threshold was reached")
                        if self.save_model:
                            self.model.save(self.save_path)

                        if self.original_env and not self.random_search:
                            self._logger.debug('Computing regression on original env: {}'.format(self.original_env.env))
                            start_time_regression = time.time()
                            self._logger.debug('Computing regression on_step')
                            _, info_orig = self.env_eval_callback.evaluate_env(
                                self.model, self.original_env, n_eval_episodes=100
                            )
                            mean_reward = info_orig["mean_reward"]
                            reward_threshold = self.env_eval_callback.get_reward_threshold()
                            tol = abs(reward_threshold * 5 / 100)
                            regression = mean_reward + tol < reward_threshold
                            regression_time = time.time() - start_time_regression
                            self._logger.debug('Computing regression on_step time elapsed: {} s. Mean reward: {}'.format(regression_time, mean_reward))

                    self.task_completed = True
                    # stop training
                    self.communication_queue.put(
                        ExecutionResult(adequate_performance=adequate_performance,
                                        regression=regression,
                                        regression_time=regression_time,
                                        training_time=(time.time() - self.training_time),
                                        info=info,
                                        task_completed=True,
                                        )
                    )
                else:
                    # continue training
                    self.communication_queue.put(ExecutionResult(adequate_performance=adequate_performance, info=info))
                return not adequate_performance

        return True

    def _save_best_model_using_mean(self, x_rewards: np.ndarray, y_rewards: np.ndarray) -> None:
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y_rewards[-100:])
        if self.verbose > 0:
            self._logger.debug("Num timesteps: {}".format(self.num_timesteps))
            self._logger.debug(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    self.best_mean_reward, mean_reward
                )
            )

        # New best model, you could save the agent here
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            # Example for saving best model
            if self.save_model:
                if self.verbose > 0:
                    self._logger.debug("Saving new best model to {}".format(self.save_path))
                self.model.save(self.save_path)

    def _save_normalization_artifacts(self) -> None:
        # if normalize is active
        if isinstance(self.eval_env, VecNormalize) and not self.continue_learning:
            path = os.path.join(self.log_dir, 'vecnormalize.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))

            # don't know why but rewards are still normalized
            self.eval_env = VecNormalize.load(os.path.join(self.log_dir, 'vecnormalize.pkl'), self.eval_env.unwrapped)

    def _save_best_model_using_eval_callback(self) -> Tuple[float, float]:

        self._save_normalization_artifacts()

        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)

        mean_reward, std_reward = custom_evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=False,
            deterministic=self.deterministic,
        )

        if mean_reward > self.best_mean_reward_eval:
            # if self.verbose > 0:
            self._logger.debug('{} - New best mean reward eval: {} (vs {})'
                               .format(self.num_timesteps, mean_reward, self.best_mean_reward_eval))
            self.best_mean_reward_eval = mean_reward
            # Example for saving best model
            if self.verbose > 0:
                self._logger.debug("Saving new best model to {}".format(self.save_path_eval))
            if self.save_model:
                self.model.save(self.save_path_eval)

        return mean_reward, std_reward

    def _on_training_end(self) -> None:

        x_rewards, y_rewards = _ts2xy(load_results(self.log_dir), X_TIMESTEPS)

        if self.eval_env is not None and not self.env_eval_callback:
            _, _ = self._save_best_model_using_eval_callback()
            # release resources
            self.eval_env.close()
        elif len(x_rewards) > 0 and not self.continue_learning:
            self._save_best_model_using_mean(x_rewards=x_rewards, y_rewards=y_rewards)

        if self.env_eval_callback and self.communication_queue and not self.task_completed:
            self._logger.debug("Starting evaluation...")
            adequate_performance, info = self.env_eval_callback.evaluate_env(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )

            if self.save_model:
                self.model.save(self.save_path)

            regression = True
            regression_time = 0.0
            if not adequate_performance:
                self._logger.debug("Algo was not able to cope with env changes given {} timesteps".format(self.num_timesteps))
            else:
                if self.save_model:
                    self.model.save(self.save_path)

                if self.original_env and not self.random_search:
                    start_time_regression = time.time()
                    self._logger.debug('Computing regression on_training_end')
                    _, info_orig = self.env_eval_callback.evaluate_env(
                        self.model, self.original_env, n_eval_episodes=100
                    )
                    mean_reward = info_orig["mean_reward"]
                    reward_threshold = self.env_eval_callback.get_reward_threshold()
                    tol = abs(reward_threshold * 5 / 100)
                    regression = mean_reward + tol < reward_threshold
                    regression_time = time.time() - start_time_regression
                    self._logger.debug('Computing regression on_training_end time elapsed: {} s. Mean reward {}'.format(regression_time, mean_reward))

            self.communication_queue.put(
                ExecutionResult(adequate_performance=adequate_performance,
                                regression=regression,
                                regression_time=regression_time,
                                training_time=(time.time() - self.training_time),
                                info=info,
                                task_completed=True,)
            )


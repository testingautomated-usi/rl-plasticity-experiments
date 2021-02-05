import csv
import glob
import multiprocessing
import os
import warnings
from queue import Queue
from typing import Tuple

import gym
import numpy as np
import stable_baselines3
import tensorflow as tf
import yaml
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from tensorflow.python.platform import tf_logging as tf_log

tf.get_logger().setLevel(tf_log.ERROR)
gym.logger.set_level(gym.logger.ERROR)

from stable_baselines import DQN, PPO2, SAC
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import (CheckpointCallback,
                                               EvalCallback,
                                               StopTrainingOnRewardThreshold)
from stable_baselines.common.noise import (AdaptiveParamNoiseSpec,
                                           NormalActionNoise,
                                           OrnsteinUhlenbeckActionNoise)
from stable_baselines.common.schedules import constfn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.logger import configure

from abstract_agent import AbstractAgent
from algo.env_predicate_pair import EnvPredicatePair
from custom_callbacks import (LoggingTrainingMetricsCallback,
                              SaveVecNormalizeCallback)
from custom_callbacks3 import \
    LoggingTrainingMetricsCallback as LoggingTrainingMetricsCallbackSb3
from custom_callbacks3 import \
    SaveVecNormalizeCallback as SaveVecNormalizeCallbackSb3
from env_utils import (get_n_actions, get_reward_threshold, make_custom_env,
                       make_env_parallel, normalize_env)
from envs.env_eval_callback import EnvEvalCallback
from envs.env_variables import EnvVariables
from evaluation import custom_evaluate_policy
from log import Log
from progress_bar_manager import ProgressBarManager
from training.custom_dqn import CustomDQN
from training.custom_sac import CustomSAC
from utilities import (HOME, PREFIX_DIR_MODELS_SAVE, LinearNormalActionNoise,
                       linear_schedule)

if multiprocessing.cpu_count() <= 4:
    n_cpu_tf_sess = multiprocessing.cpu_count() // 2
else:
    n_cpu_tf_sess = multiprocessing.cpu_count()


def filter_tf_version_warnings():
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    tf.autograph.set_verbosity(0)


def get_value_given_key(filename, key) -> str:
    with open(filename, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            last_row = row
        if key in last_row:
            return last_row[key]
        return None


def load_hyperparams(algo_name=None, env_name=None, model_suffix=None):
    # Load hyperparameters from yaml file
    abs_hyperparams_dir = os.path.abspath(HOME + "/hyperparams")
    filename = (
        abs_hyperparams_dir + "/{}.yml".format(algo_name)
        if not model_suffix
        else abs_hyperparams_dir + "/{}_{}.yml".format(algo_name, model_suffix)
    )
    with open(filename, "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_name in list(hyperparams_dict.keys()):
            return hyperparams_dict[env_name]
        else:
            if model_suffix:
                raise ValueError("Hyperparameters not found for {}_{}-{}".format(algo_name, model_suffix, env_name))
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(algo_name, env_name))


def _parse_normalize(dictionary):
    normalize_kwargs = {}
    if "normalize" in dictionary.keys():
        normalize = dictionary["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
        del dictionary["normalize"]

    return normalize_kwargs


DEFAULT_N_EVAL_EPISODES = 0


class Agent(AbstractAgent):
    def __init__(
        self,
        algo_name: str = "ppo2",
        env_name: str = "CartPole-v1",
        log_to_tensorboard: bool = False,
        tb_log_name: str = "ppo2",
        train_total_timesteps: int = None,
        n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
        render: bool = False,
        num_envs: int = 1,
        model_to_load: str = None,
        continue_learning: bool = False,
        continue_learning_suffix: str = "continue_learning",
        discrete_action_space: bool = False,
        eval_callback: bool = False,
        env_variables: EnvVariables = None,
        env_eval_callback: EnvEvalCallback = None,
        show_progress_bar: bool = False,
        log_every: int = 1000,
        save_replay_buffer: bool = True,
        save_model: bool = True,
        algo_hyperparams: str = None,
        sb_version: str = "sb2",
        model_suffix: str = None,
    ):

        self.algo_name = algo_name
        self.env_name = env_name
        self.log_to_tensorboard = log_to_tensorboard
        self.tb_log_name = tb_log_name
        self.train_total_timesteps = train_total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.render = render
        self.num_envs = num_envs
        self.model_to_load = model_to_load
        self.continue_learning = continue_learning
        self.continue_learning_suffix = continue_learning_suffix
        self.discrete_action_space = discrete_action_space
        self.eval_callback = eval_callback
        self.env_kwargs = env_variables
        self.env_eval_callback = env_eval_callback
        self.show_progress_bar = show_progress_bar
        self.log_every = log_every
        self.save_replay_buffer = save_replay_buffer
        self.save_model = save_model
        self.algo_hyperparams = algo_hyperparams
        self.model_suffix = model_suffix
        self.logger = Log("Agent")
        assert sb_version == "sb2" or sb_version == "sb3", "sb_version == sb2 or sb3: {}".format(sb_version)
        self.sb_version = sb_version

        filter_tf_version_warnings()
        self.logger.debug("Instantiating agent")

        if algo_name == "sac":
            assert not discrete_action_space, "discrete_action_space not supported in sac"
        elif algo_name == "dqn":
            assert discrete_action_space, "continues_action_space not supported in dqn"
        elif algo_name == "ppo2":
            self.logger.warn("PPO with {} action space".format("continuous" if not discrete_action_space else "discrete"))

    def _preprocess_hyperparams(self, _hyperparams):
        # Convert to python object if needed
        if "policy_kwargs" in _hyperparams.keys() and isinstance(_hyperparams["policy_kwargs"], str):
            _hyperparams["policy_kwargs"] = eval(_hyperparams["policy_kwargs"])

        n_timesteps = _hyperparams.pop("n_timesteps", None)
        n_envs = _hyperparams.pop("n_envs", None)
        log_every = _hyperparams.pop("log_every", None)
        if not self.continue_learning:
            if not log_every:
                self.logger.debug("log_every not defined in yml file: using command line log_every {}".format(self.log_every))
                log_every = self.log_every
            else:
                self.logger.debug("using log_every as defined in yml file: {}".format(log_every))
        else:
            self.logger.debug("priority to command line log_every {}".format(self.log_every))
            log_every = self.log_every

        # Parse noise string
        if self.algo_name in ["ddpg", "sac", "td3"] and _hyperparams.get("noise_type") is not None:
            noise_type = _hyperparams["noise_type"].strip()
            noise_std = _hyperparams["noise_std"]
            n_actions = get_n_actions(env_name=self.env_name, env_variables=self.env_kwargs)
            self.logger.debug("n_actions: {}".format(n_actions))
            if "adaptive-param" in noise_type:
                assert self.algo_name == "ddpg", "Parameter is not supported by SAC"
                _hyperparams["param_noise"] = AdaptiveParamNoiseSpec(initial_stddev=noise_std, desired_action_stddev=noise_std)
            elif "normal" in noise_type:
                if "lin" in noise_type:
                    _hyperparams["action_noise"] = LinearNormalActionNoise(
                        mean=np.zeros(n_actions),
                        sigma=noise_std * np.ones(n_actions),
                        final_sigma=_hyperparams.get("noise_std_final", 0.0) * np.ones(n_actions),
                        max_steps=n_timesteps,
                    )
                else:
                    _hyperparams["action_noise"] = NormalActionNoise(
                        mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                    )
            elif "ornstein-uhlenbeck" in noise_type:
                _hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
                )
            else:
                raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
            self.logger.debug("Applying {} noise with std {}".format(noise_type, noise_std))
            del _hyperparams["noise_type"]
            del _hyperparams["noise_std"]
            if "noise_std_final" in _hyperparams:
                del _hyperparams["noise_std_final"]

        normalize_kwargs = _parse_normalize(dictionary=_hyperparams)

        if n_envs is None:
            self.logger.debug("n_envs not defined in yml file: using command line n_envs {}".format(self.num_envs))
            n_envs = self.num_envs
        else:
            self.logger.debug("using n_envs as num of envs defined in yml file:".format(n_envs))

        if not self.continue_learning:
            # priority to yml defined n_timesteps
            if n_timesteps is None:
                self.logger.debug(
                    "n_timesteps not defined in yml file: using command line n_timesteps {}".format(self.train_total_timesteps)
                )
                n_timesteps = self.train_total_timesteps
            else:
                self.logger.debug("using n_timesteps as total timesteps defined in yml file: {}".format(n_timesteps))
                n_timesteps = int(n_timesteps)
        else:
            if self.train_total_timesteps and self.train_total_timesteps != -1:
                assert self.train_total_timesteps <= int(n_timesteps), "train_total_timesteps <= n_timesteps: {}, {}".format(
                    self.train_total_timesteps, n_timesteps
                )
                # priority to command line n_timesteps
                self.logger.debug("priority to command line n_timesteps {}".format(self.train_total_timesteps))
                n_timesteps = self.train_total_timesteps
            elif self.train_total_timesteps == -1:
                assert n_timesteps, "n_timesteps should have a value: {}".format(n_timesteps)
                n_timesteps = int(n_timesteps)
                self.logger.info("training in continual learning = training from scratch. n_timesteps {}".format(n_timesteps))
            else:
                assert n_timesteps, "n_timesteps should have a value: {}".format(n_timesteps)
                n_timesteps = int(n_timesteps // 2)
                self.logger.debug(
                    "train_total_timesteps not specified in continue_learning: "
                    "taking half of original n_timesteps defined in yml file {}".format(n_timesteps)
                )

        assert n_timesteps % log_every == 0, "it should be possible to divide n_timesteps for log_every: {}, {}".format(
            n_timesteps, log_every
        )
        return normalize_kwargs, n_envs, n_timesteps, log_every, _hyperparams

    def _preprocess_storage_dirs(self):
        if self.model_suffix:
            best_model_save_path = (
                PREFIX_DIR_MODELS_SAVE + "/" + self.algo_name + "/logs_" + self.tb_log_name + "_" + self.model_suffix
            )
        else:
            best_model_save_path = PREFIX_DIR_MODELS_SAVE + "/" + self.algo_name + "/logs_" + self.tb_log_name
        if self.log_to_tensorboard:
            tensorboard_log_dir = PREFIX_DIR_MODELS_SAVE + "/" + self.algo_name + "/logs_" + self.tb_log_name
        else:
            tensorboard_log_dir = None
        return best_model_save_path, tensorboard_log_dir

    def _set_global_seed(self, seed):
        if self.sb_version == "sb3":
            set_random_seed(seed)
        else:
            set_global_seeds(seed)

    # TODO: could be optimized when used in search (some variables can be passed instead of being created from
    #  scratch)
    def train(
        self,
        seed: int,
        communication_queue: Queue = None,
        current_iteration: int = -1,
        search_suffix: str = "1",
        env_variables: EnvVariables = None,
        random_search: bool = False,
    ):

        self._set_global_seed(seed=seed)

        env_kwargs_to_set = env_variables if env_variables else self.env_kwargs
        self.logger.debug("env_variables: {}".format(env_kwargs_to_set.get_params_string()))

        reward_threshold = get_reward_threshold(env_name=self.env_name)

        best_model_save_path, tensorboard_log_dir = self._preprocess_storage_dirs()

        if current_iteration != -1 and not self.continue_learning:
            best_model_save_path = best_model_save_path + "_" + str(current_iteration)

        self.logger.debug("best_model_save_path: {}".format(best_model_save_path))

        if communication_queue or search_suffix != "1":
            continue_learning_suffix = self.continue_learning_suffix + "_" + search_suffix
        else:
            continue_learning_suffix = self.continue_learning_suffix

        os.environ["OPENAI_LOG_FORMAT"] = "log,csv"
        if self.continue_learning:
            os.environ["OPENAI_LOGDIR"] = best_model_save_path + "_" + continue_learning_suffix
        else:
            os.environ["OPENAI_LOGDIR"] = best_model_save_path
        configure()

        if self.algo_hyperparams:
            self.logger.debug("Overriding file specified hyperparams with {}".format(eval(self.algo_hyperparams)))
            hyperparams = eval(self.algo_hyperparams)
        else:
            hyperparams = load_hyperparams(algo_name=self.algo_name, env_name=self.env_name, model_suffix=self.model_suffix)

        (normalize_kwargs, n_envs, n_timesteps, log_every, hyperparams,) = self._preprocess_hyperparams(
            _hyperparams=hyperparams
        )

        if n_envs > 1 and self.algo_name == "ppo2":
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv(
                [
                    make_env_parallel(
                        sb_version=self.sb_version,
                        seed=seed,
                        rank=i,
                        env_name=self.env_name,
                        continue_learning=self.continue_learning,
                        log_dir=best_model_save_path,
                        env_kwargs=env_kwargs_to_set,
                        algo_name=self.algo_name,
                        continue_learning_suffix=continue_learning_suffix,
                    )
                    for i in range(n_envs)
                ]
            )
            if len(normalize_kwargs) > 0:
                env = normalize_env(
                    env=env,
                    vectorize=False,
                    orig_log_dir=best_model_save_path,
                    continue_learning=self.continue_learning,
                    sb_version=self.sb_version,
                    normalize_kwargs=normalize_kwargs,
                )
        else:
            env = make_custom_env(
                seed=seed,
                sb_version=self.sb_version,
                env_kwargs=env_kwargs_to_set,
                normalize_kwargs=normalize_kwargs,
                continue_learning=self.continue_learning,
                log_dir=best_model_save_path,
                env_name=self.env_name,
                algo_name=self.algo_name,
                continue_learning_suffix=continue_learning_suffix,
            )

        if self.n_eval_episodes > DEFAULT_N_EVAL_EPISODES:
            analysis_callback = self.build_callback(
                algo_name=self.algo_name,
                continue_learning=self.continue_learning,
                call_every=log_every,
                eval_callback=self.eval_callback,
                _reward_threshold=reward_threshold,
                eval_episodes=self.n_eval_episodes,
                _eval_env=make_custom_env(
                    seed=seed,
                    continue_learning=self.continue_learning,
                    sb_version=self.sb_version,
                    env_kwargs=env_kwargs_to_set,
                    env_name=self.env_name,
                    log_dir=best_model_save_path,
                    algo_name=self.algo_name,
                    normalize_kwargs=normalize_kwargs,
                    evaluate=True,
                    evaluate_during_learning=True,
                    continue_learning_suffix=continue_learning_suffix,
                ),
                original_env=make_custom_env(
                    seed=seed,
                    continue_learning=self.continue_learning,
                    sb_version=self.sb_version,
                    env_kwargs=self.env_kwargs,
                    env_name=self.env_name,
                    log_dir=best_model_save_path,
                    algo_name=self.algo_name,
                    normalize_kwargs=normalize_kwargs,
                    evaluate=True,
                    evaluate_during_learning=True,
                ),
                env_name=self.env_name,
                _best_model_save_path=best_model_save_path,
                num_envs=n_envs,
                total_timesteps=n_timesteps,
                continue_learning_suffix=continue_learning_suffix,
                communication_queue=communication_queue,
                env_eval_callback=self.env_eval_callback,
                save_replay_buffer=self.save_replay_buffer,
                save_model=self.save_model,
                random_search=random_search,
            )
        else:
            analysis_callback = self.build_callback(
                algo_name=self.algo_name,
                continue_learning=self.continue_learning,
                call_every=log_every,
                eval_callback=self.eval_callback,
                _reward_threshold=reward_threshold,
                eval_episodes=self.n_eval_episodes,
                env_name=self.env_name,
                _best_model_save_path=best_model_save_path,
                num_envs=n_envs,
                continue_learning_suffix=continue_learning_suffix,
                save_replay_buffer=self.save_replay_buffer,
                save_model=self.save_model,
                random_search=random_search,
            )

        if self.continue_learning:
            model = self.create_model(
                seed=seed,
                algo_name=self.algo_name,
                env=env,
                tensorboard_log_dir=tensorboard_log_dir,
                hyperparams=hyperparams,
                best_model_save_path=best_model_save_path,
                n_timesteps=n_timesteps,
                continue_learning=True,
                env_name=self.env_name,
                model_to_load=self.model_to_load,
                save_replay_buffer=self.save_replay_buffer,
            )
        else:
            model = self.create_model(
                seed=seed,
                algo_name=self.algo_name,
                env=env,
                tensorboard_log_dir=tensorboard_log_dir,
                hyperparams=hyperparams,
                env_name=self.env_name,
                n_timesteps=n_timesteps,
                model_to_load=self.model_to_load,
                save_replay_buffer=self.save_replay_buffer,
            )

        try:
            callback_list = [analysis_callback]

            # if len(normalize_kwargs) > 0 and not self.continue_learning:
            #     callback_list = [self._build_vec_normalize_callback(save_path=best_model_save_path,
            #                                                         log_every=log_every), analysis_callback]

            if self.show_progress_bar:
                with ProgressBarManager(total_timesteps=n_timesteps, sb_version=self.sb_version) as progress_callback:
                    callback_list.append(progress_callback)
                    if self.continue_learning and self.log_to_tensorboard:
                        model.learn(
                            total_timesteps=n_timesteps,
                            callback=callback_list,
                            tb_log_name=self.tb_log_name + "_" + continue_learning_suffix,
                        )
                    else:
                        model.learn(
                            total_timesteps=n_timesteps, callback=callback_list, tb_log_name=self.tb_log_name,
                        )

            else:
                if self.continue_learning and self.log_to_tensorboard:
                    model.learn(
                        total_timesteps=n_timesteps,
                        callback=callback_list,
                        tb_log_name=self.tb_log_name + "_" + continue_learning_suffix,
                    )
                else:
                    self.logger.debug("Model learn start...")
                    model.learn(
                        total_timesteps=n_timesteps, callback=callback_list, tb_log_name=self.tb_log_name,
                    )
                    self.logger.debug("Model learn end")
        except KeyboardInterrupt:
            pass
        finally:
            if len(normalize_kwargs) > 0 and not self.continue_learning:
                # Important: save the running average, for testing the agent we need that normalization
                model.get_vec_normalize_env().save(os.path.join(best_model_save_path, "vecnormalize.pkl"))

            # Release resources
            env.close()

    def _build_vec_normalize_callback(self, save_path: str, log_every: int):

        if self.sb_version == "sb2":
            return SaveVecNormalizeCallback(log_every=log_every, save_path=save_path)
        return SaveVecNormalizeCallbackSb3(log_every=log_every, save_path=save_path)

    def test_with_callback(self, seed, env_variables: EnvVariables, n_eval_episodes: int = None) -> EnvPredicatePair:

        assert self.env_eval_callback, "env_eval_callback should be instantiated"

        self._set_global_seed(seed=seed)

        self.logger.debug("env_variables: {}".format(env_variables.get_params_string()))

        best_model_save_path, tensorboard_log_dir = self._preprocess_storage_dirs()

        if self.algo_hyperparams:
            self.logger.debug("Overriding file specified hyperparams with {}".format(eval(self.algo_hyperparams)))
            hyperparams = eval(self.algo_hyperparams)
        else:
            hyperparams = load_hyperparams(algo_name=self.algo_name, env_name=self.env_name)

        normalize_kwargs = _parse_normalize(dictionary=hyperparams)

        eval_env = make_custom_env(
            seed=seed,
            sb_version=self.sb_version,
            env_kwargs=env_variables,
            algo_name=self.algo_name,
            env_name=self.env_name,
            normalize_kwargs=normalize_kwargs,
            log_dir=best_model_save_path,
            evaluate=True,
            continue_learning_suffix=self.continue_learning_suffix,
        )
        model = self.create_model(
            seed=seed,
            algo_name=self.algo_name,
            env=eval_env,
            tensorboard_log_dir=tensorboard_log_dir,
            hyperparams=hyperparams,
            best_model_save_path=best_model_save_path,
            model_to_load=self.model_to_load,
            env_name=self.env_name,
        )

        n_eval_episodes_to_run = n_eval_episodes if n_eval_episodes else self.n_eval_episodes
        adequate_performance, info = self.env_eval_callback.evaluate_env(
            model=model, env=eval_env, n_eval_episodes=n_eval_episodes_to_run, sb_version=self.sb_version,
        )
        return EnvPredicatePair(env_variables=env_variables, predicate=adequate_performance, execution_info=info,)

    def test_without_callback(self, seed, n_eval_episodes: int = 0, model_path: str = None) -> Tuple[float, float]:
        assert n_eval_episodes > 0 or self.n_eval_episodes > 0, "n_eval_episodes > 0: {}, {}".format(
            n_eval_episodes, self.n_eval_episodes
        )

        self._set_global_seed(seed=seed)

        if n_eval_episodes == 0:
            n_eval_episodes = self.n_eval_episodes

        best_model_save_path, tensorboard_log_dir = self._preprocess_storage_dirs()
        if model_path:
            best_model_save_path = model_path

        if self.algo_hyperparams:
            self.logger.debug("Overriding file specified hyperparams with {}".format(eval(self.algo_hyperparams)))
            hyperparams = eval(self.algo_hyperparams)
        else:
            hyperparams = load_hyperparams(algo_name=self.algo_name, env_name=self.env_name)

        normalize_kwargs = _parse_normalize(dictionary=hyperparams)

        eval_env = make_custom_env(
            seed=seed,
            sb_version=self.sb_version,
            env_kwargs=self.env_kwargs,
            algo_name=self.algo_name,
            env_name=self.env_name,
            log_dir=best_model_save_path,
            normalize_kwargs=normalize_kwargs,
            evaluate=True,
            continue_learning_suffix=self.continue_learning_suffix,
        )
        model = self.create_model(
            seed=seed,
            algo_name=self.algo_name,
            env=eval_env,
            tensorboard_log_dir=tensorboard_log_dir,
            hyperparams=hyperparams,
            best_model_save_path=best_model_save_path,
            model_to_load=self.model_to_load,
            env_name=self.env_name,
        )

        mean_reward, std_reward = custom_evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, render=self.render, deterministic=True,
        )

        # release resources
        eval_env.close()

        return mean_reward, std_reward

    def test(self, seed, continue_learning_suffix: str = None, env_variables: EnvVariables = None):

        assert self.n_eval_episodes > 0, "n_eval_episodes > 0: {}".format(self.n_eval_episodes)

        self._set_global_seed(seed=seed)

        env_kwargs_to_set = env_variables if env_variables else self.env_kwargs
        self.logger.debug("env_variables: {}".format(env_kwargs_to_set.get_params_string()))

        best_model_save_path, tensorboard_log_dir = self._preprocess_storage_dirs()

        if self.algo_hyperparams:
            self.logger.debug("Overriding file specified hyperparams with {}".format(eval(self.algo_hyperparams)))
            hyperparams = eval(self.algo_hyperparams)
        else:
            hyperparams = load_hyperparams(algo_name=self.algo_name, env_name=self.env_name)

        normalize_kwargs = _parse_normalize(dictionary=hyperparams)

        if self.continue_learning and not continue_learning_suffix:
            best_model_save_path = best_model_save_path + "_" + self.continue_learning_suffix + "/"
        elif self.continue_learning and continue_learning_suffix:
            best_model_save_path = best_model_save_path + "_" + continue_learning_suffix + "/"

        eval_env = make_custom_env(
            seed=seed,
            sb_version=self.sb_version,
            env_kwargs=env_kwargs_to_set,
            algo_name=self.algo_name,
            env_name=self.env_name,
            log_dir=best_model_save_path,
            normalize_kwargs=normalize_kwargs,
            evaluate=True,
            continue_learning_suffix=self.continue_learning_suffix,
        )
        model = self.create_model(
            seed=seed,
            algo_name=self.algo_name,
            env=eval_env,
            tensorboard_log_dir=tensorboard_log_dir,
            hyperparams=hyperparams,
            best_model_save_path=best_model_save_path,
            model_to_load=self.model_to_load,
            env_name=self.env_name,
        )

        if self.show_progress_bar:
            with ProgressBarManager(total_timesteps=self.n_eval_episodes, sb_version=self.sb_version) as progress_callback:
                mean_reward, std_reward = custom_evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    callback=progress_callback,
                    deterministic=True,
                )
        else:
            mean_reward, std_reward = custom_evaluate_policy(
                model, eval_env, n_eval_episodes=self.n_eval_episodes, render=self.render, deterministic=True,
            )

        self.logger.debug(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # release resources
        eval_env.close()

    def load_model(self, directory, model_name=None):
        abs_dir_models = os.path.abspath(directory)
        if model_name:
            model_to_load = os.path.join(abs_dir_models, model_name + ".zip")
            self.logger.debug("Searching model file {}".format(model_to_load))
        else:
            self.logger.debug("Searching model file in directory {}".format(abs_dir_models))
            model_files = glob.glob(abs_dir_models + "/*.zip")
            model_to_load = max(model_files, key=os.path.getmtime)
        if not os.path.exists(model_to_load):
            raise FileNotFoundError("File " + model_to_load + " not found")
        self.logger.debug("Loading model file {}".format(model_to_load))
        return model_to_load

    def create_model(
        self,
        seed,
        algo_name,
        env,
        tensorboard_log_dir,
        hyperparams,
        best_model_save_path=None,
        model_to_load=None,
        continue_learning=False,
        env_name="CartPole-v1",
        n_timesteps=-1,
        save_replay_buffer: bool = True,
    ):

        old_hyperparams = dict()

        # Create learning rate schedules for ppo2 and sac
        if algo_name in ["ppo2", "sac", "td3"]:
            for key in ["learning_rate", "cliprange", "cliprange_vf"]:
                if key not in hyperparams:
                    continue
                if isinstance(hyperparams[key], str):
                    self.logger.debug("Key {}, value {}".format(key, hyperparams[key]))
                    old_hyperparams[key] = hyperparams[key]
                    schedule, initial_value = hyperparams[key].split("_")
                    initial_value = float(initial_value)
                    hyperparams[key] = linear_schedule(initial_value)
                elif isinstance(hyperparams[key], (float, int)):
                    # Negative value: ignore (ex: for clipping)
                    if hyperparams[key] < 0:
                        continue
                    old_hyperparams[key] = float(hyperparams[key])
                    hyperparams[key] = constfn(float(hyperparams[key]))
                else:
                    raise ValueError("Invalid value for {}: {}".format(key, hyperparams[key]))

        if algo_name == "ppo2":

            if self.sb_version == "sb3":
                raise NotImplementedError("PPO still in sb2")

            if best_model_save_path and continue_learning:
                model = PPO2.load(
                    self.load_model(best_model_save_path, model_to_load),
                    env=env,
                    tensorboard_log=tensorboard_log_dir,
                    verbose=1,
                )
                key = "cliprange"
                cl_cliprange_value = 0.08  # new policy can be a bit different than the old one
                if key in old_hyperparams:
                    if isinstance(old_hyperparams[key], str):
                        self.logger.debug("Setting cliprange to lin_{}".format(cl_cliprange_value))
                        model.cliprange = linear_schedule(cl_cliprange_value)
                    elif isinstance(old_hyperparams[key], (float, int)):
                        self.logger.debug("Setting cliprange to value {}".format(cl_cliprange_value))
                        model.cliprange = constfn(cl_cliprange_value)
                else:
                    # default value is too high for continual learning (0.2)
                    self.logger.debug("Setting cliprange to value {}".format(cl_cliprange_value))
                    model.cliprange = cl_cliprange_value

                return model
            elif best_model_save_path:
                return PPO2.load(
                    self.load_model(best_model_save_path, model_to_load),
                    env=env,
                    tensorboard_log=tensorboard_log_dir,
                    verbose=1,
                    n_cpu_tf_sess=n_cpu_tf_sess,
                )
            return PPO2(env=env, verbose=1, tensorboard_log=tensorboard_log_dir, **hyperparams, n_cpu_tf_sess=n_cpu_tf_sess,)

        elif algo_name == "sac":
            if self.sb_version == "sb3":
                if best_model_save_path and continue_learning:
                    model = stable_baselines3.SAC.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        seed=seed,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                    )
                    model.load_replay_buffer(path=best_model_save_path + "/replay_buffer")
                    self.logger.debug("Model replay buffer size: {}".format(model.replay_buffer.size()))
                    self.logger.debug("Setting learning_starts to 0")
                    model.learning_starts = 0

                    value = get_value_given_key(best_model_save_path + "/progress.csv", "ent_coef")
                    if value:
                        ent_coef = float(value)
                        self.logger.debug("Restore model old ent_coef: {}".format("auto_" + str(ent_coef)))
                        model.ent_coef = "auto_" + str(ent_coef)
                        model.target_entropy = str(ent_coef)

                    return model
                elif best_model_save_path:
                    return stable_baselines3.SAC.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        seed=seed,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                        n_cpu_tf_sess=n_cpu_tf_sess,
                    )
                assert n_timesteps > 0, "n_timesteps > 0: {}".format(n_timesteps)
                return stable_baselines3.SAC(env=env, verbose=0, seed=seed, tensorboard_log=tensorboard_log_dir, **hyperparams)

            else:
                if best_model_save_path and continue_learning:
                    model = CustomSAC.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                    )
                    self.logger.debug("Model replay buffer size: {}".format(len(model.replay_buffer)))
                    self.logger.debug("Setting learning_starts to 0")
                    model.learning_starts = 0
                    if not save_replay_buffer:
                        self.logger.debug("Setting save_replay_buffer to False")
                        model.save_replay_buffer = False

                    value = get_value_given_key(best_model_save_path + "/progress.csv", "ent_coef")
                    if value:
                        ent_coef = float(value)
                        self.logger.debug("Restore model old ent_coef: {}".format("auto_" + str(ent_coef)))
                        model.ent_coef = "auto_" + str(ent_coef)
                        model.target_entropy = str(ent_coef)

                    return model

                elif best_model_save_path:
                    # do not load replay buffer since we are in testing mode (no continue_learning)
                    return SAC.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                        n_cpu_tf_sess=n_cpu_tf_sess,
                    )
                return CustomSAC(
                    total_timesteps=n_timesteps,
                    env=env,
                    verbose=1,
                    tensorboard_log=tensorboard_log_dir,
                    **hyperparams,
                    n_cpu_tf_sess=n_cpu_tf_sess,
                    save_replay_buffer=save_replay_buffer,
                )

        elif algo_name == "dqn":

            if self.sb_version == "sb3":

                if best_model_save_path:
                    if continue_learning:
                        model = stable_baselines3.DQN.load(
                            self.load_model(best_model_save_path, model_to_load),
                            env=env,
                            seed=seed,
                            tensorboard_log=tensorboard_log_dir,
                            verbose=0,
                        )
                        model.load_replay_buffer(path=best_model_save_path + "/replay_buffer")
                        model.learning_starts = 0
                        model.exploration_fraction = 0.0005
                        model.exploration_initial_eps = model.exploration_final_eps
                        model.exploration_schedule = get_linear_fn(
                            model.exploration_initial_eps, model.exploration_final_eps, model.exploration_fraction
                        )
                        self.logger.debug("Model replay buffer size: {}".format(model.replay_buffer.size()))
                        self.logger.debug("Setting learning_starts to {}".format(model.learning_starts))
                        self.logger.debug("Setting exploration_fraction to {}".format(model.exploration_fraction))
                        self.logger.debug("Setting exploration_initial_eps to {}".format(model.exploration_initial_eps))
                        return model
                    return stable_baselines3.DQN.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        seed=seed,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                    )
                return stable_baselines3.DQN(env=env, verbose=0, seed=seed, tensorboard_log=tensorboard_log_dir, **hyperparams)
            else:
                if best_model_save_path:
                    if continue_learning:
                        model = CustomDQN.load(
                            self.load_model(best_model_save_path, model_to_load),
                            env=env,
                            tensorboard_log=tensorboard_log_dir,
                            verbose=1,
                        )
                        self.logger.debug("Model replay buffer size: {}".format(len(model.replay_buffer)))
                        self.logger.debug(
                            "Setting exploration initial eps to exploration final eps {}".format(model.exploration_final_eps)
                        )
                        self.logger.debug("Setting learning_starts to 0")
                        if not save_replay_buffer:
                            self.logger.debug("Setting save_replay_buffer to False")
                            model.save_replay_buffer = False
                        model.learning_starts = 0
                        model.exploration_fraction = 0.005
                        model.exploration_initial_eps = model.exploration_final_eps
                        return model
                    return DQN.load(
                        self.load_model(best_model_save_path, model_to_load),
                        env=env,
                        tensorboard_log=tensorboard_log_dir,
                        verbose=1,
                        n_cpu_tf_sess=n_cpu_tf_sess,
                    )
                return CustomDQN(
                    env=env,
                    save_replay_buffer=save_replay_buffer,
                    verbose=1,
                    tensorboard_log=tensorboard_log_dir,
                    **hyperparams,
                    n_cpu_tf_sess=n_cpu_tf_sess,
                )
        raise NotImplementedError("algo_name {} not supported yet".format(algo_name))

    def build_checkpoint_callback(self, save_freq=10000, save_path=None):
        self.logger.debug("Checkpoint callback called every {} timesteps".format(save_freq))
        return CheckpointCallback(save_freq=save_freq, save_path=save_path)

    def build_logging_training_metrics_callback(
        self,
        algo_name="ppo2",
        env_name=None,
        log_every=1000,
        save_path=None,
        num_envs=1,
        _eval_env=None,
        original_env=None,
        total_timesteps=0,
        n_eval_episodes=10,
        communication_queue=None,
        env_eval_callback=None,
        continue_learning=False,
        save_replay_buffer=True,
        save_model=True,
        random_search=False,
    ):
        self.logger.debug("Logging training metrics callback called every {}".format(log_every))
        if self.sb_version == "sb3":
            return LoggingTrainingMetricsCallbackSb3(
                log_every=log_every,
                log_dir=save_path,
                num_envs=num_envs,
                env_name=env_name,
                total_timesteps=total_timesteps,
                eval_env=_eval_env,
                original_env=original_env,
                n_eval_episodes=n_eval_episodes,
                communication_queue=communication_queue,
                env_eval_callback=env_eval_callback,
                continue_learning=continue_learning,
                save_replay_buffer=save_replay_buffer,
                save_model=save_model,
                random_search=random_search,
            )
        return LoggingTrainingMetricsCallback(
            log_every=log_every,
            log_dir=save_path,
            num_envs=num_envs,
            env_name=env_name,
            total_timesteps=total_timesteps,
            eval_env=_eval_env,
            original_env=original_env,
            n_eval_episodes=n_eval_episodes,
            communication_queue=communication_queue,
            env_eval_callback=env_eval_callback,
            continue_learning=continue_learning,
            save_model=save_model,
            random_search=random_search,
        )

    def build_eval_callback(
        self, eval_freq=10000, reward_threshold=900, log_path=None, eval_episodes=10, eval_env=None,
    ):
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=eval_episodes,
            callback_on_new_best=callback_on_best,
            verbose=1,
        )
        self.logger.debug(
            "Eval callback called every {} timesteps: stop training when mean reward is above {} in {} episodes".format(
                eval_freq, reward_threshold, eval_episodes
            )
        )
        return eval_callback

    def build_callback(
        self,
        algo_name="ppo2",
        continue_learning=False,
        call_every=1000,
        eval_callback=False,
        _reward_threshold=900,
        eval_episodes=10,
        _eval_env=None,
        original_env=None,
        _best_model_save_path=None,
        num_envs=1,
        env_name=None,
        continue_learning_suffix="continue_learning",
        communication_queue=None,
        env_eval_callback=None,
        total_timesteps=0,
        save_replay_buffer=True,
        save_model=True,
        random_search=False,
    ):
        if continue_learning:
            save_path = _best_model_save_path + "_" + continue_learning_suffix + "/"
        else:
            save_path = _best_model_save_path

        if eval_callback:
            return self.build_eval_callback(
                eval_env=_eval_env,
                eval_freq=call_every,
                reward_threshold=_reward_threshold,
                log_path=save_path,
                eval_episodes=eval_episodes,
            )
        else:
            if _eval_env:
                return self.build_logging_training_metrics_callback(
                    algo_name=algo_name,
                    env_name=env_name,
                    log_every=call_every,
                    save_path=save_path,
                    num_envs=num_envs,
                    _eval_env=_eval_env,
                    original_env=original_env,
                    n_eval_episodes=eval_episodes,
                    communication_queue=communication_queue,
                    env_eval_callback=env_eval_callback,
                    total_timesteps=total_timesteps,
                    continue_learning=continue_learning,
                    save_replay_buffer=save_replay_buffer,
                    save_model=save_model,
                    random_search=random_search,
                )
            return self.build_logging_training_metrics_callback(
                algo_name=algo_name,
                env_name=env_name,
                log_every=call_every,
                save_path=save_path,
                num_envs=num_envs,
                save_replay_buffer=save_replay_buffer,
                save_model=save_model,
            )

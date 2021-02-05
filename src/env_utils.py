import os
from typing import Dict

import gym
from gym.wrappers import TimeLimit
from stable_baselines3.common import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnv3
from stable_baselines3.common.vec_env import SubprocVecEnv as SubprocVecEnv3
from stable_baselines3.common.vec_env import VecFrameStack as VecFrameStack3
from stable_baselines3.common.vec_env import VecNormalize as VecNormalize3
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import (DummyVecEnv, VecFrameStack,
                                             VecNormalize)

from envs.acrobot_env_variables import AcrobotEnvVariables
from envs.acrobot_env_wrapper import AcrobotEnvWrapper
from envs.acrobot_eval_callback import AcrobotEvalCallback
from envs.cartpole_env_variables import CartPoleEnvVariables
from envs.cartpole_env_wrapper import CartPoleEnvWrapper
from envs.cartpole_eval_callback import CartPoleEvalCallback
from envs.env_eval_callback import EnvEvalCallback
from envs.env_variables import EnvVariables
from envs.mountaincar_env_variables import MountainCarEnvVariables
from envs.mountaincar_env_wrapper import MountainCarEnvWrapper
from envs.mountaincar_eval_callback import MountainCarEvalCallback
from envs.pendulum_env_variables import PendulumEnvVariables
from envs.pendulum_env_wrapper import PendulumEnvWrapper
from envs.pendulum_eval_callback import PendulumEvalCallback
from log import Log

logger = Log("env_utils")


def normalize_env(
    env,
    orig_log_dir,
    sb_version,
    vectorize=True,
    continue_learning=False,
    evaluate=False,
    evaluate_during_learning=False,
    normalize_kwargs=None,
):
    if vectorize:
        env = DummyVecEnv([lambda: env])

    logger.debug("Normalize: {}".format(normalize_kwargs))
    if evaluate:
        # FIXME in continue learning training should be True so that we update the running average of obs and
        #  rewards with new samples; if I do that, the algo performs very poorly even with no changes in the env
        if sb_version == "sb3":
            env = VecNormalize3(env, training=False, **normalize_kwargs)
        else:
            env = VecNormalize(env, training=False, **normalize_kwargs)

        if not evaluate_during_learning or continue_learning:
            if not os.path.exists(os.path.join(orig_log_dir, "vecnormalize.pkl")):
                env_name = get_env_name(env=env.unwrapped, sb_version=sb_version)
                index_last_separator = orig_log_dir.rindex("/")
                new_orig_log_dir = os.path.join(orig_log_dir[0:index_last_separator], "logs_" + env_name)
                logger.debug(
                    "{} does not exist. Trying to search it in the original model directory {}".format(
                        os.path.join(orig_log_dir, "vecnormalize.pkl"), new_orig_log_dir
                    )
                )
                assert os.path.exists(new_orig_log_dir), "{} does not exist"
                assert os.path.exists(os.path.join(new_orig_log_dir, "vecnormalize.pkl")), (
                    os.path.join(new_orig_log_dir, "vecnormalize.pkl") + " does not exist"
                )
                logger.debug("[evaluate] Loading {}".format(os.path.join(new_orig_log_dir, "vecnormalize.pkl")))
                if sb_version == "sb3":
                    env = VecNormalize3.load(os.path.join(new_orig_log_dir, "vecnormalize.pkl"), env)
                else:
                    env = VecNormalize.load(os.path.join(new_orig_log_dir, "vecnormalize.pkl"), env)
            else:
                logger.debug("[evaluate] Loading {}".format(os.path.join(orig_log_dir, "vecnormalize.pkl")))
                if sb_version == "sb3":
                    env = VecNormalize3.load(os.path.join(orig_log_dir, "vecnormalize.pkl"), env)
                else:
                    env = VecNormalize.load(os.path.join(orig_log_dir, "vecnormalize.pkl"), env)

        # Deactivate training and reward normalization
        env.training = False
        env.norm_reward = False

    elif continue_learning:
        # FIXME: don't know why but during continue learning I have to disable training otherwise performance
        #  is not the same as in the model trained from scratch even without changing the params of the environment.
        #  in rl-baselines-zoo this is not done during continue learning:
        #  https://github.com/araffin/rl-baselines-zoo/blob/master/train.py#L365
        if sb_version == "sb3":
            env = VecNormalize3(env, training=False, **normalize_kwargs)
        else:
            env = VecNormalize(env, training=False, **normalize_kwargs)

        assert os.path.exists(os.path.join(orig_log_dir, "vecnormalize.pkl")), (
            os.path.join(orig_log_dir, "vecnormalize.pkl") + " does not exist"
        )
        logger.debug("[continue_learning] Loading {}".format(os.path.join(orig_log_dir, "vecnormalize.pkl")))
        if sb_version == "sb3":
            env = VecNormalize3.load(os.path.join(orig_log_dir, "vecnormalize.pkl"), env)
        else:
            env = VecNormalize.load(os.path.join(orig_log_dir, "vecnormalize.pkl"), env)

    else:
        if sb_version == "sb3":
            env = VecNormalize3(env, **normalize_kwargs)
        else:
            env = VecNormalize(env, **normalize_kwargs)

    return env


def make_env_parallel(
    sb_version,
    seed,
    rank=0,
    env_name="CartPole-v1",
    continue_learning=False,
    log_dir=None,
    env_kwargs: EnvVariables = None,
    algo_name="ppo2",
    continue_learning_suffix="continue_learning",
):
    if continue_learning and log_dir:
        log_dir = log_dir + "_" + continue_learning_suffix + "/"

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        if sb_version == "sb3":
            set_random_seed(seed + rank)
        else:
            set_global_seeds(seed + rank)

        # this seed will be overridden by the last but one statement of this function
        env = make_custom_env(seed=0, sb_version=sb_version, env_kwargs=env_kwargs, env_name=env_name, algo_name=algo_name)

        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        info_keywords = ()

        env = Monitor(env, log_file, info_keywords=info_keywords)
        env.seed(seed + rank)

        return env

    return _init


def get_reward_threshold(env_name="CartPole-v1"):
    env = gym.make(env_name)
    reward_threshold = env.spec.reward_threshold
    env.close()
    return reward_threshold


def get_env_name(env, sb_version: str = "sb2") -> str:

    if sb_version == "sb3":
        while isinstance(env, VecNormalize3) or isinstance(env, VecFrameStack3):
            env = env.venv
    else:
        while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
            env = env.venv

    if sb_version == "sb3":
        if isinstance(env, DummyVecEnv3):
            assert len(env.envs) >= 0, "num_envs > 0: {}".format(len(env.envs))
            env = env.envs[0]
    else:
        if isinstance(env, DummyVecEnv):
            assert len(env.envs) >= 0, "num_envs > 0: {}".format(len(env.envs))
            env = env.envs[0]

    if sb_version == "sb3":
        if isinstance(env, SubprocVecEnv3):
            raise NotImplementedError("SubprocVecEnv not supported")
    else:
        if isinstance(env, DummyVecEnv):
            raise NotImplementedError("SubprocVecEnv not supported")

    env_string = env.__str__()
    index_last_angular_bracket_open = env_string.rindex("<")
    index_last_angular_bracket_closed = env_string.rindex(">")
    if "instance" in env_string:
        to_standardize = env_string[index_last_angular_bracket_open + 1 : index_last_angular_bracket_closed - 1]
        to_standardize = to_standardize.replace(" instance", "")
    else:
        to_standardize = env_string[index_last_angular_bracket_open + 1 : index_last_angular_bracket_closed - 2]

    return standardize_env_name(to_standardize)


def get_n_actions(env_name, env_variables: EnvVariables):
    env = make_custom_env(seed=0, sb_version="sb2", env_name=env_name, env_kwargs=env_variables)
    n_actions = env.action_space.shape[0]
    env.close()
    return n_actions


def make_custom_env(
    seed,
    sb_version,
    env_kwargs: EnvVariables = None,
    env_name="CartPole-v1",
    continue_learning=False,
    log_dir=None,
    algo_name="ppo2",
    evaluate=False,
    evaluate_during_learning=False,
    normalize_kwargs=None,
    continue_learning_suffix="continue_learning",
):
    orig_log_dir = log_dir
    if continue_learning and log_dir:
        log_dir = log_dir + "_" + continue_learning_suffix + "/"

    if normalize_kwargs is None:
        normalize_kwargs = {}

    info_keywords = ()

    if env_name == "CartPole-v1":
        cartpole_env_params = env_kwargs.instantiate_env()
        env = CartPoleEnvWrapper(**cartpole_env_params)
        env = TimeLimit(env, max_episode_steps=500)

    elif env_name == "Pendulum-v0":
        pendulum_env_params = env_kwargs.instantiate_env()
        env = PendulumEnvWrapper(**pendulum_env_params)
        env = TimeLimit(env, max_episode_steps=200)

    elif env_name == "MountainCar-v0" and algo_name != "sac":
        mountaincar_env_params = env_kwargs.instantiate_env()
        env = MountainCarEnvWrapper(**mountaincar_env_params)
        env = TimeLimit(env, max_episode_steps=200)

    elif env_name == "MountainCar-v0" and algo_name == "sac":
        mountaincar_env_params = env_kwargs.instantiate_env()
        env = MountainCarEnvWrapper(**mountaincar_env_params)
        env = TimeLimit(env, max_episode_steps=999)

    elif env_name == "Acrobot-v1":
        acrobot_env_params = env_kwargs.instantiate_env()
        env = AcrobotEnvWrapper(**acrobot_env_params)
        env = TimeLimit(env, max_episode_steps=500)

    else:
        env = gym.make(env_name)

    if log_dir is not None and not evaluate:
        log_file = os.path.join(log_dir, "0")
        logger.debug("Saving monitor files in {}".format(log_file))
        env = Monitor(env, log_file, info_keywords=info_keywords)

    if len(normalize_kwargs) > 0:
        env = normalize_env(
            env=env,
            sb_version=sb_version,
            orig_log_dir=orig_log_dir,
            continue_learning=continue_learning,
            evaluate=evaluate,
            evaluate_during_learning=evaluate_during_learning,
            normalize_kwargs=normalize_kwargs,
        )

    if (
        len(normalize_kwargs) == 0
        and not evaluate_during_learning
        and ((evaluate and algo_name == "ppo2") or (continue_learning and algo_name == "ppo2"))
    ):
        env = DummyVecEnv([lambda: env])

    env.seed(seed)
    return env


def standardize_env_name(env_name: str) -> str:
    assert (
        len(env_name.split("-")) == 2 or "Env" in env_name
    ), "env_name should be splittable with - or it should contain Env: {}".format(env_name)
    if len(env_name.split("-")) != 2:
        env_name = env_name.replace("Env", "")
        return env_name.lower()
    return env_name.lower().split("-")[0]


def instantiate_env_variables(
    algo_name, discrete_action_space, env_name, param_names=None, env_values=None, model_suffix: str = None
) -> EnvVariables:
    evaluated_env_values = None
    if env_values:
        if isinstance(env_values, str):
            evaluated_env_values = eval(env_values)
        elif isinstance(env_values, Dict):
            evaluated_env_values = env_values
        else:
            raise NotImplementedError("env_values must be an instance of str o Dict. env_values: {}".format(env_values))

    if standardize_env_name(env_name=env_name) == "cartpole":
        if env_values and evaluated_env_values:
            _env_variables = CartPoleEnvVariables(
                **evaluated_env_values,
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix
            )
        else:
            _env_variables = CartPoleEnvVariables(
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix,
            )

    elif standardize_env_name(env_name=env_name) == "pendulum":
        if env_values and evaluated_env_values:
            _env_variables = PendulumEnvVariables(
                **evaluated_env_values,
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix
            )
        else:
            _env_variables = PendulumEnvVariables(
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix,
            )

    elif standardize_env_name(env_name=env_name) == "mountaincar":
        if env_values and evaluated_env_values:
            _env_variables = MountainCarEnvVariables(
                **evaluated_env_values,
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix
            )
        else:
            _env_variables = MountainCarEnvVariables(
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix,
            )

    elif standardize_env_name(env_name=env_name) == "acrobot":
        if env_values and evaluated_env_values:
            _env_variables = AcrobotEnvVariables(
                **evaluated_env_values,
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix
            )
        else:
            _env_variables = AcrobotEnvVariables(
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                param_names=param_names,
                model_suffix=model_suffix,
            )

    else:
        raise NotImplementedError("Env {} not supported".format(env_name))

    return _env_variables


def instantiate_eval_callback(env_name) -> EnvEvalCallback:
    if env_name == "CartPole-v1":
        reward_threshold = 475.0
        unacceptable_pct_degradation = 20.0
        _env_eval_callback = CartPoleEvalCallback(
            reward_threshold=reward_threshold, unacceptable_pct_degradation=unacceptable_pct_degradation,
        )
    elif env_name == "Pendulum-v0":
        reward_threshold = -145.0
        unacceptable_pct_degradation = 30.0
        _env_eval_callback = PendulumEvalCallback(
            reward_threshold=reward_threshold, unacceptable_pct_degradation=unacceptable_pct_degradation,
        )
    elif env_name == "MountainCar-v0":
        reward_threshold = -110.0
        unacceptable_pct_degradation = 15.0
        _env_eval_callback = MountainCarEvalCallback(
            reward_threshold=reward_threshold, unacceptable_pct_degradation=unacceptable_pct_degradation,
        )
    elif env_name == "Acrobot-v1":
        reward_threshold = -90.0
        unacceptable_pct_degradation = 20.0
        _env_eval_callback = AcrobotEvalCallback(
            reward_threshold=reward_threshold, unacceptable_pct_degradation=unacceptable_pct_degradation,
        )
    else:
        raise NotImplementedError("Env {} not supported".format(env_name))

    return _env_eval_callback

import argparse
import multiprocessing
from typing import List

import numpy as np
from stable_baselines.common.noise import ActionNoise
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
import os
import sys

from statistics.effect_size import vargha_delaney
from statistics.effect_size import cohend
from statistics.power_analysis import parametric_power_analysis
from statistics.wilcoxon import summary, mannwhitney_test


def parse_at_prefixed_params(args, param_name):
    try:
        index = args.index('--' + param_name)
        return args[index + 1]
    except ValueError:
        return None


home_abs_path = parse_at_prefixed_params(args=sys.argv, param_name='home_abs_path')
logging_level = parse_at_prefixed_params(args=sys.argv, param_name='logging_level')
num_of_threads = parse_at_prefixed_params(args=sys.argv, param_name='num_of_threads')

HOME = home_abs_path if home_abs_path else '..'
LOGGING_LEVEL = logging_level.upper() if logging_level else 'DEBUG'
NUM_OF_THREADS = int(num_of_threads) if num_of_threads else multiprocessing.cpu_count() - 1  # account for main thread
assert LOGGING_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
PREFIX_DIR_MODELS_SAVE = HOME + '/rl-trained-agents'

SUPPORTED_ALGOS = ['ppo2', 'sac', 'dqn']
SUPPORTED_ENVS = ['CartPole-v1', 'Pendulum-v0', 'MountainCar-v0', 'Acrobot-v1']
CONTINUAL_LEARNING_RANGE_MULTIPLIER = 4

# ================== Custom Policies =================


class TinyDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(TinyDQNPolicy, self).__init__(*args, **kwargs, layers=[64], layer_norm=True, feature_extraction="mlp")


class MediumDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MediumDQNPolicy, self).__init__(*args, **kwargs, layers=[256, 256], layer_norm=True,
                                              feature_extraction="mlp")


class LargeDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(LargeDQNPolicy, self).__init__(
            *args, **kwargs, layers=[256, 256, 256], layer_norm=True, feature_extraction="mlp"
        )


class HugeDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(HugeDQNPolicy, self).__init__(
            *args, **kwargs, layers=[256, 256, 256, 256], layer_norm=True, feature_extraction="mlp"
        )


class BigBigDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(BigBigDQNPolicy, self).__init__(
            *args, **kwargs, layers=[256, 256, 256, 256, 256], layer_norm=True, feature_extraction="mlp"
        )


class BigBigBigDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(BigBigBigDQNPolicy, self).__init__(
            *args, **kwargs, layers=[256, 256, 256, 256, 256, 256], layer_norm=True, feature_extraction="mlp"
        )


class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs, layers=[16], feature_extraction="mlp")


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")


class TinySACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(TinySACPolicy, self).__init__(*args, **kwargs, layers=[256], feature_extraction="mlp")


class LargeSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(LargeSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256, 256], feature_extraction="mlp")


class LargeBasePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(LargeBasePolicy, self).__init__(*args, **kwargs, layers=[256, 256, 256], feature_extraction="mlp")


class MediumBasePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(MediumBasePolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")


register_policy("CustomSACPolicy", CustomSACPolicy)
register_policy("TinySACPolicy", TinySACPolicy)
register_policy("LargeSACPolicy", LargeSACPolicy)
register_policy("LargeBasePolicy", LargeBasePolicy)
register_policy("MediumBasePolicy", MediumBasePolicy)
register_policy("TinyDQNPolicy", TinyDQNPolicy)
register_policy("MediumDQNPolicy", MediumDQNPolicy)
register_policy("LargeDQNPolicy", LargeDQNPolicy)
register_policy("HugeDQNPolicy", HugeDQNPolicy)
register_policy("BigBigDQNPolicy", BigBigDQNPolicy)
register_policy("BigBigBigDQNPolicy", BigBigBigDQNPolicy)
register_policy("CustomMlpPolicy", CustomMlpPolicy)


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


class LinearNormalActionNoise(ActionNoise):
    """
    A gaussian action noise with linear decay for the standard deviation.
    :param mean: (np.ndarray) the mean value of the noise
    :param sigma: (np.ndarray) the scale of the noise (std here)
    :param max_steps: (int)
    :param final_sigma: (np.ndarray)
    """

    def __init__(self, mean, sigma, max_steps, final_sigma=None):
        self._mu = mean
        self._sigma = sigma
        self._step = 0
        self._max_steps = max_steps
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma

    def __call__(self):
        t = min(1.0, self._step / self._max_steps)
        sigma = (1 - t) * self._sigma + t * self._final_sigma
        self._step += 1
        return np.random.normal(self._mu, sigma)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_probability_range(arg):
    try:
        value = float(arg)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))

    if value < 0.0 or value > 1.0:
        message = "Expected 0.0 <= value <= 0.1, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)
    return value


def check_halve_or_double(halve_or_double) -> str:
    if halve_or_double == 'halve' or halve_or_double == 'double':
        return halve_or_double

    message = "halve_or_double == halve or double, {}".format(halve_or_double)
    raise argparse.ArgumentTypeError(message)


def check_param_names(csv: str) -> List[str]:
    try:
        param_names = csv.split(sep=",")
        if len(param_names) <= 1:
            raise SyntaxError('At least 2 param names must be specified: {}'.format(csv))
    except Exception:
        raise SyntaxError('param names must be comma separated: {}'.format(csv))
    return param_names


def check_file_existence(file_name: str) -> str:
    assert os.path.exists(file_name), 'file_name {} does not exist'.format(file_name)
    return file_name


def filter_resampling_artifacts(files: List[str]) -> List[str]:
    return list(filter(lambda x: '_resampling' not in x, files))


def get_result_dir_iteration_number(dirname: str) -> int:
    assert os.path.isdir(dirname), 'dirname {} must be a directory'.format(dirname)
    dirname = dirname.replace('_resampling', '')
    index_last_underscore = dirname.rindex('_')
    return int(dirname[index_last_underscore + 1:])


def get_result_file_iteration_number(filename: str) -> int:
    filename = filename.replace('_resampling', '')
    index_last_underscore = filename.rindex('_')
    index_last_dot = filename.rindex('.')
    return int(filename[index_last_underscore + 1:index_last_dot])


def compute_statistics(a: List[float], b: List[float], _logger, alpha: float = 0.05, power=0.80,
                       only_summary: bool = False, the_higher_the_better: bool = False):
    mean_0, std_0, min_0, max_0 = summary(a=a)
    mean_1, std_1, min_1, max_1 = summary(a=b)
    _logger.info('summary first mode. Mean: {}, Std: {}, Min: {}, Max: {}'.format(mean_0, std_0, min_0, max_0))
    _logger.info('summary second mode. Mean: {}, Std: {}, Min: {}, Max: {}'.format(mean_1, std_1, min_1, max_1))
    if mean_0 != 0.0 and mean_1 != 0.0:
        if not only_summary:
            try:
                _, p_value = mannwhitney_test(a=a, b=b)
                _logger.info('wilcoxon: p_value = {}'.format(p_value))
                if p_value > alpha:
                    estimate, magnitude = cohend(a=a, b=b)
                    _logger.info('cohen\'s d: {}, {}'.format(estimate, magnitude))
                    sample_size = parametric_power_analysis(effect=estimate, alpha=alpha, power=power)
                    _logger.info('sample size required for alpha {} and power {}: {}'.format(alpha, power, sample_size))
                estimate, magnitude = vargha_delaney(a=a, b=b)
                if the_higher_the_better:
                    estimate = 1 - estimate
                _logger.info('vargha_delaney: estimate = {}, magnitude = {}'.format(estimate, magnitude))
            except ValueError:
                _logger.warn('not possible to compute statistical test')


def norm(env_vars_1, env_vars_2=None) -> float:
    if env_vars_2:
        return np.linalg.norm(np.array(env_vars_1.get_values()) - np.array(env_vars_2.get_values()))
    return np.linalg.norm(np.array(env_vars_1.get_values()))


def get_num_digits_after_floating_point(number: float) -> int:
    return len(str(number).split('.')[1])


def get_num_digits_before_floating_point(number: float) -> int:
    return len(str(abs(number)).split('.')[0])

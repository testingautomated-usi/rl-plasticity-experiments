from typing import Tuple, Dict

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy as evaluate_policy_sb3

from envs.env_eval_callback import EnvEvalCallback
from log import Log


class MountainCarEvalCallback(EnvEvalCallback):

    # i.e. performance is not adequate when there is a 20% degradation wrt reward threshold
    def __init__(self, reward_threshold=-110.0, unacceptable_pct_degradation=20.0):
        self.reward_threshold = reward_threshold
        self.unacceptable_pct_degradation = unacceptable_pct_degradation
        self.logger = Log('MountainCarEvalCallback')

    def evaluate_env(self, model, env, n_eval_episodes, sb_version='sb2') -> Tuple[bool, Dict]:

        if sb_version == 'sb2':
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=False,
                                                      deterministic=True)
        elif sb_version == 'sb3':
            mean_reward, std_reward = evaluate_policy_sb3(model, env, n_eval_episodes=n_eval_episodes,
                                                          render=False, deterministic=True)
        else:
            raise NotImplemented('sb_version can be either sb2 or sb3. Found: {}'.format(sb_version))

        percentage_drop = abs(100.0 - (100.0 * mean_reward) / self.reward_threshold) \
            if mean_reward < self.reward_threshold else 0.0
        self.logger.debug('Mean reward: {}, Std reward: {}, Percentage drop: {}'
                          .format(mean_reward, std_reward, percentage_drop))
        adequate_performance = mean_reward > (self.reward_threshold -
                                              abs((self.reward_threshold * self.unacceptable_pct_degradation / 100.0)))
        info = dict()
        info["mean_reward"] = mean_reward
        info["std_reward"] = std_reward
        info["percentage_drop"] = percentage_drop

        # release resources
        env.close()

        return adequate_performance, info

    def get_reward_threshold(self) -> float:
        return self.reward_threshold



import numpy as np
from stable_baselines.common.vec_env import VecEnv, unwrap_vec_normalize


def custom_evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    return_episode_info=False,
):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    # if isinstance(env, VecNormalize):
    #     env = env.venv

    _vec_normalize_env = unwrap_vec_normalize(env)

    episode_rewards, episode_lengths, episodes_info = [], [], []
    if callback:
        callback.on_eval_start()

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_info = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            if _vec_normalize_env is not None:
                reward_ = _vec_normalize_env.get_original_reward().squeeze()
                episode_reward += reward_
            else:
                episode_reward += reward
            episode_info.append(_info)
            episode_length += 1
            if render:
                env.render("human")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episodes_info.append(episode_info)
        if callback:
            if not callback.on_eval_episode_step():
                break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " "{:.2f} < {:.2f}".format(
            mean_reward, reward_threshold
        )
    if return_episode_info and not return_episode_rewards:
        return mean_reward, std_reward, episodes_info
    elif return_episode_info and return_episode_rewards:
        return episode_rewards, episode_lengths, episodes_info

    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward

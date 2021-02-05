import numpy as np
import tensorflow as tf
from stable_baselines import DQN, logger
from stable_baselines.common import SetVerbosity, TensorboardWriter, tf_util
from stable_baselines.common.buffers import PrioritizedReplayBuffer, ReplayBuffer
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import VecEnv


class CustomDQN(DQN):
    """

    Custom version of DQN (DQN).
    It is adapted from the stable-baselines version.

    Notable changes:
    - save replay buffer and restore it while loading

    """

    def __init__(self, save_replay_buffer: bool = True, **kwargs):
        super(CustomDQN, self).__init__(**kwargs)
        self.save_replay_buffer = save_replay_buffer

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="DQN",
        reset_num_timesteps=True,
        replay_wrapper=None,
    ):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(
            self.graph, self.tensorboard_log, tb_log_name, new_tb_log
        ) as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                if self.replay_buffer and len(self.replay_buffer) > 0:
                    # TODO: maybe substitute with a prioritized buffer to give preference to the transitions added
                    # during continual learning
                    pass
                else:
                    self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(
                    prioritized_replay_beta_iters, initial_p=self.prioritized_replay_beta0, final_p=1.0,
                )
            else:
                if self.replay_buffer and len(self.replay_buffer) > 0:
                    # TODO: maybe substitute with a prioritized buffer to give preference to the transitions added
                    # during continual learning
                    pass
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )

            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for _ in range(total_timesteps):
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.0
                else:
                    update_eps = 0.0
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1.0
                        - self.exploration.value(self.num_timesteps)
                        + self.exploration.value(self.num_timesteps) / float(self.env.action_space.n)
                    )
                    kwargs["reset"] = reset
                    kwargs["update_param_noise_threshold"] = update_param_noise_threshold
                    kwargs["update_param_noise_scale"] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer, self.num_timesteps)

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get("is_success")
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0:

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert (
                            self.beta_schedule is not None
                        ), "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(
                            self.batch_size, beta=self.beta_schedule.value(self.num_timesteps), env=self._vec_normalize_env,
                        )
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes,) = experience
                    else:
                        (obses_t, actions, rewards, obses_tp1, dones,) = self.replay_buffer.sample(
                            self.batch_size, env=self._vec_normalize_env
                        )
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(
                                obses_t,
                                actions,
                                rewards,
                                obses_tp1,
                                obses_tp1,
                                dones,
                                weights,
                                sess=self.sess,
                                options=run_options,
                                run_metadata=run_metadata,
                            )
                            writer.add_run_metadata(run_metadata, "step%d" % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(
                                obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights, sess=self.sess,
                            )
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(
                            obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights, sess=self.sess,
                        )

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer), \
                            'replay_buffer should be an instance of PrioritizedReplayBuffer: {}'\
                                .format(type(self.replay_buffer))
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    callback.on_rollout_start()

                if (
                    can_sample
                    and self.num_timesteps > self.learning_starts
                    and self.num_timesteps % self.target_network_update_freq == 0
                ):
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular(
                        "% time spent exploring", int(100 * self.exploration.value(self.num_timesteps)),
                    )
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        if self.save_replay_buffer:
            data = {
                "double_q": self.double_q,
                "param_noise": self.param_noise,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "prioritized_replay": self.prioritized_replay,
                "prioritized_replay_eps": self.prioritized_replay_eps,
                "batch_size": self.batch_size,
                "target_network_update_freq": self.target_network_update_freq,
                "prioritized_replay_alpha": self.prioritized_replay_alpha,
                "prioritized_replay_beta0": self.prioritized_replay_beta0,
                "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
                "exploration_final_eps": self.exploration_final_eps,
                "exploration_fraction": self.exploration_fraction,
                "learning_rate": self.learning_rate,
                "replay_buffer": self.replay_buffer,
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
            }
        else:
            data = {
                "double_q": self.double_q,
                "param_noise": self.param_noise,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "prioritized_replay": self.prioritized_replay,
                "prioritized_replay_eps": self.prioritized_replay_eps,
                "batch_size": self.batch_size,
                "target_network_update_freq": self.target_network_update_freq,
                "prioritized_replay_alpha": self.prioritized_replay_alpha,
                "prioritized_replay_beta0": self.prioritized_replay_beta0,
                "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
                "exploration_final_eps": self.exploration_final_eps,
                "exploration_fraction": self.exploration_fraction,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
            }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

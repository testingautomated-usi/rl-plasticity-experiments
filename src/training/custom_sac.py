import numpy as np
import tensorflow as tf
from stable_baselines import SAC
from stable_baselines.common import SetVerbosity, tf_util
from stable_baselines.common.buffers import ReplayBuffer

from log import Log


class CustomSAC(SAC):
    """
    Custom version of Soft Actor-Critic (SAC).
    It is adapted from the stable-baselines version.

    Notable changes:
    - save replay buffer and restore it while loading

    """

    def __init__(self, total_timesteps=-1, save_replay_buffer: bool = True, **kwargs):
        super(CustomSAC, self).__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self.save_replay_buffer = save_replay_buffer
        self.logger = Log("CustomSAC")

    # changes: do not replace replay buffer when its size > 0
    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                if self.replay_buffer and len(self.replay_buffer) > 0:
                    # TODO: maybe substitute with a prioritized buffer to give preference to the transitions added
                    # during continual learning
                    pass
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, **self.policy_kwargs)
                    self.target_policy = self.policy(
                        self.sess, self.observation_space, self.action_space, **self.policy_kwargs
                    )

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name="terminals")
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name="rewards")
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name="actions",)
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    (self.deterministic_action, policy_out, logp_pi,) = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(
                        self.processed_obs_ph, self.actions_ph, create_qf=True, create_vf=True,
                    )
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(
                        self.processed_obs_ph, policy_out, create_qf=True, create_vf=False, reuse=True,
                    )

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == "auto":
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if "_" in self.ent_coef:
                            init_value = float(self.ent_coef.split("_")[1])
                            assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable(
                            "log_ent_coef", dtype=tf.float32, initializer=np.log(init_value).astype(np.float32),
                        )
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(
                        self.processed_next_obs_ph, create_qf=False, create_vf=True
                    )
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(self.rewards_ph + (1 - self.terminals_ph) * self.gamma * self.value_target)

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=tf_util.get_trainable_vars("model/pi"))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = tf_util.get_trainable_vars("model/values_fn")

                    source_params = tf_util.get_trainable_vars("model/values_fn/vf")
                    target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [tf.assign(target, source) for target, source in zip(target_params, source_params)]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = [
                            "policy_loss",
                            "qf1_loss",
                            "qf2_loss",
                            "value_loss",
                            "entropy",
                        ]
                        # All ops to call during one training step
                        self.step_ops = [
                            policy_loss,
                            qf1_loss,
                            qf2_loss,
                            value_loss,
                            qf1,
                            qf2,
                            value_fn,
                            logp_pi,
                            self.entropy,
                            policy_train_op,
                            train_values_op,
                        ]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ["ent_coef_loss", "ent_coef"]
                                self.step_ops += [
                                    ent_coef_op,
                                    ent_coef_loss,
                                    self.ent_coef,
                                ]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar("policy_loss", policy_loss)
                    tf.summary.scalar("qf1_loss", qf1_loss)
                    tf.summary.scalar("qf2_loss", qf2_loss)
                    tf.summary.scalar("value_loss", value_loss)
                    tf.summary.scalar("entropy", self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar("ent_coef_loss", ent_coef_loss)
                        tf.summary.scalar("ent_coef", self.ent_coef)

                    tf.summary.scalar("learning_rate", tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    # changes: saved replay buffer
    def save(self, save_path, cloudpickle=False):
        if self.save_replay_buffer:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else "auto",
                "replay_buffer": self.replay_buffer,
                "target_entropy": self.target_entropy,
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
            }
        else:
            data = {
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "train_freq": self.train_freq,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else "auto",
                "target_entropy": self.target_entropy,
                "gamma": self.gamma,
                "verbose": self.verbose,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "policy": self.policy,
                "n_envs": self.n_envs,
                "n_cpu_tf_sess": self.n_cpu_tf_sess,
                "seed": self.seed,
                "action_noise": self.action_noise,
                "random_exploration": self.random_exploration,
                "_vectorize_action": self._vectorize_action,
                "policy_kwargs": self.policy_kwargs,
            }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

"""
copied from openai gym and adapted
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnvWrapper(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 g: float = 10.0,
                 dt: float = 0.05,
                 mass: float = 1.0,
                 length: float = 1.0,
                 discrete_action_space: bool = False,
                 manual: bool = False
                 ):
        # cannot be changed because it is part of the observation space
        self.max_speed = 8
        # cannot be changed because it is part of the action space
        self.max_torque = 2.0

        self.dt = dt
        self.g = g
        self.m = mass
        self.l = length
        self.viewer = None

        self.discrete_action_space = discrete_action_space
        self.manual = manual

        high = np.array([1., 1., self.max_speed])
        if self.discrete_action_space or self.manual:
            if self.manual:
                self.action_space = spaces.Discrete(3)
            else:
                self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        if self.discrete_action_space:
            u = self.max_torque if u == 1 else -self.max_torque
        elif self.manual:
            if u != 0:
                u = self.max_torque if u == 1 else -self.max_torque
        else:
            u = np.clip(u, -self.max_torque, self.max_torque)[0]

        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __str__(self) -> str:
        return '(dt: {}, length: {}, mass: {}, discrete_action_space: {})'\
            .format(self.dt, self.l, self.m, self.discrete_action_space)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    import time
    from gym.wrappers import TimeLimit
    import gym

    # # original
    # env = PendulumEnv(manual=True)
    # original
    env = PendulumEnvWrapper(manual=True)
    env = TimeLimit(env, max_episode_steps=200)

    SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
    # can test what skip is still usable.

    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False


    def key_press(key, mod):
        global human_agent_action, human_wants_restart, human_sets_pause
        # enter
        if key == 0xff0d: human_wants_restart = True
        # backspace
        if key == 32: human_sets_pause = not human_sets_pause
        a = int(key - ord('0'))
        # left
        if a == 65313:
            human_agent_action = 1
        # right
        elif a == 65315:
            human_agent_action = 2
        else:
            return


    def key_release(key, mod):
        global human_agent_action
        a = int(key - ord('0'))
        if a == 65313 or a == 65315:
            # do nothing
            human_agent_action = 0
        else:
            return


    env.reset()
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release


    def rollout(env):
        global human_agent_action, human_wants_restart, human_sets_pause
        human_wants_restart = False
        obser = env.reset()
        skip = 0
        total_reward = 0
        total_timesteps = 0
        while 1:
            if not skip:
                # print("taking action {}".format(human_agent_action))
                a = human_agent_action
                total_timesteps += 1
                skip = SKIP_CONTROL
            else:
                skip -= 1

            obser, r, done, info = env.step(a)
            if done != 0:
                print("reward %0.3f" % r)
            total_reward += r
            window_still_open = env.render()
            if window_still_open == False: return False
            if done: break
            if human_wants_restart: break
            while human_sets_pause:
                env.render()
                time.sleep(0.1)
            time.sleep(0.1)
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


    while 1:
        window_still_open = rollout(env)
        if window_still_open == False: break

"""
copied from openai gym and adapted
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnvWrapper(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, masscart: float = 1.0, masspole: float = 0.1, length: float = 0.5,
                 discrete_action_space: bool = True, manual: bool = False, cart_friction: float = 0.0):
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.length = length  # actually half the pole's length
        self.cart_friction = cart_friction
        self.pole_angle = 12
        # Angle at which to fail the episode
        self.theta_threshold_radians = self.pole_angle * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.discrete_action_space = discrete_action_space

        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.manual = manual

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        if discrete_action_space:
            if manual:
                self.action_space = spaces.Discrete(3)
            else:
                self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = self.state
        if self.discrete_action_space:
            if self.manual:
                if action == 0:
                    force = 0.0
                elif action == 1:
                    force = self.force_mag
                elif action == 2:
                    force = -self.force_mag
            else:
                force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = self.force_mag * float(action[0])
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta - self.cart_friction * np.sign(
            x_dot)) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' "
                    "-- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def __str__(self) -> str:
        return '(masscart: {}, masspole: {}, length: {}, cart_friction: {}, discrete_action_space: {})'\
            .format(self.masscart, self.masspole, self.length, self.cart_friction, self.discrete_action_space)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":

    import time
    import gym

    # original
    env = CartPoleEnvWrapper(masscart=1.0, masspole=0.1, length=0.5, cart_friction=0.0, manual=True)

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
            # if r != 0:
            #     print("reward %0.3f" % r)
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

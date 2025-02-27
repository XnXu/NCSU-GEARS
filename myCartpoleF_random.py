"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import torch
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class CartPoleSwingUp(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    """
    Description:
        A pole is attached by an un-actuated joint to a cart

    Source:
        This environment corresponds to the version of the cart-pole problem described by Emi Kennedy and Tran


    Observation: 

        Type: Box(5)
        Num	Observation                 Min                         Max
        0	Cart Position             -0.25                         0.25
        1	Cart Velocity             -Inf                          Inf
        2	Cos(Pole Angle)           cos(-2pi)                     cos(2pi)
        3   Sin(Pole Angle)           sin(-2pi)                     sin(2pi)
        4	Pole Velocity At Tip      -Inf                          Inf

    Actions:
        Type: Box(1)
        Num	Observation                 Min         Max
        0	Voltage to motor            -10          10

        #so, input from -1 to 1, then multiplied by max_volt

        Note: The amount the velocity that is reduced or increased is not fixed;
        it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is LQR for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:

        Cart Position is more than 0.25 (center of the cart reaches the edge of the display)
        Episode length is greater than (4 min * 60 seconds per min / 0.01 s time interval) = 24000 steps
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 0 over 100 consecutive trials.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.fps = 100 # Nikki changed from 50 to 100

        self.gravity = 9.81
        self.masscart = 0.57+0.37
        self.masspole = 0.230
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.3302  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.r_mp = 6.35e-3 # motor pinion radius
        '''# mod Jm'''
        self.Jm = 3.90e-7 # rotor moment of inertia
        
        self.Kg = 3.71 # planetary gearbox gear ratio
        self.Rm = 2.6 # motor armature resistance
        self.Kt = 0.00767 # motor torque constant
        self.Km = 0.00767 # Back-ElectroMotive-Force (EMF) Constant V.s/RAD
        # both of these are in N.m.s/RAD, not degrees
        ''' # mod Beq'''
        self.Beq = 5.4 #  equivalent viscous damping coecient as seen at the motor pinion
        ''' # mod Bp'''
        self.Bp = 0.0024 # viscous damping doecient, as seen at the pendulum axis
        
        # param = Jm, Beq, Bp
        # self.Jm = param["Jm"]
        # self.Beq = param["Beq"]
        # self.Bp = param["Bp"]


        self.force_mag = 10.0 # should be 8 for our case?
        self.tau = 1/self.fps  # seconds between state updates
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.tau)),
        }

        # self.kinematics_integrator = "semi-implicit-euler" # newly added from native CartPole
        self.kinematics_integrator = "RK4"

        # copied the following from mountain_car

        """
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        """


        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12  * math.pi / 180 # according to Dylan's thesis
        self.theta_threshold_radians = 0.2 # approximatedly 12 deg
#        self.x_threshold = 0.25 # lab result triggers watchdog
        self.x_threshold = 0.25
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        # recall observation = sin theta, cos theta, theta_dot, x, x_dot !!!
        # recall state = x, x_dot, theta, theta_dot
        high = np.array(
                [
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max
                ],
                dtype = np.float32)
        # action is sampled uniformly from [-1,1]
        self.max_action = 1.0
        # self.action_space = spaces.Box(low = -self.max_action, high = self.max_action)
        # self.observation_space = spaces.Box(low = -high, high = high)

        self.action_space = spaces.Box(
                low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32
                ) # \in \mathbb{R}^1 bounded between -1 to 1
        self.observation_space = spaces.Box(-high, high, dtype=np.float32) # in \mathbb{R}^5

        #from new native CartPole
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        self.previous_force = 0
        # seems to be outdated
        """
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
        return [seed]
        """

        """
        ## In native CartPole beginning part - possibly already has integrated seeding
        >>> env.reset(seed=123, options={"low": 0, "high": 1})
        (array([0.6823519 , 0.05382102, 0.22035988, 0.18437181], dtype=float32), {})
        """

    def RHS(self, y, force):

        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2
        y = self.state

        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        return np.array( (x_dot, xacc, theta_dot, thetaacc), dtype=np.float32).flatten()


    def stepPhysics(self, force):
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"

        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2


        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        if self.kinematics_integrator == "semi-euler":
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.kinematics_integrator == "RK4":
            # RK4
            y = self.state
            k1 = self.RHS(y, force = force)
            k2 = self.RHS(y + self.tau * k1 / 2, force = force)
            k3 = self.RHS(y + self.tau * k2 / 2, force = force)
            k4 = self.RHS(y + self.tau * k3, force = force)
            y = y + self.tau/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, x_dot, theta, theta_dot = y

        # self.state = (x, x_dot, theta, theta_dot)
        return np.array( (x, x_dot,theta, theta_dot), dtype = np.float32).flatten()

    def stepSwingUp(self, force):
        # assert self.action_space.contains(
        #     0.1 * force
        # ), f"{force!r} ({type(force)}) invalid"
        # Currently same as stepPhysics
        # Need to modify??

        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # theta = math.atan(sintheta/costheta) # nikki_: this isn't corrent, atan maps to -pi/2 to pi/2 but our angle can be -pi to pi
        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2


        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

        if self.kinematics_integrator == "semi-euler":
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.kinematics_integrator == "RK4":
            # RK4
            y = self.state
            k1 = self.RHS(y, force = force)
            k2 = self.RHS(y + self.tau * k1 / 2, force = force)
            k3 = self.RHS(y + self.tau * k2 / 2, force = force)
            k4 = self.RHS(y + self.tau * k3, force = force)
            y = y + self.tau/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, x_dot, theta, theta_dot = y

        # self.state = (x, x_dot, theta, theta_dot)
        return np.array( (x, x_dot,theta, theta_dot), dtype = np.float32).flatten()

    def reward(self, terminated, off_track, action):

        x, x_dot, theta, theta_dot = self.state
        cos = np.cos(theta)
        sin = np.sin(theta)

        # theta = np.arctan2(sin, cos) nikki_: this coordinate is different
        #theta = np.arctan2(cos, -sin) - math.pi / 2
        theta = np.arctan2(sin, cos)

        #if abs(theta) < 0.1:
        #    print(theta)
        x_bound = 0.23
        B = 0
        if abs(x) > x_bound:
            B = 1
        reward = -0.1 * (5 * theta**2 + 0.5 * x**2 + 0.2 * x_dot**2 + 0.1 * (self.previous_force - self.force_mag * action ) **2 ) - 100 * B
        # Reward function

        # Apply off-track penalty and termination
        #if terminated or off_track:
        if off_track:
            reward -= 100  # Large penalty for failure


        return reward
        #(GEARS (?))

    def step(self, action):
        # !!this block may contain bugs!!
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        # force = float(np.clip( self.max_action * action, -self.max_action, self.max_action))

        # Nikki modified the following force from continuous_mountain_car
        # force = min(max(action[0], -self.max_action), self.max_action) * self.max_action
        self.state = self.stepSwingUp(force)
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(abs(theta) < self.theta_threshold_radians / 4)  # 1 if pole stands up - can turn off
        off_track = bool(abs(x) > self.x_threshold)

        reward = self.reward(terminated, off_track, 0) # TODO: penalize u_dot
        self.previous_force = force
        # Nikki copied the following reward from continuous_mountain_car
        # reward = 0.0
        # if terminated:
        #     reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1
        # Nikki changed from above to below according  to:
        # Matlab Train DDPG to swing up and balance pole
        #

        obs = np.array( (x, x_dot, np.cos(theta), np.sin(theta), theta_dot), dtype=np.float32).flatten()

        """
        #NX changed from above to below
        reward = int(not terminated)
        info = {"reward_survive": reward}
        """

        if self.render_mode == "human":
            self.render()
        terminated = False
        return obs, reward, terminated, off_track, {}
                 # quad_reward, terminated, False, {}
                 # nikki_: should the step also outputs self.state?



    ## This def is copied from native cartpole - !!may contain bugs!!
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            # options, -0.05, 0.05  # default low
            options, -0.08, 0.08 #NX changed from above
        )  # default high
        # self.state = np.array(
        #   self.np_random.uniform(low=low, high=high, size=(4,))
        #   - [0,0,math.pi,0] ).flatten()
        self.state = np.array(
                [ self.np_random.uniform(low = -0.05, high = 0.05),
                  0 ,
                  self.np_random.uniform(low = -0.001, high = 0.001) - math.pi,
                  0 ]
                ).flatten()
        self.steps_beyond_terminated = None

        x, x_dot, theta, theta_dot = self.state
        # changed obs to be consistent as before in step

        obs = np.array( (x, x_dot, np.cos(theta), np.sin(theta), theta_dot), dtype=np.float32).flatten()
        #obs = np.array((np.sin(theta), np.cos(theta), theta_dot, x, x_dot),
        #              dtype=np.float32).flatten()
        # to go from observation (obs) angles to state space angle, need the following transformation (rotate the coordinate by pi/2):
        # state_angle = math.atan2(obs_cosangle, -obs_sinangle) - pi/2

        if self.render_mode == "human":
            self.render()
        return np.array(obs, dtype=np.float32), {}




    ## this render def is copied from CartPole and never changed...

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False



# below is copied without modification from native cartpole

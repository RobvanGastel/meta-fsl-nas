import numpy as np
import math

import gym
from gym import spaces


class CartPolePOMDPWrapper(gym.ObservationWrapper):
    """
    Original observation space:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad
    3       Pole Angular Velocity     -Inf                    Inf

    Reduced to POMDP by removing the cart and angle velocity.
    Type: Box(2)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Pole Angle                -0.418 rad (-24 deg)    0.418 rad
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.env.spec.id == "CartPole-v1" or \
            env.env.spec.id == "CartPole-v2", \
            "Should only be used to wrap CartPole environments."

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def observation(self, observation):
        return observation[::2]

import numpy as np

from gym import spaces
import gym

import gym_minigrid
from gym_minigrid.wrappers import ViewSizeWrapper


class FourRoomsEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, seed):

        # ViewSizeWrapper(
        self._env = gym.make('MiniGrid-FourRooms-v0')
        self._env.seed(seed)

        self._observation_space = spaces.Box(0, 255,
                                             shape=(49, ), dtype=np.int8)
        self._action_space = self._env.action_space

        self.reward_range = self._env.reward_range
        self.max_ep_len = 100

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        return self._env.reset()['image'].mean(axis=2).astype(np.float32).flatten()

    def step(self, action):
        obs, rew, done, info_dict = self._env.step(action)
        obs = obs['image'].mean(axis=2).astype(np.float32).flatten()
        return obs, rew, done, info_dict

    def render(self):
        pass

    def close(self):
        self._env.close()

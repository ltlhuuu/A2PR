import numpy as np
import gym
import copy

from core.utils.torch_utils import random_seed


class MountainCar:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (2,)
        self.action_dim = 3
        self.env = gym.make('MountainCar-v0')
        self.env._seed = seed
        self.env._max_episode_steps = np.inf # control timeout setting in agent

    def generate_state(self, coords):
        return coords

    def reset(self):
        return np.asarray(self.env.reset())

    def step(self, a):
        state, reward, done, info = self.env.step(a[0])
        # self.env.render()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return np.array(self.env.state)

    def info(self, key):
        return


import gym
import copy

from stable_baselines.common.atari_wrappers import LazyFrames
import numpy as np
from collections import deque

from gym import spaces
from environment.server import Server

class NetwEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, k=4, num_of_data=8, num_of_target=3, num_of_user =24, action_method='ChooseTier', serverPort=5555):
        self.server = Server(serverPort, 5, num_of_data)
        self.k = k
        self.num_of_data = num_of_data
        self.current_target = np.zeros(self.num_of_data)
        self.action_method = action_method
        self.num_of_target = num_of_target
        self.num_of_user = num_of_user
        self.frames = np.zeros((k, self.num_of_data, 4), dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high=1, shape=(self.num_of_data* 3,), dtype=np.float32)
        #temp = np.ones((self.num_of_data,))*self.num_of_targeddt
        #self.action_space = spaces.MultiDiscrete(temp)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(k, self.num_of_data, 4), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = action.reshape(self.num_of_data, 3)
        argmax_action = np.argmax(action, axis=1)
        action = argmax_action -1
        action = np.clip(action + self.current_target, 0 , self.num_of_target-1)
        while True:
            obs, reward, done, error = self.server.communicate(action)
            if not error:
                break
        obs = obs.astype(np.float32)
        obs_x = np.zeros((self.num_of_data, 4), dtype=np.float32)
        for i in range(len(obs)):
            obs_x[i][0] = copy.deepcopy(obs[i][0] / self.num_of_user)
            obs_x[i][1] = copy.deepcopy(obs[i][1] / self.num_of_target)
            obs_x[i][2] = copy.deepcopy((obs[i][2] - obs[i][3])/1000000)
            obs_x[i][3] = copy.deepcopy((obs[i][5] - obs[i][4])/10000000)
        for i in range(self.num_of_data):
            self.current_target[i] = obs[i][1];
        info = {"None": 1}
        self.frames[-1] = obs_x
        self.frames = np.roll(self.frames, 1, axis=0)
        return self.frames, reward, done, info

    def reset(self):
        while True:
            obs, reward, done, error = self.server.communicate(None)
            if not error:
                break
        for i in range(self.num_of_data):
            self.current_target[i] = i%self.num_of_target
        obs = obs.astype(np.float32)
        obs_x = np.zeros((self.num_of_data, 4), dtype=np.float32)
        for i in range(len(obs)):
            obs_x[i][0] = obs[i][0] / self.num_of_user
            obs_x[i][1] = obs[i][1] / self.num_of_target
            obs_x[i][2] = (obs[i][2] - obs[i][3])/1000000
            obs_x[i][3] = (obs[i][5] - obs[i][4])/10000000
        for i in range(self.k):
            self.frames[i] = obs_x
        return self.frames

    def render(self, mode='human'):
        pass

import gym
from gym.spaces import Tuple
from pretrained.ddpg import DDPG
import torch
import numpy as np
import os

class FrozenTag(gym.Wrapper):
    """ Tag with pretrained prey agent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = 3
        self.pt_action_space = self.action_space[self.n_agents:]
        self.pt_observation_space = self.observation_space[self.n_agents:]
        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:self.n_agents]

    def step(self, action):
        # random_action = self.pt_action_space.sample()
        random_action = 0
        action = tuple(action) + (random_action, random_action,)
        obs, rew, done, info = super().step(action)
        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]
        done = done[:self.n_agents]
        return obs, rew, done, info

class RandomTag(gym.Wrapper):
    """ Tag with pretrained prey agent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:-1]

    def step(self, action):
        random_action = self.pt_action_space.sample()
        action = tuple(action) + (random_action,)
        obs, rew, done, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info


class PretrainedTag(gym.Wrapper):
    """ Tag with pretrained prey agent """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = 3
        self.pt_action_space = self.action_space[self.n_agents:]
        self.pt_observation_space = self.observation_space[self.n_agents:]
        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])

        self.prey1 = DDPG(14, 5, 50, 128, 0.01)
        self.prey2 = DDPG(14, 5, 50, 128, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), 'prey_params.pt')
        save_dict = torch.load(param_path)
        self.prey1.load_params(save_dict['agent_params'][-1])
        self.prey1.policy.eval()
        self.last_prey_obs1 = None

        self.prey2.load_params(save_dict['agent_params'][-1])
        self.prey2.policy.eval()
        self.last_prey_obs2 = None

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs1 = obs[-2]
        self.last_prey_obs2 = obs[-1]
        return obs[:-2]

    def step(self, action):
        # print(f"Check prey shape {self.last_prey_obs1.shape}")
        prey_action1 = self.prey1.step(self.last_prey_obs1[:14])
        prey_action2 = self.prey2.step(np.concatenate((self.last_prey_obs2[:10],self.last_prey_obs2[14:]),axis=0))
        action = tuple(action) + (prey_action1, prey_action2,)
        obs, rew, done, info = super().step(action)
        self.last_prey_obs1 = obs[-2]
        self.last_prey_obs2 = obs[-1]
        obs = obs[:-2]
        rew = rew[:-2]
        done = done[:-2]
        return obs, rew, done, info
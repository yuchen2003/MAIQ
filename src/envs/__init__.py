from functools import partial
import pretrained
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

from .mpe_env.make_env import make_env as mpe_make

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
env = mpe_make('simple_adversary')
# REGISTRY["simple_adv"] = partial(env_fn, env=env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        # print(f"!!!!!!!!!!{self._env.n_agents}")
        self.n_agents = self._env.n_agents
        self._obs = None

        # 仅仅是为了离散的状态下获得有效动作，对于连续状态没什么用
        # self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.shape)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions, latent_code=None):
        """ Returns reward, terminated, info """
        # 感觉这个是离散状态下的代码，
        # actions = [int(a) for a in actions]

        # print(f"Check latent code shape!!!{latent_code.shape}")
        self._obs, reward, done, info = self._env.step(actions)
        # for mainfogail
        # reward = [r[1] if np.where(latent_code[0] == 1)[0][0] == 0 else r[2] for r in reward]
        # for others
        # reward = [r[1] for r in reward]

        # print(f"Check reward shape!!!!!{len(reward)}")
        # assert False
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        # print(f'Obs::{len(self._obs)}, {self._obs[0].shape}')
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    # def get_state(self):
        # return np.concatenate(self._obs, axis=0).astype(np.float32)

    # def get_state_size(self):
    #     """ Returns the shape of the state"""
    #     return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        # invalid = [0] * (self.longest_action_space.n - len(valid))
        invalid = [0] * (self.longest_action_space.shape[0] - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        # print(f"SSSSSOOOOOOOOOObs shape,{len(self._obs)}")
        assert len(self._obs) == self.n_agents, "Obs shape must be same with agents num"
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)

from functools import partial
from absl import logging
from .multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

try:
    from smac.env import StarCraft2Env
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
except Exception as error:
    logging.error(f'Cannot import SMAC env, please check your installation. Error info:\n{error}')

try:
    from smac.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    REGISTRY["sc2v2"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
except Exception as error:
    logging.error(f'Cannot import SMACv2 env, please check your installation. Error info:\n{error}')

try:
    from envs.gfootball import GoogleFootballEnv
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
except Exception as error:
    logging.error(f'Cannot import Google football env, please check your installation. Error info:\n{error}')

try:
    from .stag_hunt import StagHunt
    REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
except Exception as error:
    logging.error(f'Cannot import Stag Hunt env, please check your installation. Error info:\n{error}')

try:
    from .foraging import ForagingEnv
    REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
except Exception as error:
    logging.error(f'Cannot import Foraging env, please check your installation. Error info:\n{error}')

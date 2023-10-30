from .maddpg import MADDPGCritic
from .sac import SACCritic, SACQnet, ISACVNet, ISACQNet, CSACVNet, CSACQNet
from .centralV import CentralVCritic
from .coma import COMACritic

REGISTRY = {}

REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["sac_critic"] = SACCritic
REGISTRY["sac_q_net"] = SACQnet
REGISTRY["isac_v_net"] = ISACVNet
REGISTRY["isac_q_net"] = ISACQNet
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic"] = COMACritic
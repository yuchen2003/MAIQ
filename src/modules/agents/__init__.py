REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_linda_agent import LindaAgent
from .sac_agent import SACAgent, ISACAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["linda"] = LindaAgent
REGISTRY["sac"] = SACAgent
REGISTRY["isac"] = ISACAgent

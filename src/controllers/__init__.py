REGISTRY = {}

from .basic_controller import BasicMAC
from .maddpg_controller import MADDPGMAC
from .isac_controller import ISACMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["isac_mac"] = ISACMAC
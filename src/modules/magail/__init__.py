REGISTRY = {}
from .discriminator import Discriminator
from .rnn_discriminator import RNNDiscriminator

REGISTRY["fc"] = Discriminator
REGISTRY["rnn"] = RNNDiscriminator

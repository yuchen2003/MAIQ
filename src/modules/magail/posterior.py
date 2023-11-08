#!/usr/bin/env python
# Created at 2020/2/15
from typing import Tuple

import torch as th
import torch.nn as nn

from utils.rl_utils import device, resolve_activate_function
disc_types = ['decentralized', 'centralized']


class Discriminator(nn.Module):
    def __init__(self, num_states, num_actions, n_agent, disc_type='decentralized',
                 num_hiddens: Tuple = (128, 128),
                 activation: str = "relu",
                 drop_rate=None, use_noise=False, noise_std=0.1):
        super(Discriminator, self).__init__()

        if disc_type not in disc_types:
            assert False

        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.disc_type = disc_type
        self.drop_rate = drop_rate
        self.use_noise = use_noise
        self.noise_std = noise_std

        self.output_shape = n_agent if disc_type == 'centralized' else 1

        # self.output_shape = 1
        # set up module units
        # self.all_states_shape = sum([state.shape[0] for state in self.num_states])
        # self.all_action_shape = sum([action.shape[0] for action in self.num_actions])

        # if disc_type == 'decentralized':
        #     input_shape = self.num_states + num_actions
        # elif disc_type == 'centralized':
        #     input_shape = self.all_states_shape + self.all_action_shape
        # else:
        #     assert False
        input_shape = self.num_states + num_actions
        # print("input_shape", input_shape)

        _module_units = [input_shape]
        _module_units.extend(num_hiddens)
        _module_units += self.output_shape,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(*module_unit))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))
        # self._module_list.add_module(f"Layer_{idx + 1}_Activation", nn.Tanh())

    def forward(self, states, actions):
        """
        give states, calculate the estimated values
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: values
        """
        x = th.cat([states, actions], dim=-1)
        if self.use_noise:  # trick: add gaussian noise to discriminator
            x += th.normal(0, self.noise_std, size=x.shape, device=device)
        for module in self._module_list:
            x = module(x)
        return x


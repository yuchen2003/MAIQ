import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ========== critic network for joint sac ==========

class CSACVNet(nn.Module):
    def __init__(self, scheme, args):
        super(CSACVNet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, inputs):
        # inputs.shape: [batch_size, seq_len, n_agents * n_actions]
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape * self.args.n_agents

class CSACQNet(nn.Module):
    def __init__(self, scheme, args):
        super(CSACQNet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_actions ** self.n_agents)
    
    def forward(self, inputs):
        inputs_shape = inputs.shape
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x).reshape(*inputs_shape[:-1], self.n_actions ** self.n_agents)
        return q
 
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_last_action:
            assert 0
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape * self.args.n_agents

# ========== critic network for isac ==========

class ISACVNet(nn.Module):
    def __init__(self, scheme, args):
        super(ISACVNet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_agents)

    def forward(self, inputs):
        # inputs.shape: [batch_size, seq_len, n_agents * n_actions]
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        # input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape * self.args.n_agents

class ISACQNet(nn.Module):
    """
    采用ISACQNet没有问题，因为需要计算每个智能体的Q，然后通过VDN求和
    """
    def __init__(self, scheme, args):
        super(ISACQNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        if args.task_type == "continuous":
            self.n_actions = 1
        else:
            self.n_actions = args.n_actions

        self.input_shape = self._get_input_shape(scheme)

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_agents * self.n_actions)
    
    def forward(self, inputs):
        inputs_shape = inputs.shape
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x).reshape(*inputs_shape[:-1], self.n_agents, self.n_actions)
        if self.args.use_tanh:
            q = F.tanh(q)
            if self.args.divide_1_gamma:
                q = 1.0 * q / (1 - self.args.gamma)
        return q
 
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        return input_shape * self.args.n_agents


# ========== below is old critic ==========

class SACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_agents)

    def forward(self, inputs):
        # inputs = th.cat((inputs), dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

    def _get_input_shape(self, scheme):
        # obs集合
        input_shape = scheme["obs"]["vshape"]
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape * self.n_agents

class SACQnet(nn.Module):
    def __init__(self, scheme, args):
        super(SACQnet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = self._get_input_shape(scheme)
        self.input_shape = self.obs_shape + self.n_actions * self.n_agents
        if self.args.obs_last_action:
            self.input_shape += self.n_actions

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_agents)

    def forward(self, s, a):
        # s = s.reshape(-1, self.obs_shape)
        # a = a.reshape(-1, self.n_actions * self.n_agents)
        x = th.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        # print(scheme["state"]["vshape"], scheme["obs"]["vshape"], self.n_agents, scheme["actions_one"])
        # whether to add the individual observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape * self.n_agents
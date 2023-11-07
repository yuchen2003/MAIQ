import torch
import torch.nn as nn
import torch.nn.functional as F


class ISACAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ISACAgent, self).__init__()
        self.args = args

        task_type = self.args.task_type
        assert task_type in ["continuous", "discrete"]
        self.task_continuous = True if task_type == "continuous" else False

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.action_head = nn.Linear(args.hidden_dim, args.n_actions * args.n_agents)
        self.continuous_std = nn.Linear(args.hidden_dim, args.n_actions * args.n_agents)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        # print(f"input shape {inputs.shape}")
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        x = F.relu(self.fc2(h))
        mu = self.action_head(x)
        log_std = self.continuous_std(x)
        log_std = torch.tanh(log_std)

        # TODO 先暂时写死
        log_std_min, log_std_max = (-5, 2)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()
        return mu, std, h 


class CSACAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CSACAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.action_head = nn.Linear(args.hidden_dim, args.n_actions ** args.n_agents)
        
    def init_hidden(self):
        # make hidden stAtes on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_() 

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        x = F.relu(self.fc2(h))
        outputs = self.action_head(x)
        return outputs, h


class SACAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SACAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        # sac增加
        self.mu_head = nn.Linear(args.hidden_dim, self.args.n_agents)
        self.log_std_head = nn.Linear(args.hidden_dim, self.args.n_agents)
        self.min_log_std = -20
        self.max_log_std = 2

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        x = self.fc2(h)

        mu = self.mu_head(x)
        log_sigma = F.relu(self.log_std_head(x))
        log_sigma = torch.clamp(log_sigma, self.min_log_std, self.max_log_std)

        return mu, log_sigma, h


class SARLRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SARLRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions* args.n_agents)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


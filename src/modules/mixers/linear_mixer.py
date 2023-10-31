import torch as th
import torch.nn as nn
import numpy as np

class LinearMixer(nn.Module):
    def __init__(self, args, abs='softmax'):
        super(LinearMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.w = nn.Linear(self.state_dim, self.n_agents)
        self.abs = abs
        if abs:
            self.b = nn.Linear(self.state_dim, 1)
    
    def getV(self, Q):
        return self.args.alpha * th.logsumexp(Q / self.args.alpha, dim=3)
    
    def forward(self, agent_qs, states, type='Q'):
        states = states.reshape(-1, self.state_dim)
        w = self.w(states).reshape(agent_qs.shape[0], -1, self.n_agents)
        if self.abs == "abs":
            w = th.abs(w)
            # w = th.sigmoid(w) - 0.5
            b = self.b(states).reshape(agent_qs.shape[0], -1, 1)
        elif self.abs == "relu":
            w = th.relu(w)
            b = self.b(states).reshape(agent_qs.shape[0], -1, 1)
        else:
            w = th.softmax(w, dim=-1)
        # (batch_size, seq_len, n_agents)
        if type == 'Q':
            x = agent_qs * w
            if self.abs != 'softmax':
                x = x + b
            x = x.sum(dim=2, keepdim=True)
        else:
            x = agent_qs * w.unsqueeze(-1)
            x = self.getV(x)
            x = x.sum(dim=2, keepdim=True)
            if self.abs != 'softmax':
                x = x + b
        return x
    


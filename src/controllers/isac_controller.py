from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd
import math

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# This multi-agent controller shares parameters between agents
class ISACMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return agent_outputs

    def forward(self, ep_batch, t, action_outputs=None, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, std, self.hidden_states  = self.agent(agent_inputs, self.hidden_states)


        if self.args.task_type == "continuous": 
            if test_mode:
                agent_mean = agent_outs
            # dist = th.distributions.Normal(agent_outs, std)
            # normal_sample = dist.rsample()

            # log_prob = dist.log_prob(normal_sample) 
            # log_prob_reshaped = log_prob.view(ep_batch.batch_size, self.args.n_agents, -1) 
            # normal_sample = normal_sample.view(ep_batch.batch_size, self.args.n_agents, -1) 

            # log_prob_reshaped = th.sum(log_prob_reshaped, axis=-1, keepdims=True)
            # log_prob_reshaped -= th.sum(
            #     2 * (np.log(2.0) - normal_sample - F.softplus(-2 * normal_sample)),
            #     axis=-1,
            #     keepdims=True,
            # )
            # action_outputs = th.tanh(normal_sample)
            # print("log shape", action_outputs.shape, log_prob_reshaped.shape)
            # log_prob = dist.log_prob(normal_sample)
            # action_outputs = th.tanh(normal_sample)
            # log_prob = log_prob - torch.log(1 - torch.tanh(outputs).pow(2) + 1e-7)
            # outputs = outputs * self.action_bound
            # masked_policies = action_outputs.clone()
            # action_outputs[avail_actions == 0.0] = 0.0
        
                
            if action_outputs == None:
                dist = th.distributions.Normal(agent_outs, std)
                normal_sample = dist.rsample()

                log_prob = dist.log_prob(normal_sample) 
                log_prob_reshaped = log_prob.view(ep_batch.batch_size, self.args.n_agents, -1) 
                normal_sample = normal_sample.view(ep_batch.batch_size, self.args.n_agents, -1) 

                log_prob_reshaped = th.sum(log_prob_reshaped, axis=-1, keepdims=True)
                log_prob_reshaped -= th.sum(
                    2 * (np.log(2.0) - normal_sample - F.softplus(-2 * normal_sample)),
                    axis=-1,
                    keepdims=True,
                )
                action_outputs = th.tanh(normal_sample)

            else:
                dist = SquashedNormal(agent_outs, std)
                action_outputs = action_outputs.view(-1, self.args.n_agents * self.args.n_actions)
                #print(action_outputs.max(), action_outputs.min())
                action_outputs = th.clamp(action_outputs, -1 + 1e-7, 1 - 1e-7)
                log_prob = dist.log_prob(action_outputs)
                log_prob_reshaped = log_prob.view(ep_batch.batch_size, self.args.n_agents, -1) 
                # std = (1 - th.tanh(agent_outs) ** 2) * std
                # agent_outs = th.tanh(agent_outs)
                # action_outputs = action_outputs.view(-1, self.args.n_agents * self.args.n_actions)
                # dist = th.distributions.Normal(0, 1)
                # log_prob = dist.log_prob(action_outputs) - th.log(std + 1e-8)
                # log_prob_reshaped = log_prob.view(ep_batch.batch_size, self.args.n_agents, -1) 
            
            log_prob_reshaped = th.sum(log_prob_reshaped, axis=-1, keepdims=True)

            if test_mode:
                agent_mean = th.tanh(agent_mean)
                return agent_mean.view(ep_batch.batch_size, self.args.n_agents, -1), log_prob_reshaped # test, cont
            return action_outputs.view(ep_batch.batch_size, self.args.n_agents, -1), log_prob_reshaped # train, cont

        return agent_outs.view(ep_batch.batch_size, self.args.n_agents, -1), None                      # discrete

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])
        # Don't need agent onehot_id
        # if self.args.obs_agent_id:
        #     inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        # input_shape += scheme["actions"]["vshape"][0]
        return input_shape * self.args.n_agents

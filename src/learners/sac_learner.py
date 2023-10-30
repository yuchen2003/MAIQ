import copy
from hashlib import sha1
from re import S
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from modules.agents import REGISTRY as q_net_registry
from components.standarize_stream import RunningMeanStd
import torch
from torch.distributions import Normal

class SACLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        
        ############################ SAC ########################################################

        self.policy_net = mac
        self.q_net_1 = critic_registry[args.q_net](scheme, args)
        self.q_net_2 = critic_registry[args.q_net](scheme, args)
        self.target_q_net_1 = critic_registry[args.q_net](scheme, args)
        self.target_q_net_2 = critic_registry[args.q_net](scheme, args)
        # self.v_net = critic_registry[args.v_net](scheme, args)
        # self.target_v_net = critic_registry[args.v_net](scheme, args)
        
        # self.value_net = critic_registry[args.critic_type](scheme, args)
        # self.Target_value_net = critic_registry[args.critic_type](scheme, args)
        # self.Q_net = critic_registry[args.q_net](scheme, args)
        
        self.q_net_parameters = list(self.q_net_1.parameters()) + list(self.q_net_2.parameters())
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.args.lr)
        # self.v_optimizer = Adam(self.v_net.parameters(), lr=self.args.lr)
        self.q_optimizer = Adam(self.q_net_parameters, lr=self.args.lr)
        
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_(param.data)

        ################################### maddpg ################################################
        # self.mac = mac
        # self.target_mac = copy.deepcopy(self.mac)
        # self.agent_params = list(mac.parameters())

        # self.critic = critic_registry[args.critic_type](scheme, args)
        # self.target_critic = copy.deepcopy(self.critic)
        # self.critic_params = list(self.critic.parameters())

        # self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
        # self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)

        ####################################################################################

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

        self.device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        # print(f"actions is {actions}")
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # compute policy distribution
        policy_out = []
        log_prob_out_ = []
        self.policy_net.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, log_prob = self.policy_net.forward(batch, t=t)
            policy_out.append(agent_outs)
            log_prob_out_.append(log_prob)
        policy_out = th.stack(policy_out, dim=1)[:, :] # [batch_size, seq_len, n_agents, n_actions]
        # q_policy_out = th.stack(policy_out, dim=1)[:, 1:] # [batch_size, seq_len, n_agents, n_actions]
        log_prob_out = th.stack(log_prob_out_, dim=1)[:, :-1] # [batch_size, seq_len, n_agents, n_actions]
        q_log_prob_out = th.stack(log_prob_out_, dim=1)[:, 1:] # [batch_size, seq_len, n_agents, n_actions]

        # compute obs input
        q_inputs_flatten, next_q_inputs, current_q_inputs = self._build_inputs(batch, policy_out)
        # current_obs_flatten = inputs_flatten[:, :-1]
        # next_obs_flatten = inputs_flatten[:, 1:]
        # update q
        q_loss, q_grad_norm = self._update_q(q_inputs_flatten, next_q_inputs, q_log_prob_out, rewards, terminated, actions, mask)
        # update policy
        policy_loss, policy_grad_norm = self._update_policy(batch, current_q_inputs, log_prob_out, mask)

        # soft update
        self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat
            # self.logger.log_stat("v_loss", v_loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            # self.logger.log_stat("v_grad_norm", v_grad_norm.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm.item(), t_env)
            self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)  
            self.log_stats_t = t_env
    
    def _update_q(self, inputs_flatten, next_q_inputs, q_log_prob_out, rewards, terminated, actions, mask):
        ## compute the q loss
        current_obs_flatten = inputs_flatten[:, :-1]
        # next_obs_flatten = inputs_flatten[:, 1:]
        # target_value = self.target_v_net(next_obs_flatten)  # [bs, seq_len, n_agents]
        # target_q_value = rewards + (1 - terminated[:, 1:]) * self.args.gamma * target_value.detach()
        q_values_1 = self.q_net_1(current_obs_flatten)   # [bs, seq_len, n_agents, n_actions]
        q_values_2 = self.q_net_2(current_obs_flatten)   # [bs, seq_len, n_agents, n_actions]
        
        # next_q_inputs_detach = next_q_inputs.detach()
        q_log_prob_out_detach = q_log_prob_out.detach()
        entropies = th.sum(
            q_log_prob_out_detach, dim=-2, keepdim=False
        )
        target_value = th.minimum(self.target_q_net_1(next_q_inputs), self.target_q_net_2(next_q_inputs)).sum(dim=-2)  # [bs, seq_len, n_agents, n_actions]
        target_q_value = rewards + (1 - terminated) * self.args.gamma * (target_value.detach() - self.args.sac_alpha*entropies)
        q_value_1 = th.sum(q_values_1, dim=-2)   # remove last dim
        q_value_2 = th.sum(q_values_2, dim=-2)

        q_error_1 = (q_value_1 - target_q_value.detach())
        q_error_2 = (q_value_2 - target_q_value.detach())
        q_mask = mask.expand_as(q_error_1)
        q_loss = ((q_error_1 * q_mask) ** 2).sum() / q_mask.sum() + ((q_error_2 * q_mask) ** 2).sum() / q_mask.sum()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_net_parameters, self.args.grad_norm_clip)
        self.q_optimizer.step()
        # for name, parms in self.q_net_1.named_parameters():	
            # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            # ' -->grad_value:',parms.grad)
        return q_loss, q_grad_norm


    def _update_policy(self, batch, current_obs_flatten, log_prob_out, mask):
        ## compute the policy loss
        # current_obs_flatten = inputs_flatten[:, :-1]
        q_values_1 = self.q_net_1(current_obs_flatten)   # [bs, seq_len, n_agents, n_actions]
        q_values_2 = self.q_net_2(current_obs_flatten)   # [bs, seq_len, n_agents, n_actions]
        
        q_values = th.min(q_values_1, q_values_2)

        entropies = th.sum(
            log_prob_out, dim=-2, keepdim=False
        )
        q = th.sum(
            q_values, dim=-2, keepdim=False,
        )
        policy_loss = - q + self.args.sac_alpha * entropies
        policy_mask = mask.expand_as(policy_loss)
        policy_loss = (policy_loss * policy_mask).sum() / policy_mask.sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
        self.policy_optimizer.step()

        return policy_loss, policy_grad_norm

    def _build_inputs(self, batch, policy_out=None, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        q_inputs , next_q_inputs, current_q_inputs = [], [], []
        q_inputs.append(batch["obs"][:, ts])
        next_q_inputs.append(batch["obs"][:, 1:, ts])
        current_q_inputs.append(batch["obs"][:, :-1, ts])

        q_inputs.append(batch["actions"][:, ts])
        next_q_inputs.append(policy_out[:, 1:])
        current_q_inputs.append(policy_out[:, :-1])
        # Don't need agent onehot_id
        # if self.args.obs_agent_id:
        #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        q_inputs = th.cat([x.reshape(bs, max_t, -1) for x in q_inputs], dim=-1)
        next_q_inputs = th.cat([x.reshape(bs, max_t-1, -1) for x in next_q_inputs], dim=-1)
        current_q_inputs = th.cat([x.reshape(bs, max_t-1, -1) for x in current_q_inputs], dim=-1)
        # next_q_inputs = next_q_inputs.to(self.args.device)
        # current_q_inputs = current_q_inputs.to(self.args.device)
        return q_inputs, next_q_inputs, current_q_inputs

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.policy_net.cuda()
        self.q_net_1.cuda()
        self.q_net_2.cuda()
        self.target_q_net_1.cuda()
        self.target_q_net_2.cuda()
        # self.v_net.cuda()
        # self.target_v_net.cuda()

    def save_models(self, path):
        self.policy_net.save_models(path)
        # th.save(self.v_net.state_dict(), "{}/v_net.th".format(path))
        th.save(self.q_net_1.state_dict(), "{}/q_net_1.th".format(path))
        th.save(self.q_net_2.state_dict(), "{}/q_net_2.th".format(path))
        th.save(self.policy_optimizer.state_dict(), "{}/policy_opt.th".format(path))
        # th.save(self.v_optimizer.state_dict(), "{}/v_opt.th".format(path))
        th.save(self.q_optimizer.state_dict(), "{}/q_opt.th".format(path))

    def load_models(self, path):
        self.policy_net.load_models(path)
        # self.v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        # self.target_v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
        self.q_net_1.load_state_dict(th.load("{}/q_net_1.th".format(path), map_location=lambda storage, loc: storage))
        self.q_net_2.load_state_dict(th.load("{}/q_net_2.th".format(path), map_location=lambda storage, loc: storage))
        self.policy_optimizer.load_state_dict(th.load("{}/policy_opt.th".format(path), map_location=lambda storage, loc: storage))
        # self.v_optimizer.load_models(th.load("{}/v_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimizer.load_state_dict(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))

# import copy
# from hashlib import sha1
# from re import S
# from components.episode_buffer import EpisodeBatch
# from modules.critics.maddpg import MADDPGCritic
# import torch as th
# from torch.optim import RMSprop, Adam
# from controllers.maddpg_controller import gumbel_softmax
# from modules.critics import REGISTRY as critic_registry
# from modules.agents import REGISTRY as q_net_registry
# from components.standarize_stream import RunningMeanStd
# import torch
# from torch.distributions import Normal
# from functools import partial

# class MAIQContinuousLeartest_rewardnerOnline:
#     """在连续版本的SAC上面修改"""
#     def __init__(self, mac, scheme, logger, args):
#         self.args = args
#         self.n_agents = args.n_agents
#         self.n_actions = args.n_actions
#         self.logger = logger
        
#         ############################ SAC ########################################################

#         self.policy_net = mac
#         self.q_net_1 = critic_registry[args.q_net](scheme, args)
#         # self.q_net_2 = critic_registry[args.q_net](scheme, args)
#         self.target_q_net_1 = critic_registry[args.q_net](scheme, args)
#         # self.target_q_net_2 = critic_registry[args.q_net](scheme, args)
#         # self.v_net = critic_registry[args.v_net](scheme, args)
#         # self.target_v_net = critic_registry[args.v_net](scheme, args)
        
#         # self.value_net = critic_registry[args.critic_type](scheme, args)
#         # self.Target_value_net = critic_registry[args.critic_type](scheme, args)
#         # self.Q_net = critic_registry[args.q_net](scheme, args)
        
#         self.q_net_parameters = list(self.q_net_1.parameters())
#         # self.q_net_parameters = list(self.q_net_1.parameters()) + list(self.q_net_2.parameters())

#         self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.args.lr)
#         # self.v_optimizer = Adam(self.v_net.parameters(), lr=self.args.lr)
#         self.q_optimizer = Adam(self.q_net_parameters, lr=self.args.lr)
        
#         for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
#             target_param.data.copy_(param.data)
#         # for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
#             # target_param.data.copy_(param.data)

#         self.log_stats_t = -self.args.learner_log_interval - 1

#         self.last_target_update_episode = 0

#         self.device = "cuda" if args.use_cuda else "cpu"
#         if self.args.standardise_returns:
#             self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
#         if self.args.standardise_rewards:
#             self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)


#     def clone(self, batch: EpisodeBatch, t_env: int):
#         actions = batch["actions"]
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#         actions = actions[:, :-1]

#         # No experiences to train on in this minibatch
#         if mask.sum() == 0:
#             self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
#             self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
#             return

#         # mask = mask.repeat(1, 1, self.n_agents)

#         for _ in range(self.args.epochs):
#             policy_out = []
#             self.policy_net.init_hidden(batch.batch_size)
#             for t in range(batch.max_seq_length-1):
#                 agent_outs, log_prob = self.policy_net.forward(batch, t=t)
#                 policy_out.append(agent_outs)
#             policy_out = th.stack(policy_out, dim=1) # [batch_size, seq_len, n_agents, n_actions]

#             bc_loss = th.nn.MSELoss()(policy_out, copy.deepcopy(actions).float())
#             # Optimise agents
#             self.policy_optimizer.zero_grad()
#             bc_loss.backward()
#             policy_grad_norm = th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
#             self.policy_optimizer.step()

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("behavior clone bc_loss", bc_loss.item(), t_env)
#             self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)  
#             self.log_stats_t = t_env

#     def build_batch_online(self, expert_batch: EpisodeBatch, agent_batch: EpisodeBatch, scheme, groups, preprocess):
#         batch = partial(EpisodeBatch, scheme, groups, expert_batch.batch_size*2, expert_batch.max_seq_length,
#                                  preprocess=preprocess, device=self.args.device)()
        
#         obs_expert = expert_batch['obs']
#         obs_agent = agent_batch['obs']
#         obs = th.cat((obs_expert, obs_agent), dim=0)

#         actions_expert = expert_batch['actions']
#         actions_agent = agent_batch['actions']
#         actions = th.cat((actions_expert, actions_agent), dim=0)

#         terminated_expert = expert_batch['terminated']
#         terminated_agent = agent_batch['terminated']
#         terminated = th.cat((terminated_expert, terminated_agent), dim=0)

#         filled_expert = expert_batch['filled']
#         filled_agent = agent_batch['filled']
#         filled = th.cat((filled_expert, filled_agent), dim=0)

#         batch.update({"obs":obs})
#         batch.update({"actions":actions})
#         batch.update({"terminated":terminated})
#         batch.update({"filled":filled})
#         self.is_expert = torch.cat([torch.ones_like(th.tensor(range(expert_batch.batch_size)), dtype=torch.bool),
#                            torch.zeros_like(th.tensor(range(expert_batch.batch_size)), dtype=torch.bool)], dim=0)
#         return batch


#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        
#         # Get the relevant quantities
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
#         policy_out, log_prob_out = self.actor_sample(batch)
#         # compute obs input
#         q_inputs_flatten, next_q_inputs, current_q_inputs = self._build_inputs(batch, policy_out)
#         # current_obs_flatten = inputs_flatten[:, :-1]
#         # next_obs_flatten = inputs_flatten[:, 1:]
#         # update q
#         q_loss, q_grad_norm = self._update_q(batch, q_inputs_flatten, terminated, mask)
#         # update policy
#         policy_loss, policy_grad_norm = self._update_policy(current_q_inputs, log_prob_out, mask)

#         # soft update
#         self._update_targets_soft(self.args.target_update_interval_or_tau)

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat
#             # self.logger.log_stat("v_loss", v_loss.item(), t_env)
#             self.logger.log_stat("q_loss", q_loss.item(), t_env)
#             self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
#             # self.logger.log_stat("v_grad_norm", v_grad_norm.item(), t_env)
#             self.logger.log_stat("q_grad_norm", q_grad_norm.item(), t_env)
#             self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)  
#             self.log_stats_t = t_env
    
#     def actor_sample(self, batch):
#         # compute policy distribution
#         policy_out = []
#         log_prob_out_ = []
#         self.policy_net.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             agent_outs, log_prob = self.policy_net.forward(batch, t=t)
#             policy_out.append(agent_outs)
#             log_prob_out_.append(log_prob)
#         policy_out = th.stack(policy_out, dim=1) # [batch_size, seq_len, n_agents, n_actions]
#         log_prob_out = th.stack(log_prob_out_, dim=1) # [batch_size, seq_len, n_agents, n_actions]

#         return policy_out, log_prob_out

#     def getV(self, batch, current=True):
#         """
#         这里的action是重新根据采样得到的，并非用的expert的action
#         """
#         obs_action = []

#         policy_out, log_prob_out = self.actor_sample(batch)
#         # 判断采用当前轨迹 or 下一时刻轨迹
#         if current:
#             log_prob_out = log_prob_out[:, :-1]
#             policy_out = policy_out[:, :-1]
#             obs_action.append(batch["obs"][:, :-1])
#         else:
#             log_prob_out = log_prob_out[:, 1:]
#             policy_out = policy_out[:, 1:]
#             obs_action.append(batch["obs"][:, 1:])

#         current_entropies = th.sum(
#             log_prob_out, dim=-2, keepdim=False
#         )

#         max_t = batch.max_seq_length
#         obs_action.append(policy_out)
#         q_inputs = th.cat([x.reshape(batch.batch_size, max_t-1, -1) for x in obs_action], dim=-1)
#         current_Q_1 = self.q_net_1(q_inputs).sum(dim=-2)
#         # current_Q_2 = self.q_net_2(q_inputs).sum(dim=-2)
#         # current_Q = th.min(current_Q_1, current_Q_2)
#         current_Q = current_Q_1
#         current_V = current_Q - self.args.sac_alpha*current_entropies

#         return current_V 

#     def get_tragetV(self, batch, current=True):
#         obs_action = []

#         policy_out, log_prob_out = self.actor_sample(batch)
#         # 判断采用当前轨迹 or 下一时刻轨迹
#         if current:
#             log_prob_out = log_prob_out[:, :-1]
#             policy_out = policy_out[:, :-1]
#             obs_action.append(batch["obs"][:, :-1])
#         else:
#             log_prob_out = log_prob_out[:, 1:]
#             policy_out = policy_out[:, 1:]
#             obs_action.append(batch["obs"][:, 1:])

#         target_entropies = th.sum(
#             log_prob_out, dim=-2, keepdim=False
#         )

#         max_t = batch.max_seq_length
#         obs_action.append(policy_out)
#         q_inputs = th.cat([x.reshape(batch.batch_size, max_t-1, -1) for x in obs_action], dim=-1)
#         target_Q_1 = self.target_q_net_1(q_inputs).sum(dim=-2)
#         # target_Q_2 = self.target_q_net_2(q_inputs).sum(dim=-2)
#         # target_Q = th.min(target_Q_1, target_Q_2)
#         target_Q = target_Q_1
#         target_V = target_Q - self.args.sac_alpha*target_entropies

#         return target_V 
#         # log_prob_out_ = []
#         # self.policy_net.init_hidden(batch.batch_size)
#         # for t in range(batch.max_seq_length):
#         #     agent_outs, log_prob = self.policy_net.forward(batch, t=t)
#         #     log_prob_out_.append(log_prob)
#         # log_prob_out = th.stack(log_prob_out_, dim=1) # [batch_size, seq_len, n_agents, n_actions]
#         # if current:
#         #     log_prob_out = log_prob_out[:, :-1]
#         # else:
#         #     log_prob_out = log_prob_out[:, 1:]
#         # target_entropies = th.sum(
#         #     log_prob_out, dim=-2, keepdim=False
#         # )
#         # target_Q_1 = self.target_q_net_1(inputs_flatten).sum(dim=-2)
#         # target_Q_2 = self.target_q_net_2(inputs_flatten).sum(dim=-2)
#         # target_Q = th.minimum(target_Q_1, target_Q_2)
#         # current_V = target_Q - self.args.sac_alpha*target_entropies

#         # return current_V 

#     def q_loss(self, q_value, batch, inputs_flatten,terminated, mask ):
#         current_inputs_flatten = inputs_flatten[:, :-1]
#         next_q_inputs = inputs_flatten[:, 1:]
#         # 4-deminson [batch_size, episode_len, agent_id, action_dim] 
#         # For Q network
#         current_V = self.getV(batch, True)
#         with th.no_grad():
#             next_V = self.get_tragetV(batch, False)

#         # y = self.args.gamma * current_V
#         y_next = (1 - terminated) * self.args.gamma * next_V

#         q_error = -(q_value - y_next)[self.is_expert]
#         q_mask = mask[self.is_expert].expand_as(q_error)
#         # loss1 = -Q(s,a)+V^\pi(s)
#         loss_1 = (q_mask * q_error).sum() / q_mask.sum()

#         # value_loss = current_V - y_next
#         q_mask = mask.expand_as(q_value)
#         value_loss = (1-self.args.gamma)*current_V[:, 0]
#         loss_2 = (q_mask[:, 0] * value_loss).sum() / q_mask[:, 0].sum()

#         # loss2 = 1 / 4a * (Q(s, a) - v(s'))^2
#         next_q_error = (q_value - y_next)
#         loss_3 = 1.0 / (4 * self.args.alpha) * ((next_q_error * q_mask) ** 2).sum() / q_mask.sum()

#         loss = loss_1 + loss_2 + loss_3
#         return loss


#     def _update_q(self, batch, inputs_flatten, terminated, mask):
#         # loss = min -E(Q-V(s))+1/4/alpha E[(Q-\gamma V(s'))^2]
#         ## compute the q loss
#         current_inputs_flatten = inputs_flatten[:, :-1]
#         next_q_inputs = inputs_flatten[:, 1:]
#         q_values_1 = self.q_net_1(current_inputs_flatten)   # [bs, seq_len, n_agents, n_actions]
#         # q_values_2 = self.q_net_2(current_inputs_flatten)   # [bs, seq_len, n_agents, n_actions]
        
#         q_value_1 = th.sum(q_values_1, dim=-2)   # remove last dim
#         # q_value_2 = th.sum(q_values_2, dim=-2)   # remove last dim
#         # q_value = th.min(q_value_1, q_value_2)
#         q_value = q_value_1

#         loss = self.q_loss(q_value, batch, inputs_flatten, terminated, mask)
#         # q_loss_1 = self.q_loss(q_value_1, batch, inputs_flatten, terminated, mask)
#         # q_loss_2 = self.q_loss(q_value_2, batch, inputs_flatten, terminated, mask)

#         # loss = (q_loss_1 + q_loss_2)/2

#         self.q_optimizer.zero_grad()
#         loss.backward()
#         q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_net_parameters, self.args.grad_norm_clip)
#         self.q_optimizer.step()
#         # for name, parms in self.q_net_1.named_parameters():	
#             # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
#             # ' -->grad_value:',parms.grad)
#         return loss, q_grad_norm


#     def _update_policy(self, current_input_flatten, log_prob_out_, mask):
#         ## compute the policy loss
#         q_values_1 = self.q_net_1(current_input_flatten).sum(dim=-2)   # [bs, seq_len, n_agents, n_actions]
#         # q_values_2 = self.q_net_2(current_input_flatten).sum(dim=-2)   # [bs, seq_len, n_agents, n_actions]
        
#         # q_values = th.min(q_values_1, q_values_2)
#         q_values = q_values_1
#         log_prob_out = log_prob_out_[:, :-1] # [batch_size, seq_len, n_agents, n_actions]

#         entropies = th.sum(
#             log_prob_out, dim=-2, keepdim=False
#         )
#         # q = th.sum(
#         #     q_values, dim=-2, keepdim=False,
#         # )
#         q = q_values
#         policy_loss = - q + self.args.sac_alpha * entropies
#         policy_mask = mask.expand_as(policy_loss)
#         policy_loss = (policy_loss * policy_mask).sum() / policy_mask.sum()

#         self.policy_optimizer.zero_grad()
#         policy_loss.backward()
#         policy_grad_norm = th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
#         self.policy_optimizer.step()

#         return policy_loss, policy_grad_norm

#     def _build_inputs(self, batch, policy_out=None, t=None):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length if t is None else 1
#         ts = slice(None) if t is None else slice(t, t + 1)

#         q_inputs , next_q_inputs, current_q_inputs = [], [], []
#         q_inputs.append(batch["obs"][:, ts])
#         next_q_inputs.append(batch["obs"][:, 1:, ts])
#         current_q_inputs.append(batch["obs"][:, :-1, ts])

#         q_inputs.append(batch["actions"][:, ts])
#         next_q_inputs.append(policy_out[:, 1:])
#         current_q_inputs.append(policy_out[:, :-1])
#         # Don't need agent onehot_id
#         # if self.args.obs_agent_id:
#         #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

#         q_inputs = th.cat([x.reshape(bs, max_t, -1) for x in q_inputs], dim=-1)
#         next_q_inputs = th.cat([x.reshape(bs, max_t-1, -1) for x in next_q_inputs], dim=-1)
#         current_q_inputs = th.cat([x.reshape(bs, max_t-1, -1) for x in current_q_inputs], dim=-1)
#         # next_q_inputs = next_q_inputs.to(self.args.device)
#         # current_q_inputs = current_q_inputs.to(self.args.device)
#         return q_inputs, next_q_inputs, current_q_inputs

#     def _update_targets_hard(self):
#         self.target_mac.load_state(self.mac)
#         self.target_critic.load_state_dict(self.critic.state_dict())

#     def _update_targets_soft(self, tau):
#         for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
#         # for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
#             # target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#     def cuda(self):
#         self.policy_net.cuda()
#         self.q_net_1.cuda()
#         # self.q_net_2.cuda()
#         self.target_q_net_1.cuda()
#         # self.target_q_net_2.cuda()
#         # self.v_net.cuda()
#         # self.target_v_net.cuda()

#     def save_models(self, path):
#         self.policy_net.save_models(path)
#         # th.save(self.v_net.state_dict(), "{}/v_net.th".format(path))
#         th.save(self.q_net_1.state_dict(), "{}/q_net_1.th".format(path))
#         # th.save(self.q_net_2.state_dict(), "{}/q_net_2.th".format(path))
#         th.save(self.policy_optimizer.state_dict(), "{}/policy_opt.th".format(path))
#         # th.save(self.v_optimizer.state_dict(), "{}/v_opt.th".format(path))
#         th.save(self.q_optimizer.state_dict(), "{}/q_opt.th".format(path))

#     def load_models(self, path):
#         self.policy_net.load_models(path)
#         # self.v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
#         # Not quite right but I don't want to save target networks
#         # self.target_v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
#         self.q_net_1.load_models(th.load("{}/q_net_1.th".format(path), map_location=lambda storage, loc: storage))
#         # self.q_net_2.load_models(th.load("{}/q_net_2.th".format(path), map_location=lambda storage, loc: storage))
#         self.policy_optimizer.load_models(th.load("{}/policy_opt.th".format(path), map_location=lambda storage, loc: storage))
#         # self.v_optimizer.load_models(th.load("{}/v_opt.th".format(path), map_location=lambda storage, loc: storage))
#         self.q_optimizer.load_models(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))

import copy
from functools import partial
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

class MAIQContinuousLearnerOnline:
    """在连续版本的SAC上面修改"""
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        
        ############################ SAC ########################################################

        self.policy_net = mac
        self.q_net_1 = critic_registry[args.q_net](scheme, args)
        #self.q_net_2 = critic_registry[args.q_net](scheme, args)
        self.target_q_net_1 = critic_registry[args.q_net](scheme, args)
        #self.target_q_net_2 = critic_registry[args.q_net](scheme, args)
        # self.v_net = critic_registry[args.v_net](scheme, args)
        # self.target_v_net = critic_registry[args.v_net](scheme, args)
        
        # self.value_net = critic_registry[args.critic_type](scheme, args)
        # self.Target_value_net = critic_registry[args.critic_type](scheme, args)
        # self.Q_net = critic_registry[args.q_net](scheme, args)
        
        self.q_net_parameters = list(self.q_net_1.parameters())
        # self.q_net_parameters = list(self.q_net_1.parameters()) + list(self.q_net_2.parameters())

        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.args.lr)
        # self.v_optimizer = Adam(self.v_net.parameters(), lr=self.args.lr)
        self.q_optimizer = Adam(self.q_net_parameters, lr=self.args.lr)
        
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(param.data)
        # for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            # target_param.data.copy_(param.data)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

        self.device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask_q = batch["filled"][:, :-1].float()
        mask_pi = batch["filled"][:, :-1].float()
        mask_pi[:, 1:] = mask_q[:, 1:] * (1 - terminated[:, :-1])

        
        q_loss, loss1, loss2, loss3, q_grad_norm = self._update_q(batch, terminated, mask_q)
        policy_loss, bc_loss, policy_grad_norm, q_value = self._update_policy(batch, mask_pi)

        # soft update
        self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat
            # self.logger.log_stat("v_loss", v_loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("loss_1", loss1.item(), t_env)
            self.logger.log_stat("loss_2", loss2.item(), t_env)
            self.logger.log_stat("loss_3", loss3.item(), t_env)
            self.logger.log_stat("bc_loss", bc_loss.item(), t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            # self.logger.log_stat("v_grad_norm", v_grad_norm.item(), t_env)
            self.logger.log_stat("q_value", q_value.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm.item(), t_env)
            self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)  
            self.log_stats_t = t_env
    
    def actor_sample(self, batch):
        # compute policy distribution
        policy_out = []
        log_prob_out_ = []
        self.policy_net.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, log_prob = self.policy_net.forward(batch, t=t)
            policy_out.append(agent_outs)
            log_prob_out_.append(log_prob)
        policy_out = th.stack(policy_out, dim=1) # [batch_size, seq_len, n_agents, n_actions]
        log_prob_out = th.stack(log_prob_out_, dim=1) # [batch_size, seq_len, n_agents, n_actions]

        return policy_out, log_prob_out



    def _update_q(self, batch, terminated, mask):
        # loss = min -E(Q-V(s))+1/4/alpha E[(Q-\gamma V(s'))^2]
        
        policy_out, log_prob_out = self.actor_sample(batch)
        q_inputs_flatten, next_q_inputs, current_q_inputs = self._build_inputs(batch, policy_out)
        q_values = self.q_net_1(q_inputs_flatten)   # [bs, seq_len, n_agents, n_actions]
        # q_values_2 = self.q_net_2(current_inputs_flatten)   # [bs, seq_len, n_agents, n_actions]
        with th.no_grad():
            q_next_value = self.target_q_net_1(next_q_inputs)
            next_v = q_next_value - self.args.sac_alpha * log_prob_out[:, 1:]
            next_v = next_v.sum(dim=-2)
        y_next = (1 - terminated) * self.args.gamma * next_v
        
        cur_v = self.q_net_1(current_q_inputs) - self.args.sac_alpha * log_prob_out[:, :-1]
        
        cur_v = cur_v.sum(dim=-2)
        q_values = q_values.sum(dim=-2)

        q_error = -(q_values - y_next)
        q_mask = mask.expand_as(q_error)
        #loss_1 = (q_mask * q_error).sum() / q_mask.sum()
        loss_1 = (q_mask[self.is_expert] * q_error[self.is_expert]).sum() / q_mask[self.is_expert].sum()
        value_error = (cur_v - y_next)
        value_mask = mask.expand_as(value_error)
        loss_2 = (value_mask * value_error).sum() / value_mask.sum()

        # loss2 = 1 / 4a * (Q(s, a) - v(s'))^2
        next_q_error = (q_values - y_next)
        loss_3 = 1.0 / (4 * self.args.alpha) * ((next_q_error * q_mask) ** 2).sum() / q_mask.sum() 
 
        loss = loss_1 + loss_2 + loss_3

        self.q_optimizer.zero_grad()
        loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_net_parameters, self.args.grad_norm_clip)
        self.q_optimizer.step()
        # for name, parms in self.q_net_1.named_parameters():	
            # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            # ' -->grad_value:',parms.grad)
        return loss, loss_1, loss_2, loss_3, q_grad_norm


    def _update_policy(self, batch, mask):
        policy_out, log_prob_out_ = self.actor_sample(batch)
        q_inputs_flatten, next_q_inputs, current_q_inputs = self._build_inputs(batch, policy_out)
        ## compute the policy loss
        q_values_1 = self.q_net_1(current_q_inputs)  # [bs, seq_len, n_agents, n_actions]
        # q_values_2 = self.q_net_2(current_input_flatten).sum(dim=-2)   # [bs, seq_len, n_agents, n_actions]
        
        # q_values = th.min(q_values_1, q_values_2)
        q_values = q_values_1
        log_prob_out = log_prob_out_[:, :-1] # [batch_size, seq_len, n_agents, n_actions]

        entropies = th.sum(
            log_prob_out, dim=-2, keepdim=False
        )
        q = th.sum(
            q_values, dim=-2, keepdim=False,
        )
        # q = q_values
        policy_loss = - q + self.args.sac_alpha * entropies
        policy_mask = mask.expand_as(policy_loss)
        #lmbda = 7.5 / ((policy_loss.detach().abs() * policy_mask).sum() / policy_mask.sum())
        policy_loss = (policy_loss * policy_mask).sum() / policy_mask.sum()
        #policy_loss = policy_loss * lmbda
        
        bc_loss = (policy_out - batch["actions"]) ** 2
        bc_loss = bc_loss[:, :-1].sum(dim=-2,keepdim=False)
        bc_mask = mask.expand_as(bc_loss)
        bc_loss = (bc_loss * bc_mask).sum() / bc_mask.sum()

        loss = policy_loss + 0 * bc_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
        self.policy_optimizer.step()

        return policy_loss, bc_loss, policy_grad_norm, q.mean()

    def build_batch_online(self, expert_batch: EpisodeBatch, agent_batch: EpisodeBatch, scheme, groups, preprocess):
        batch = partial(EpisodeBatch, scheme, groups, expert_batch.batch_size*2, expert_batch.max_seq_length,
                                 preprocess=preprocess, device=self.args.device)()
        
        obs_expert = expert_batch['obs']
        obs_agent = agent_batch['obs']
        obs = th.cat((obs_expert, obs_agent), dim=0)

        actions_expert = expert_batch['actions']
        actions_agent = agent_batch['actions']
        actions = th.cat((actions_expert, actions_agent), dim=0)

        terminated_expert = expert_batch['terminated']
        terminated_agent = agent_batch['terminated']
        terminated = th.cat((terminated_expert, terminated_agent), dim=0)

        filled_expert = expert_batch['filled']
        filled_agent = agent_batch['filled']
        filled = th.cat((filled_expert, filled_agent), dim=0)

        batch.update({"obs":obs})
        batch.update({"actions":actions})
        batch.update({"terminated":terminated})
        batch.update({"filled":filled})
        self.is_expert = torch.cat([torch.ones_like(th.tensor(range(expert_batch.batch_size)), dtype=torch.bool),
                           torch.zeros_like(th.tensor(range(expert_batch.batch_size)), dtype=torch.bool)], dim=0)
        return batch
    
    def _build_inputs(self, batch, policy_out=None, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        q_inputs , next_q_inputs, current_q_inputs = [], [], []
        q_inputs.append(batch["obs"][:, :-1, ts])
        next_q_inputs.append(batch["obs"][:, 1:, ts])
        current_q_inputs.append(batch["obs"][:, :-1, ts])

        q_inputs.append(batch["actions"][:, :-1, ts])
        next_q_inputs.append(policy_out[:, 1:])
        current_q_inputs.append(policy_out[:, :-1])
        # Don't need agent onehot_id
        # if self.args.obs_agent_id:
        #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        q_inputs = th.cat([x.reshape(bs, max_t-1, -1) for x in q_inputs], dim=-1)
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
        # for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            # target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.policy_net.cuda()
        self.q_net_1.cuda()
        # self.q_net_2.cuda()
        self.target_q_net_1.cuda()
        # self.target_q_net_2.cuda()
        # self.v_net.cuda()
        # self.target_v_net.cuda()

    def save_models(self, path):
        self.policy_net.save_models(path)
        # th.save(self.v_net.state_dict(), "{}/v_net.th".format(path))
        th.save(self.q_net_1.state_dict(), "{}/q_net_1.th".format(path))
        # th.save(self.q_net_2.state_dict(), "{}/q_net_2.th".format(path))
        th.save(self.policy_optimizer.state_dict(), "{}/policy_opt.th".format(path))
        # th.save(self.v_optimizer.state_dict(), "{}/v_opt.th".format(path))
        th.save(self.q_optimizer.state_dict(), "{}/q_opt.th".format(path))

    def load_models(self, path):
        self.policy_net.load_models(path)
        # self.v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        # self.target_v_net.load_models(th.load("{}/v_net.th".format(path), map_location=lambda storage, loc: storage))
        self.q_net_1.load_models(th.load("{}/q_net_1.th".format(path), map_location=lambda storage, loc: storage))
        # self.q_net_2.load_models(th.load("{}/q_net_2.th".format(path), map_location=lambda storage, loc: storage))
        self.policy_optimizer.load_models(th.load("{}/policy_opt.th".format(path), map_location=lambda storage, loc: storage))
        # self.v_optimizer.load_models(th.load("{}/v_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.q_optimizer.load_models(th.load("{}/q_opt.th".format(path), map_location=lambda storage, loc: storage))

    # def clone(self, batch: EpisodeBatch, t_env: int):
    #     actions = batch["actions"]
    #     terminated = batch["terminated"][:, :-1].float()
    #     mask = batch["filled"][:, :-1].float()
    #     mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    #     actions = actions[:, :-1]

    #     # No experiences to train on in this minibatch
    #     if mask.sum() == 0:
    #         self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
    #         self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
    #         return

    #     # mask = mask.repeat(1, 1, self.n_agents)

    #     for _ in range(self.args.epochs):
    #         policy_out = []
    #         self.policy_net.init_hidden(batch.batch_size)
    #         for t in range(batch.max_seq_length-1):
    #             agent_outs, log_prob = self.policy_net.forward(batch, t=t)
    #             policy_out.append(agent_outs)
    #         policy_out = th.stack(policy_out, dim=1) # [batch_size, seq_len, n_agents, n_actions]

    #         bc_loss = th.nn.MSELoss()(policy_out, copy.deepcopy(actions).float())
    #         # Optimise agents
    #         self.policy_optimizer.zero_grad()
    #         bc_loss.backward()
    #         policy_grad_norm = th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.grad_norm_clip)
    #         self.policy_optimizer.step()

    #     if t_env - self.log_stats_t >= self.args.learner_log_interval:
    #         self.logger.log_stat("behavior clone bc_loss", bc_loss.item(), t_env)
    #         self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)  
    #         self.log_stats_t = t_env

    # def getV(self, batch):
    #     """
    #     这里的action是重新根据采样得到的，并非用的expert的action
    #     """
    #     obs_action = []

    #     policy_out, log_prob_out = self.actor_sample(batch)
        
    #     obs_action.append(batch["obs"])
    #     current_entropies = th.sum(
    #         log_prob_out, dim=-2, keepdim=False
    #     )

    #     max_t = batch.max_seq_length
    #     obs_action.append(policy_out)
    #     q_inputs = th.cat([x.reshape(batch.batch_size, max_t, -1) for x in obs_action], dim=-1)
    #     current_Q_1 = self.q_net_1(q_inputs).sum(dim=-2)
    #     # current_Q_2 = self.q_net_2(q_inputs).sum(dim=-2)
    #     # current_Q = th.min(current_Q_1, current_Q_2)
    #     current_Q = current_Q_1
    #     current_V = current_Q - self.args.sac_alpha*current_entropies

    #     return current_V 

    

    # def q_loss(self, q_value, batch, inputs_flatten,terminated, mask ):
    #     current_inputs_flatten = inputs_flatten[:, :-1]
    #     next_q_inputs = inputs_flatten[:, 1:]
    #     # 4-deminson [batch_size, episode_len, agent_id, action_dim] 
    #     # For Q neqtwork
    #     current_V = self.getV(batch[:, :-1])
    #     with th.no_grad():
    #         next_V = self.getV(batch[:, 1:])

    #     # y = self.args.gamma * current_V
    #     y_next = (1 - terminated) * self.args.gamma * next_V

    #     q_error = -(q_value - y_next)
    #     # q_error = -(q_value - y_next)
    #     q_mask = mask.expand_as(q_error)
    #     # loss1 = -Q(s,a)+V^\pi(s)
    #     loss_1 = (q_mask * q_error).sum() / q_mask.sum()

    #     # value_loss = current_V - y_next
    #     # value_loss = (1-self.args.gamma)*current_V[:, 0]
    #     # loss_2 = (q_mask[:, 0] * value_loss).sum() / q_mask[:, 0].sum()

    #     value_error = current_V - y_next
    #     value_mask = mask.expand_as(value_error)
    #     loss_2 = (value_mask * value_error).sum() / value_mask.sum()

    #     # loss2 = 1 / 4a * (Q(s, a) - v(s'))^2
    #     next_q_error = (q_value - y_next)
    #     loss_3 = 1.0 / (4 * self.args.alpha) * ((next_q_error * q_mask) ** 2).sum() / q_mask.sum()

    #     loss = loss_1 + loss_3
    #     return loss, loss_1, loss_2, loss_3

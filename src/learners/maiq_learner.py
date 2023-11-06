# import copy
# from components.episode_buffer import EpisodeBatch
# from modules.mixers.vdn import VDNMixer
# from modules.mixers.qmix import QMixer
# import torch as th
# from torch.optim import RMSprop


# class MAIQLearner:
#     def __init__(self, mac, scheme, logger, args):
#         self.args = args
#         self.mac = mac
#         self.logger = logger

#         self.params = list(mac.parameters())

#         self.last_target_update_episode = 0

#         self.mixer = None

#         self.v_mixer = None

#         if args.mixer is not None:
#             if args.mixer == "vdn":
#                 self.mixer = VDNMixer()
#                 self.v_mixer = VDNMixer()
#             elif args.mixer == "qmix":
#                 self.mixer = QMixer(args).to(args.device)
#                 self.v_mixer = QMixer(args).to(args.device)
#             else:
#                 raise ValueError("Mixer {} not recognised.".format(args.mixer))
#             self.params += list(self.mixer.parameters())
#             self.params += list(self.v_mixer.parameters())
#             self.target_mixer = copy.deepcopy(self.mixer)
#             self.v_target_mixer = copy.deepcopy(self.v_mixer)

#         self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

#         # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
#         self.target_mac = copy.deepcopy(mac)

#         self.log_stats_t = -self.args.learner_log_interval - 1

#     def getV(self, Q):
#         return self.args.alpha * th.logsumexp(Q / self.args.alpha, dim=3)


#     def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
#         # Get the relevant quantities
#         rewards = batch["reward"][:, :-1]
#         actions = batch["actions"][:, :-1]
#         terminated = batch["terminated"][:, :-1].float()
#         mask = batch["filled"][:, :-1].float()
#         #t_mask = th.tensor(mask)
#         #t_mask[:, 1:] = t_mask[:, 1:] * (1 - terminated[:, :-1])
#         avail_actions = batch["avail_actions"]

#         # Calculate estimated Q-Values
#         mac_out = []
#         self.mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             agent_outs = self.mac.forward(batch, t=t)
#             mac_out.append(agent_outs)
#         mac_out = th.stack(mac_out, dim=1)  # Concat over time


#         # Pick the Q-Values for the actions taken by each agent
#         chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        


#         # Calculate the Q-Values necessary for the target
#         target_mac_out = []
#         self.target_mac.init_hidden(batch.batch_size)
#         for t in range(batch.max_seq_length):
#             target_agent_outs = self.target_mac.forward(batch, t=t)
#             target_mac_out.append(target_agent_outs)

#         # We don't need the first timesteps Q-Value estimate for calculating targets
        
#         target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

#         # Mask out unavailable actions
#         target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        
#         # 4-deminson [batch_size, episode_len, agent_id, action_dim] 
#         cur_v = self.getV(mac_out[:, :-1, : , :])
        
#         next_v = self.getV(target_mac_out)

#         # y = gamma * v(s')

#         # # Max over target Q-Values
#         # if self.args.double_q:
#         #     # Get actions that maximise live Q (for double q-learning)
#         #     mac_out_detach = mac_out.clone().detach()
#         #     mac_out_detach[avail_actions == 0] = -9999999
#         #     cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
#         #     target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
#         # else:
#         #     target_max_qvals = target_mac_out.max(dim=3)[0]

#         # Mix
#         if self.mixer is not None:
#             chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
#             cur_v = self.v_mixer(cur_v, batch["state"][:, :-1])
#             #target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
#             next_v = self.v_target_mixer(next_v, batch["state"][:, 1:])

#         y = (1 - terminated) * self.args.gamma * next_v
        
#         # loss1 = -(Q(s, a) - gamma * v(s'))
#         td_error1 = -(chosen_action_qvals - y.detach())

#         # mask1 = is_valid * (1. - terminated)
#         t_mask1 = mask.expand_as(td_error1)
#         loss1 = (t_mask1 * td_error1).sum() / t_mask1.sum()

#         # loss2 = v(s) - gamma * v(s')
#         td_error2 = (cur_v - y.detach())

#         # mask2 only for valid action
#         t_mask2 = mask.expand_as(td_error2)
#         loss2 = (t_mask2 * td_error2).sum() / t_mask2.sum()

#         # loss3 = 1 / 4a * (Q(s, a) - v(s))^2
#         loss3 = 1.0 / (4 * self.args.alpha) * ((t_mask1 * td_error1) ** 2).sum() / t_mask1.sum()

#         loss = loss1 + loss2 + loss3

#         # Calculate 1-step Q-Learning targets
#         # targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

#         # Optimise
#         self.optimiser.zero_grad()
#         loss.backward()
#         grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
#         self.optimiser.step()

#         if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
#             self._update_targets()
#             self.last_target_update_episode = episode_num

#         if t_env - self.log_stats_t >= self.args.learner_log_interval:
#             self.logger.log_stat("loss", loss.item(), t_env)
#             self.logger.log_stat("loss1", loss1.item(), t_env)
#             self.logger.log_stat("loss2", loss2.item(), t_env)
#             self.logger.log_stat("loss3", loss3.item(), t_env)
#             self.logger.log_stat("grad_norm", grad_norm, t_env)
#             mask_elems = mask.sum().item()
#             # self.logger.log_stat("td_error_abs1", (td_error1.abs().sum().item()/mask_elems), t_env)
#             # self.logger.log_stat("td_error_abs2", (td_error2.abs().sum().item()/mask_elems), t_env)
#             self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
#             #self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
#             self.log_stats_t = t_env

#     def _update_targets(self):
#         self.target_mac.load_state(self.mac)
#         if self.mixer is not None:
#             self.target_mixer.load_state_dict(self.mixer.state_dict())
#         if self.v_mixer is not None:
#             self.v_target_mixer.load_state_dict(self.v_mixer.state_dict())
#         self.logger.console_logger.info("Updated target network")

#     def cuda(self):
#         self.mac.cuda()
#         self.target_mac.cuda()
#         if self.mixer is not None:
#             self.mixer.cuda()
#             self.target_mixer.cuda()

#     def save_models(self, path):
#         self.mac.save_models(path)
#         if self.mixer is not None:
#             th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
#         th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

#     def load_models(self, path):
#         self.mac.load_models(path)
#         # Not quite right but I don't want to save target networks
#         self.target_mac.load_models(path)
#         if self.mixer is not None:
#             self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
#         self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.linear_mixer import LinearMixer

import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F

class MAIQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "linear_mixer":
                self.mixer = LinearMixer(args)
            elif args.mixer == "linear_abs_mixer":
                self.mixer = LinearMixer(args, abs="abs")
            elif args.mixer == "linear_relu_mixer":
                self.mixer = LinearMixer(args, abs="relu") 
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def getV(self, Q):
        return self.args.alpha * th.logsumexp(Q / self.args.alpha, dim=3)

    def clone(self, batch: EpisodeBatch, t_env: int, episode_num: int): # episoed_num is for updating the target network
        # Get the relevant quantities
        actions = batch["actions"]
        mask = batch["filled"].float()
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        mac_out[avail_actions[:, :] == 0] = -9999999
        mac_out = mac_out.view(-1, mac_out.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        bc_loss = F.cross_entropy(mac_out, actions.long().squeeze(-1), reduction='none')
        bc_loss = bc_loss.reshape(batch.batch_size, batch.max_seq_length, -1)
        mask = mask.expand_as(bc_loss) 
        bc_loss = (bc_loss * mask).sum() / mask.sum()

        self.optimiser.zero_grad()
        bc_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step() 


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("bc_loss", bc_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.log_stats_t = t_env
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        mac_out[avail_actions[:, :] == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # 4-deminson [batch_size, episode_len, agent_id, action_dim]
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # y = gamma * v(s')

        # TODO Is it necessary to use double q in multi_agent?

        # # Max over target Q-Values
        # if self.args.double_q:
        #     # Get actions that maximise live Q (for double q-learning)
        #     mac_out_detach = mac_out.clone().detach()
        #     mac_out_detach[avail_actions == 0] = -9999999
        #     cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        #     target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        # else:
        #     target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.args.mixer in ["linear_mixer", "linear_abs_mixer", "linear_relu_mixer"]:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            cur_v = self.mixer(mac_out[:, :-1, :, :], batch["state"][:, :-1], "V")
            next_v = self.target_mixer(target_mac_out, batch["state"][:, 1:], "V")

        elif self.mixer is not None:
            # calculate v = logsumexp(Q). Not applicable for mixers other than vdn
            cur_v = self.getV(mac_out[:, :-1, :, :])
            next_v = self.getV(target_mac_out)
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            cur_v = self.mixer(cur_v, batch["state"][:, :-1])
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            next_v = self.target_mixer(next_v, batch["state"][:, 1:])

        y = (1 - terminated) * self.args.gamma * next_v

        # loss1 = (Q(s, a) - gamma * v(s'))
        td_error1 = chosen_action_qvals - y.detach()
        # mask1 = is_valid * (1. - terminated)
        t_mask1 = mask.expand_as(td_error1)
        
        if self.args.divergence_type == "ForwardKL":
            # 1 + log(x)
            loss1 = (
                t_mask1 * (1 + th.log(th.clamp_min(td_error1, 1e-10)))
            ).sum() / t_mask1.sum()
        elif self.args.divergence_type == "ReverseKL":
            # -e^(-x-1)
            loss1 = (t_mask1 * (-th.exp(-td_error1 - 1))).sum() / t_mask1.sum()
        elif self.args.divergence_type == "Hellinger":
            # x / (x + 1)
            loss1 = (
                t_mask1 * th.clamp(th.div(td_error1, 1 + td_error1), -20, 20)
            ).sum() / t_mask1.sum()
        elif self.args.divergence_type == "PearsonChiSquared":
            # x - x^2/4 * \alpha
            loss1 = (t_mask1 * td_error1).sum() / t_mask1.sum() - 1.0 / (
                4 * self.args.alpha
            ) * ((t_mask1 * td_error1) ** 2).sum() / t_mask1.sum()
        elif self.args.divergence_type == "TotalVariation":
            # x
            loss1 = (t_mask1 * td_error1).sum() / t_mask1.sum()
        elif self.args.divergence_type == "JensenShannon":
            # log(2 - e^(-x))
            loss1 = (
                t_mask1 * th.log(th.clamp_min(2 - th.exp(-td_error1), 1e-10))
            ).sum() / t_mask1.sum()
        else:
            raise ValueError("Unknown divergence type")

        # maximize -> loss
        loss1 = -loss1

        # loss2 = v(s) - gamma * v(s')
        # mask2 only for valid action
        td_error2 = cur_v - y.detach()
        t_mask2 = mask.expand_as(td_error2)
        loss2 = (t_mask2 * td_error2).sum() / t_mask2.sum()

        loss = loss1 + loss2

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss1", loss1.item(), t_env)
            self.logger.log_stat("loss2", loss2.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs1", (td_error1.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("td_error_abs2", (td_error2.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env 

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

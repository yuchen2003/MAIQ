from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import imageio
import os

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        # self.train_returns = []
        # self.test_returns = []
        # self.train_stats = {}
        # self.test_stats = {}
        
        self.returns = {}
        self.stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, _=None, test_mode=False, path=None, tag='train', render=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        ep_obs = []
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # print(actions.shape)
            reward, terminated, env_info = self.env.step(actions[0].cpu().detach().numpy()) # 相关信息从env中读取
            if render:
                ep_obs.append(self.env.render(mode='rgb_array'))
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # cur_stats = self.test_stats if test_mode else self.train_stats
        # cur_returns = self.test_returns if test_mode else self.train_returns
        # log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        # cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        
        self.returns[tag] = self.returns.get(tag, []) + [episode_return]
        if tag not in self.stats:
            self.stats[tag] = {}
        self.stats[tag].update({k: self.stats[tag].get(k, 0) + env_info.get(k, 0) for k in set(self.stats[tag]) | set(env_info)})
        self.stats[tag]["n_episodes"] = 1 + self.stats[tag].get("n_episodes", 0)
        self.stats[tag]["ep_length"] = self.t + self.stats[tag].get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        # cur_returns.append(episode_return)
        
        if render:
            assert(path != None)
            with imageio.get_writer(os.path.join(path, f'{_}_{episode_return}.gif'), fps=5) as writer:
                for obs in ep_obs:
                    writer.append_data(obs)

        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #     self.log_train_stats_t = self.t_env

        return self.batch

    def log_info(self, tag="train"):
        log_dic = {}
        log_dic["return_mean"] = float(np.mean(self.returns[tag]))
        for k, v in self.stats[tag].items():
            if k != "n_episodes":
                log_dic[f"{k}_mean"] = float(v/self.stats[tag]["n_episodes"])
        if hasattr(self.mac, 'action_selector') and hasattr(self.mac.action_selector, "epsilon"):
                log_dic['epsilon'] = float(self.mac.action_selector.epsilon)
        self.stats[tag].clear()
        self.returns[tag].clear()

        log_dic = {f"{tag}/{k}": v for k, v in log_dic.items()}
        for k, v in log_dic.items():
            self.logger.log_stat(k, v, self.t_env)
        return log_dic

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

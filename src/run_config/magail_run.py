import datetime
import multiprocessing
import os
import pickle
import pprint
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import torch
import torch as th
# from modules.magail.discriminator import Discriminator
import torch.nn.functional as F
from torch.utils.data import DataLoader

from components.episode_buffer import ReplayBuffer
# from components.expert_dataset import ExpertDataSet
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from modules.magail import REGISTRY as discriminator_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.rl_utils import get_all_for_magail
from utils.timehelper import time_left, time_str


def run_magail(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # expert_data_path = args.expert_data_path + '/100tra.pkl'
    # print(expert_data_path)
    # setup loggers
    results_save_dir = args.results_save_dir
    logger = Logger(_log, results_save_dir)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    if args.use_wandb:
        wandb_exp_direc = os.path.join(results_save_dir, 'wandb_logs')
        logger.setup_wandb(wandb_exp_direc, args)


    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def render_sequential(args, runner):
    runner.run_render(args.expert_data_path)

    runner.close_env()


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    disc_type = args.disc_type

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    print("Scheme", scheme)
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # expert_data_path = args.expert_data_path + '/' + str(args.num_trajs) + 'tra.pkl'
    # expert_dataset = ExpertDataSet(scheme, groups, args.expert_buffer_size, env_info["episode_limit"] + 1,
    #                                expert_path=expert_data_path,
    #                                preprocess=preprocess,
    #                                device="cpu" if args.buffer_cpu_only else args.device,
    #                                logger=logger)
    expert_dataset = pickle.load(open(args.load_dataset_dir, 'rb')) 

    ################## 测试D性能是否正常
    # expert_data_path_test = args.expert_data_path + '/' + str(10000) + 'tra.pkl'
    # expert_dataset_test_D = ExpertDataSet(scheme, groups, args.expert_buffer_size, env_info["episode_limit"] + 1,
    #                                expert_path=expert_data_path_test,
    #                                preprocess=preprocess,
    #                                device="cpu" if args.buffer_cpu_only else args.device,
    #                                logger=logger)

    if disc_type == 'decentralized':
        discriminator = [
                            discriminator_REGISTRY[args.discriminator](env_info["obs_shape"], env_info["n_actions"], args.n_agents,
                                          disc_type=disc_type).to(args.device)
                        ] * env_info["n_agents"]
    elif disc_type == 'centralized':
        # TODO 仍需验证解决方案是否正确
        discriminator = discriminator_REGISTRY[args.discriminator](env_info["obs_shape"] * env_info["n_agents"],
                                      env_info["n_actions"] * env_info["n_agents"], args.n_agents,
                                      disc_type=disc_type).to(args.device)
    else:
        assert False

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # Optimizer
    if disc_type == 'centralized':
        optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.dis_lr)
    else:
        optimizer_discriminator = th.optim.Adam(discriminator[0].parameters(), lr=args.dis_lr)

    # TODO 调整调度
    scheduler_discriminator = th.optim.lr_scheduler.StepLR(optimizer_discriminator,
                                                           step_size=500,
                                                           gamma=0.95)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "" or (args.expert_data_path != "" and args.render):
        path = args.checkpoint_path if args.checkpoint_path != "" else args.expert_data_path
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(path):
            full_name = os.path.join(path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.render:
            logger.console_logger.info("In render")
            render_sequential(args, runner)
            return
        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning behavior clone for {} timesteps".format(args.bc_iters))

    for bc_t in range(args.bc_iters):
        expert_sample = expert_dataset.sample(args.batch_size)
        max_ep_t_expert = expert_sample.max_t_filled()
        expert_sample = expert_sample[:, :max_ep_t_expert]
        if expert_sample.device != args.device:
            expert_sample.to(args.device)
        learner.clone(expert_sample, bc_t)
    logger.console_logger.info("Ending behavior clone for {} timesteps".format(args.bc_iters))

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        # print(expert_batch_state.shape)
        episode_batch = runner.run(test_mode=False)

        buffer.insert_episode_batch(episode_batch)
        # print('EEEEEe', d_iter, episode_batch.batch_size, args.batch_size)
        # 获取训练轨迹
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            expert_sample = expert_dataset.sample(args.batch_size)
            # print("Reward", expert_sample["reward"][:, :-1])
            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            max_ep_t_expert = expert_sample.max_t_filled()

            episode_sample = episode_sample[:, :max_ep_t]
            expert_sample = expert_sample[:, :max_ep_t_expert]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            if expert_sample.device != args.device:
                expert_sample.to(args.device)

            # print("Check episode sample shape", expert_sample.data.transition_data['obs'][:, :-1].shape)
            # if runner.t_env > 20000 and args.stop_training is True:
            #     print("Stop training!!!", args.stop_training)
            # expert_sample_test_D = expert_dataset_test_D.sample(args.batch_size)
            # if expert_sample_test_D.device != args.device:
            #     expert_sample_test_D.to(args.device)
            for d_iter in range(args.d_iters):
                if disc_type == 'centralized':
                    # print(episode_sample.data.transition_data['obs'], episode_sample.data.transition_data['actions_onehot'])
                    gen_r = train(args, discriminator, episode_sample, expert_sample, logger, runner.t_env, disc_type,
                                  optimizer_discriminator=optimizer_discriminator,
                                  scheduler_discriminator=scheduler_discriminator)
                elif disc_type == 'decentralized':
                    gen_r = None
                    for k in range(env_info["n_agents"]):
                        if gen_r is None:
                            gen_r = train(args, discriminator[k], episode_sample, expert_sample, logger, runner.t_env,
                                          disc_type, k, optimizer_discriminator, scheduler_discriminator)
                        else:
                            gen_r = th.cat(
                                (gen_r, train(args, discriminator[k], episode_sample, expert_sample, logger, runner.t_env,
                                              disc_type, k, optimizer_discriminator, scheduler_discriminator)
                                 ), -1)

                else:
                    assert False

            # print("Check gen reward shape", gen_r.shape)
            if runner.t_env > 1000:
                learner.train(episode_sample, runner.t_env, episode, rewards=gen_r)

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += args.batch_size_run
            # args.log_interval = 1
            # print("Runner env timestep", runner.t_env, args.log_interval)
            if (runner.t_env - last_log_T) >= args.log_interval:
                # print("save log:", runner.t_env)
                logger.log_stat("episode", episode, runner.t_env)
                # logger.log_stat("dis_reward_max", th.max(gen_r), runner.t_env)
                # logger.log_stat("dis_reward_min", th.min(gen_r), runner.t_env)
                # logger.log_stat("dis_reward_sum", th.sum(gen_r), runner.t_env)
                # logger.log_stat("dis_reward_mean", th.mean(gen_r), runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def train(args, discriminator, episode_sample, expert_sample, logger, t_env, disc_type, n_agent: int = None,
          optimizer_discriminator=None, scheduler_discriminator=None):
    if disc_type == 'centralized':
        if args.discriminator == "fc":
            gen_r = discriminator(get_all_for_magail(episode_sample.data.transition_data['obs'][:, :-1]),
                                  get_all_for_magail(episode_sample.data.transition_data['actions_onehot'][:, :-1]))
            expert_r = discriminator(get_all_for_magail(expert_sample.data.transition_data['obs'][:, :-1]),
                                     get_all_for_magail(expert_sample.data.transition_data['actions_onehot'][:, :-1]))
        elif args.discriminator == 'rnn':
            # 如果采用rnn，需要将traj拆成每个timestep
            gen_r_lists = []
            hidden_state = discriminator.init_hidden(args.batch_size)
            for t in range(episode_sample.max_seq_length-1):
                gen_r, hidden_state = discriminator(get_all_for_magail(episode_sample.data.transition_data['obs'][:, t], "rnn"),
                                      get_all_for_magail(episode_sample.data.transition_data['actions_onehot'][:, t], "rnn"),
                                                    hidden_state=hidden_state)
                gen_r_lists.append(gen_r)

            expert_r_lists = []
            hidden_state = discriminator.init_hidden(args.batch_size)
            for t in range(expert_sample.max_seq_length-1):
                expert_r, hidden_state = discriminator(get_all_for_magail(expert_sample.data.transition_data['obs'][:, t], "rnn"),
                                         get_all_for_magail(expert_sample.data.transition_data['actions_onehot'][:, t], "rnn"),
                                                       hidden_state=hidden_state)
                expert_r_lists.append(expert_r)

            # 将生成的reward合并
            gen_r = th.stack(gen_r_lists, dim=1)
            expert_r = th.stack(expert_r_lists, dim=1)
            # print(gen_r.shape[:2], (args.batch_size, episode_sample.max_seq_length-1, ))
            assert gen_r.shape[:2]==(args.batch_size, episode_sample.max_seq_length-1, )

    else:
        if args.discriminator == "fc":
            # print("Check episode shape:", episode_sample.data.transition_data['obs'][:, :-1].shape)
            gen_r = discriminator(episode_sample.data.transition_data['obs'][:, :-1, n_agent],
                                  episode_sample.data.transition_data['actions_onehot'][:, :-1, n_agent])
            expert_r = discriminator(expert_sample.data.transition_data['obs'][:, :-1, n_agent],
                                     expert_sample.data.transition_data['actions_onehot'][:, :-1, n_agent])
        elif args.discriminator == 'rnn':
            # 如果采用rnn，需要将traj拆成每个timestep
            gen_r_lists = []
            hidden_state = discriminator.init_hidden(args.batch_size)
            for t in range(episode_sample.max_seq_length-1):
                gen_r, hidden_state = discriminator(get_all_for_magail(episode_sample.data.transition_data['obs'][:, t, n_agent], "rnn"),
                                                    get_all_for_magail(episode_sample.data.transition_data['actions_onehot'][:, t, n_agent], "rnn"),
                                                    hidden_state=hidden_state)
                gen_r_lists.append(gen_r)

            expert_r_lists = []
            hidden_state = discriminator.init_hidden(args.batch_size)
            for t in range(expert_sample.max_seq_length-1):
                expert_r, hidden_state = discriminator(get_all_for_magail(expert_sample.data.transition_data['obs'][:, t, n_agent], "rnn"),
                                                       get_all_for_magail(expert_sample.data.transition_data['actions_onehot'][:, t, n_agent], "rnn"),
                                                       hidden_state=hidden_state)
                expert_r_lists.append(expert_r)

            # 将生成的reward合并
            gen_r = th.stack(gen_r_lists, dim=1)
            expert_r = th.stack(expert_r_lists, dim=1)
            assert gen_r.shape[:2]==(args.batch_size, episode_sample.max_seq_length-1, )

    mask_gen = episode_sample["filled"][:, :-1].float()
    terminated_gen = episode_sample["terminated"][:, :-1].float()
    mask_gen[:, 1:] = mask_gen[:, 1:] * (1 - terminated_gen[:, :-1])
    gen_r = gen_r * mask_gen

    mask_expert = expert_sample["filled"][:, :-1].float()
    terminated_expert = expert_sample["terminated"][:, :-1].float()
    mask_expert[:, 1:] = mask_expert[:, 1:] * (1 - terminated_expert[:, :-1])
    expert_r = expert_r * mask_expert

    expert_labels = th.ones_like(expert_r)
    expert_labels = expert_labels * mask_expert

    gen_labels = th.zeros_like(gen_r)

    # label smoothing for discriminator
    if args.use_label_smoothing:
        smoothing_rate = args.label_smooth_rate
        expert_labels *= (1 - smoothing_rate)
        gen_labels += th.ones_like(gen_r) * smoothing_rate

    e_loss = F.binary_cross_entropy_with_logits(expert_r, expert_labels)
    g_loss = F.binary_cross_entropy_with_logits(gen_r, gen_labels)
    d_loss = g_loss + e_loss

    # expert_r_test_D = discriminator(get_all_for_magail(expert_sample_test_D.data.transition_data['obs'][:, :-1]),
    #                                 get_all_for_magail(expert_sample_test_D.data.transition_data['actions_onehot'][:, :-1]))
    # mask_expert_test_D = expert_sample_test_D["filled"][:, :-1].float()
    # terminated_expert_test_D = expert_sample_test_D["terminated"][:, :-1].float()
    # mask_expert_test_D[:, 1:] = mask_expert_test_D[:, 1:] * (1 - terminated_expert_test_D[:, :-1])
    # expert_r_test_D = expert_r_test_D * mask_expert_test_D
    #
    # expert_labels_test_D = th.ones_like(expert_r_test_D)
    # expert_labels_test_D = expert_labels_test_D * mask_expert_test_D
    # expert_loss_test_D = th.nn.BCELoss()(expert_r_test_D, expert_labels_test_D)
    # gen_r_sigmoid = th.sigmoid(gen_r)
    # gen_r_ = th.log(gen_r_sigmoid)
    with torch.no_grad():
        gen_r_ = -F.logsigmoid(-gen_r)

    logger.log_stat("dis loss", d_loss, t_env)
    # logger.log_stat("dis test loss", expert_loss_test_D, t_env)
    logger.log_stat("g loss", g_loss, t_env)
    logger.log_stat("e loss", e_loss, t_env)
    # print("Dis lr ", scheduler_discriminator.get_lr())
    logger.log_stat("dis lr", scheduler_discriminator.get_lr()[-1], t_env)
    logger.log_stat("dis_reward_max", th.max(gen_r_), t_env)
    logger.log_stat("dis_reward_min", th.min(gen_r_), t_env)
    logger.log_stat("dis_reward_sum", th.sum(gen_r_), t_env)
    logger.log_stat("dis_reward_mean", th.mean(gen_r_), t_env)
    # logger.log_stat("dis_reward_without_log_mean", th.mean(gen_r_sigmoid), t_env)

    # """ WGAN with Gradient Penalty"""
    # d_loss = gen_r.mean() - expert_r.mean()
    # differences_batch_state = gen_batch_state[:expert_batch_state.size(0)] - expert_batch_state
    # differences_batch_action = gen_batch_action[:expert_batch_action.size(0)] - expert_batch_action
    # alpha = torch.rand(expert_batch_state.size(0), 1)
    # interpolates_batch_state = gen_batch_state[:expert_batch_state.size(0)] + (alpha * differences_batch_state)
    # interpolates_batch_action = gen_batch_action[:expert_batch_action.size(0)] + (alpha * differences_batch_action)
    # gradients = torch.cat([x for x in map(grad_collect_func, self.D(interpolates_batch_state, interpolates_batch_action))])
    # slopes = torch.norm(gradients, p=2, dim=-1)
    # gradient_penalty = torch.mean((slopes - 1.) ** 2)
    # d_loss += 10 * gradient_penalty
    # if not args.stop_training:
    optimizer_discriminator.zero_grad()
    d_loss.backward()
    th.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_norm_clip)
    optimizer_discriminator.step()
    scheduler_discriminator.step()
    return gen_r_


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config

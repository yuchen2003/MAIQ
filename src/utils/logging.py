from collections import defaultdict
import logging
import numpy as np
import wandb
import time
import os
import json

class Logger:
    def __init__(self, console_logger, result_dir):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.use_wandb = False
        
        self.result_dir = result_dir

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_wandb(self, directory_name, args):
        os.makedirs(os.path.join(directory_name, 'wandb'), exist_ok=True)
        project_name = 'MAIQ'
        name = time.strftime("%m-%d-%H:%M:%S") + '_seed' + str(args.seed)
        self.use_wandb = True
        config = wandb.config
        args_dict = vars(args)
        run_name = args.remark + '_' + args.name + '_' + args.mixer
        if args.is_bc:
            run_name += '_bc'
        
        try:
            map_group_name = args.env_args['map_name']
        except:
            map_group_name = args.env_args['key']
        
        wandb.init(name=name, 
                   config=args_dict, 
                   dir=directory_name, 
                   project=project_name, 
                   group=args.env + '_' + map_group_name, 
                   job_type=run_name,
                   entity="liyc-group")

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        value = float(value)
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)
        if self.use_wandb:
            wandb.log({key: value}, step=t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        json.dump(self.stats, open(os.path.join(self.result_dir, 'metrics.json'), 'w'))
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger


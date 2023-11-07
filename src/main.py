import numpy as np
import os
import datetime
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
from run_config.sample import run_sample
from run_config.maiq_run_cont_online import maiq_cont_run_online
from run_config.maiq_run import maiq_run
from run_config.maiq_run_online import maiq_run_online
from run_config.magail_run import run_magail
from run_config.maairl_run import run_maairl
from run_config.render import run_render
from run_config.bc_run import bc_run

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

os.environ["WANDB_MODE"] = "offline"

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    run_func = {
        "bc": bc_run,
        "render": run_render,
        "run": run,
        "sample": run_sample,
        "maiq": maiq_run,
        "maiq_online": maiq_run_online,
        "maiq_cont_online": maiq_cont_run_online,
        "magail": run_magail,
        "maairl": run_maairl,
    }
    # run the framework
    run_func[config['run_file']](_run, config, _log)


def _get_run_file(params):
    run_file = None
    for _i, _v in enumerate(params):
        # print("```", _i, _v)
        if _v.startswith('--') and _v.split("=")[0] == "--run_file":
            run_file = _v.split("=")[1]
            del params[_i]
            return run_file
    return run_file


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        if config_name in ["simple_adv"]:
            config_name = "gymma"
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def _get_argv_config(params):
    config = {}
    to_del = []
    for _i, _v in enumerate(params):
        item = _v.split("=")[0]
        if item[:2] == "--" and item not in ["envs", "algs"]:
            config_v = _v.split("=")[1]
            try:
                config_v = eval(config_v)
            except:
                pass
            config[item[2:]] = config_v
            to_del.append(_v)
    for _v in to_del:
        params.remove(_v)
    return config


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    th.set_num_threads(1)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    run_file = _get_run_file(params)
    if run_file is None:
        run_file = 'run'
    config_dict['run_file'] = run_file

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    commandline_options = _get_argv_config(params)
    config_dict = recursive_dict_update(config_dict, commandline_options)

    config_dict['remark'] = '_' + \
        config_dict['remark'] if 'remark' in config_dict else ''
    unique_token = "{}{}_{}".format(
        config_dict['name'], config_dict['remark'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    results_save_dir = os.path.join(
        results_path,
        config_dict['env'] + os.sep +
        config_dict['env_args']['map_name'] if 'map_name' in config_dict['env_args'] else config_dict['env'],
        config_dict['name'] + config_dict['remark'],
        unique_token
    )
    results_dir_without_token = os.path.join(dirname(results_save_dir))
    results_save_dir = os.path.join(
        results_path,
        config_dict['env'] + os.sep +
        config_dict['env_args']['map_name'] if 'map_name' in config_dict['env_args'] else config_dict['env'],
        config_dict['name'] + config_dict['remark'],
        unique_token
    )
    video_path = os.path.join(results_save_dir, 'video')

    os.makedirs(results_save_dir, exist_ok=True)
    config_dict['results_save_dir'] = results_save_dir
    config_dict['results_dir_without_token'] = results_dir_without_token
    config_dict['video_path'] = video_path
    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

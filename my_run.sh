#!/bin/bash

# 1. base MARL algs
# concurrent run, printing got mixed
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000 save_model=True &
# python3 src/main.py --config=vdn --env-config=sc2 with env_args.map_name=3m t_max=2050000 save_model=True &
# wait
# echo "all subprocesses ended."

# 2. sampling expert model, should specify arguments of corresponding model in 'sample.yaml'
# python3 src/main.py --config=sample --run_file=sample --env-config=sc2 with env_args.map_name=3m obs_last_action=False
# 
# 3. MAIL algs
# python3 src/main.py --config=bc --run_file=bc --env-config=sc2 with env_args.map_name=3m # this has been deprecated

# python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=3m is_bc=True



# handscript
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000
python3 src/main.py --config=sample --run_file=sample --env-config=gymma with env_args.key=mpe:SimpleTag-v0 episode_cnt=100
python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.time_limit=25 env_args.key=mpe:SimpleTag-v0 is_bc=True
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maairl --run_file=maairl --env-config=gymma with env_args.time_limit=25 env_args.key=mpe:SimpleTag-v0 load_dataset_dir=dataset/mpe:SimpleTag-v0__100.pkl bc_iters=0 remark=no_bc_start
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=magail --run_file=magail --env-config=gymma with env_args.time_limit=25 env_args.key=mpe:SimpleTag-v0 load_dataset_dir=dataset/mpe:SimpleTag-v0__100.pkl bc_iters=0 remark=no_bc_start

python3 src/main.py --config=maddpg --env-config=gymma with env_args.time_limit=25 env_args.key=mpe:SimpleTag-v0  t_max=25000 log_interval=1000

# -------------- continuous -----------------
python3 src/main.py --config=sac --env-config=gymma with env_args.key=mpe:SimpleTag-v0
# -------------- continuous -----------------

# -------------- divergence -----------------
divergence_type: "TotalVariation" # TotalVariation, PearsonChiSquared, Hellinger, ForwardKL, ReverseKL, JensenShannon
remark: "TotalVariation"
python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2050000 divergence_type=ForwardKL remark=ForwardKL
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=ForwardKL remark=ForwardKL
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=TotalVariation remark=TotalVariation
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=PearsonChiSquared remark=PearsonChiSquared
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=Hellinger remark=Hellinger
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=ReverseKL remark=ReverseKL
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=gymma with env_args.key=mpe:SimpleTag-v0 t_max=1050000 divergence_type=JensenShannon remark=JensenShannon
# -------------- divergence -----------------

# mpe envs
# "multi_speaker_listener": "MultiSpeakerListener-v0",
# "simple_adversary": "SimpleAdversary-v0",
# "simple_crypto": "SimpleCrypto-v0",
# "simple_push": "SimplePush-v0",
# "simple_reference": "SimpleReference-v0",
# "simple_speaker_listener": "SimpleSpeakerListener-v0",
# "simple_spread": "SimpleSpread-v0",
# "simple_tag": "SimpleTag-v0",
# "simple_world_comm": "SimpleWorldComm-v0",
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
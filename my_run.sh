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
# python3 src/main.py --config=bc --run_file=bc_run  --env-config=sc2 with env_args.map_name=3m # this may have be deprecated

# python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=3m is_bc=True



# handscript
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000 &
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000  

python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s_vs_5z t_max=2060000 save_model=True &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s_vs_5z t_max=2060000 &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s_vs_5z t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2060000 save_model=True &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m t_max=2060000 &
wait

python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_9m t_max=2060000 save_model=True &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_9m t_max=2060000 &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_9m t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=10m_vs_11m t_max=2060000 save_model=True &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=10m_vs_11m t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=10m_vs_11m t_max=2060000 &
wait

python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 save_model=True &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg t_max=2060000 save_model=True &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg t_max=2060000 &
wait
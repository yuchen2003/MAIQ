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
# python3 src/main.py --config=bc --run_file=bc_run  --env-config=sc2 with env_args.map_name=3m # this has been deprecated

# python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=3m is_bc=True



# handscript
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000 &
# python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m t_max=2050000  

python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m save_model=True &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m &
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 save_model=True &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2 t_max=2060000 &
wait

# python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key=mpe:SimpleTag-v0

# test different divergences: TotalVariation, PearsonChiSquared, Hellinger, ForwardKL, ReverseKL, JensenShannon
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=TotalVariation remark=TotalVariation &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=Hellinger remark=Hellinger &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=ForwardKL remark=ForwardKL &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=ReverseKL remark=ReverseKL &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=PearsonChiSquared remark=PearsonChiSquared &
CUDA_VISIBLE_DEVICES=1 python3 src/main.py --config=maiq --run_file=maiq --env-config=sc2 with env_args.map_name=5m_vs_6m divergence_type=JensenShannon remark=JensenShannon &
wait

echo a1 && sleep 2 && echo a & 
echo b1 && sleep 1 && echo b & 
wait

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
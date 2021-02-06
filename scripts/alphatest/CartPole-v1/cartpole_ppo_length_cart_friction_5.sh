#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/workspace/rl-plasticity-experiments/src || exit
python experiments.py --algo_name ppo2 \
					  --env_name CartPole-v1 \
					  --num_iterations 8 \
					  --param_names length,cart_friction \
					  --runs_for_probability_estimation 5 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/workspace/rl-plasticity-experiments \
					  --logging_level INFO
cd ..

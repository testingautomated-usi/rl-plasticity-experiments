#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/workspace/rl-plasticity-experiments/src || exit
python experiments.py --algo_name sac \
					  --env_name Pendulum-v0 \
					  --num_iterations 11 \
					  --param_names dt,length \
					  --runs_for_probability_estimation 5 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name ppo2 \
					  --env_name Pendulum-v0 \
					  --num_iterations 127 \
					  --param_names dt,mass,length \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

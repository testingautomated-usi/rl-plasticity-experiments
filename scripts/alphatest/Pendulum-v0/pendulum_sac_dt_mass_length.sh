#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name sac \
					  --env_name Pendulum-v0 \
					  --num_iterations 95 \
					  --param_names dt,mass,length \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

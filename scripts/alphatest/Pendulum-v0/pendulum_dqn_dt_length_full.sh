#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name dqn \
					  --env_name Pendulum-v0 \
					  --num_iterations 19 \
					  --param_names dt,length \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO \
					  --full_training_time

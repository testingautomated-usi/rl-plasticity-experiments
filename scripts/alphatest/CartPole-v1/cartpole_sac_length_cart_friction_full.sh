#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name sac \
					  --env_name CartPole-v1 \
					  --num_iterations 8 \
					  --param_names length,cart_friction \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO \
					  --full_training_time

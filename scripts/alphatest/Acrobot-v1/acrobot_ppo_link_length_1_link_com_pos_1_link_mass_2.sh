#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name ppo2 \
					  --env_name Acrobot-v1 \
					  --num_iterations 79 \
					  --param_names link_length_1,link_com_pos_1,link_mass_2 \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

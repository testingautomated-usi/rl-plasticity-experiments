#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name dqn \
					  --env_name MountainCar-v0 \
					  --num_iterations 53 \
					  --param_names gravity,force \
					  --runs_for_probability_estimation 5 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

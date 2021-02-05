#!/bin/bash

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name ppo2 \
					  --env_name MountainCar-v0 \
					  --num_iterations 47 \
					  --param_names gravity,force,goal_velocity \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

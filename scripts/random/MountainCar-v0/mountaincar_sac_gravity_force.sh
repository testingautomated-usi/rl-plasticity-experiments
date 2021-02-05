#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name sac \
					  --env_name MountainCar-v0 \
					  --num_iterations 82 \
					  --param_names gravity,force \
					  --runs_for_probability_estimation 3 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO \
					  --search_type random

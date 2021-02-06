#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/workspace/rl-plasticity-experiments/src || exit
python experiments.py --algo_name ppo2 \
					  --env_name MountainCar-v0 \
					  --num_iterations 15 \
					  --param_names gravity,force \
					  --runs_for_probability_estimation 5 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/workspace/rl-plasticity-experiments \
					  --logging_level INFO
cd ..

#!/bin/bash
conda activate rl-plasticity-experiments

cd ~/rl-plasticity-experiments/src || exit
python experiments.py --algo_name dqn \
					  --env_name CartPole-v1 \
					  --num_iterations 11 \
					  --param_names length,cart_friction \
					  --runs_for_probability_estimation 5 \
					  --num_search_iterations 5 \
					  --home_abs_path ~/rl-plasticity-experiments \
					  --logging_level INFO

#!/bin/bash

algos=('ppo2' 'sac' 'dqn')

function compute_effectiveness {
  local env=$1
  local algo=$2
  local param_names=$3 # comma-separated values

  num_of_parameters=0
  delimiter=","
  con_cat_string=$param_names$delimiter
  split_multi_char=()
  while [[ $con_cat_string ]]; do
    split_multi_char+=( "${con_cat_string%%"$delimiter"*}" )
    con_cat_string=${con_cat_string#*"$delimiter"}
  done

  param_names_dir=''
  for word in "${split_multi_char[@]}"; do
    ((num_of_parameters=num_of_parameters+1))
    param_names_dir=$param_names_dir$word'_'
  done
  param_names_dir=${param_names_dir:0:((${#param_names_dir}-1))}

  local save_dir=~/Desktop/rl-experiments-artifacts/cluster/$env/results/$param_names_dir/$algo
  mkdir -p $save_dir

  python analyze_effectiveness_results.py --save_dir $save_dir --algo_name $algo \
    --first_mode_dir ~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo \
    --second_mode_dir ~/Desktop/rl-experiments-artifacts/cluster/$env/random/$param_names_dir/$algo \
    --param_names $param_names --env_name $env

}

# run with source ./analyze_effectiveness_results.sh in order that the command conda is recognized
conda activate rl-plasticity-experiments

cd src || exit

echo '************** CartPole-v1 length, cart_friction **************'
compute_effectiveness CartPole-v1 ppo2 length,cart_friction
compute_effectiveness CartPole-v1 sac length,cart_friction
compute_effectiveness CartPole-v1 dqn length,cart_friction
echo '----------------------------------------------------------------'

echo '************** Pendulum-v0 dt, length **************'
compute_effectiveness Pendulum-v0 ppo2 dt,length
compute_effectiveness Pendulum-v0 sac dt,length
compute_effectiveness Pendulum-v0 dqn dt,length
echo '----------------------------------------------------------------'

echo '************** MountainCar-v0 force, gravity **************'
compute_effectiveness MountainCar-v0 ppo2 force,gravity
compute_effectiveness MountainCar-v0 sac force,gravity
compute_effectiveness MountainCar-v0 dqn force,gravity
echo '----------------------------------------------------------------'

echo '************** Acrobot-v1 link_length_1, link_com_pos_1 **************'
compute_effectiveness Acrobot-v1 ppo2 link_length_1,link_com_pos_1
compute_effectiveness Acrobot-v1 sac link_length_1,link_com_pos_1
compute_effectiveness Acrobot-v1 dqn link_length_1,link_com_pos_1
echo '----------------------------------------------------------------'

cd ..
#!/bin/bash

algos=('ppo2' 'sac' 'dqn')

function compute_number_of_executions_skipped {
  local env=$1
  local algo=$2
  local param_names=$3 # comma-separated values
  local runs_prob_est=$4
  local training_time=$5

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

  if [[ $runs_prob_est -gt 3 ]]; then
    param_names_dir=$param_names_dir'_'$runs_prob_est
  fi

  if [[ $training_time == "full" ]]; then
    param_names_dir=$param_names_dir'_'$training_time
  fi

  python analyze_number_of_executions_skipped_results.py \
    --dir ~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo \
    --env_name $env --param_names $param_names --algo_name $algo

}

# run with source ./get_min_max_num_iterations.sh in order that the command conda is recognized
conda activate stable-baselines

cd src || exit

rpe=3
echo "++++++++++++++ 2 parameters ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_number_of_executions_skipped CartPole-v1 ppo2 length,cart_friction $rpe
compute_number_of_executions_skipped CartPole-v1 sac length,cart_friction $rpe
compute_number_of_executions_skipped CartPole-v1 dqn length,cart_friction $rpe
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_number_of_executions_skipped Pendulum-v0 ppo2 dt,length $rpe
compute_number_of_executions_skipped Pendulum-v0 sac dt,length $rpe
compute_number_of_executions_skipped Pendulum-v0 dqn dt,length $rpe
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_number_of_executions_skipped MountainCar-v0 ppo2 force,gravity $rpe
compute_number_of_executions_skipped MountainCar-v0 sac force,gravity $rpe
compute_number_of_executions_skipped MountainCar-v0 dqn force,gravity $rpe
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_number_of_executions_skipped Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $rpe
compute_number_of_executions_skipped Acrobot-v1 sac link_length_1,link_com_pos_1 $rpe
compute_number_of_executions_skipped Acrobot-v1 dqn link_length_1,link_com_pos_1 $rpe
echo '----------------------------------------------------------------'

echo '==================================================================='

echo "++++++++++++++ 3 parameters ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction, masspole **************"
compute_number_of_executions_skipped CartPole-v1 ppo2 length,cart_friction,masspole $rpe
compute_number_of_executions_skipped CartPole-v1 sac length,cart_friction,masspole $rpe
compute_number_of_executions_skipped CartPole-v1 dqn length,cart_friction,masspole $rpe
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length, mass **************"
compute_number_of_executions_skipped Pendulum-v0 ppo2 dt,length,mass $rpe
compute_number_of_executions_skipped Pendulum-v0 sac dt,length,mass $rpe
compute_number_of_executions_skipped Pendulum-v0 dqn dt,length,mass $rpe
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity, goal_velocity **************"
compute_number_of_executions_skipped MountainCar-v0 ppo2 force,gravity,goal_velocity $rpe
compute_number_of_executions_skipped MountainCar-v0 sac force,gravity,goal_velocity $rpe
compute_number_of_executions_skipped MountainCar-v0 dqn force,gravity,goal_velocity $rpe
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2 **************"
compute_number_of_executions_skipped Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2 $rpe
compute_number_of_executions_skipped Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2 $rpe
compute_number_of_executions_skipped Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2 $rpe
echo '----------------------------------------------------------------'

echo '==================================================================='

echo "++++++++++++++ 4 parameters ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction, masspole, masscart **************"
compute_number_of_executions_skipped CartPole-v1 ppo2 length,cart_friction,masspole,masscart $rpe
compute_number_of_executions_skipped CartPole-v1 sac length,cart_friction,masspole,masscart $rpe
compute_number_of_executions_skipped CartPole-v1 dqn length,cart_friction,masspole,masscart $rpe
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2, link_mass_1 **************"
compute_number_of_executions_skipped Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $rpe
compute_number_of_executions_skipped Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $rpe
compute_number_of_executions_skipped Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $rpe
echo '----------------------------------------------------------------'

echo '==================================================================='

rpe=5
echo "++++++++++++++ 2 parameters, rpe = 5 ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_number_of_executions_skipped CartPole-v1 ppo2 length,cart_friction $rpe
compute_number_of_executions_skipped CartPole-v1 sac length,cart_friction $rpe
compute_number_of_executions_skipped CartPole-v1 dqn length,cart_friction $rpe
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_number_of_executions_skipped Pendulum-v0 dqn dt,length $rpe $training_time
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_number_of_executions_skipped MountainCar-v0 sac force,gravity $rpe $training_time
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_number_of_executions_skipped Acrobot-v1 dqn link_length_1,link_com_pos_1 $rpe $training_time
echo '----------------------------------------------------------------'

echo '==================================================================='


cd ..
#!/bin/bash

algos=('ppo2' 'sac' 'dqn')

function compute_volume {
  local env=$1
  local algo=$2
  local param_names=$3 # comma-separated values
  local granularity=$4
  local runs_prob_est=$5
  local smooth=$6
  local normalize_limits=$7
  local training_time=$8

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

  save_dir=~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo

  if [[ $normalize_limits == "true" ]]; then
    python analyze_volume_results.py --dir $save_dir \
    --algo_name $algo --grid_granularity_percentage_of_range $granularity --env_name $env \
    --plot_file_path ~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo \
    --param_names $param_names --smooth $smooth --plot_only_approximated --max_points_x 50 --max_points_y 50 \
    --regression_probability --normalize_limits --logging_level INFO
    mkdir -p $save_dir/normalize-limits
    mv $save_dir/analyze_volume_results_adapt_regress_probability_* $save_dir/normalize-limits
    mv $save_dir/heatmap_adaptation_probability_iteration_* $save_dir/normalize-limits
    mv $save_dir/heatmap_regression_probability_iteration_* $save_dir/normalize-limits
  else
    python analyze_volume_results.py --dir $save_dir \
    --algo_name $algo --grid_granularity_percentage_of_range $granularity --env_name $env \
    --plot_file_path ~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo \
    --param_names $param_names --smooth $smooth --plot_only_approximated --max_points_x 50 --max_points_y 50 \
    --regression_probability --logging_level INFO
    mkdir -p $save_dir/not-normalize-limits
    mv $save_dir/analyze_volume_results_adapt_regress_probability_* $save_dir/not-normalize-limits
    mv $save_dir/heatmap_adaptation_probability_iteration_* $save_dir/not-normalize-limits
    mv $save_dir/heatmap_regression_probability_iteration_* $save_dir/not-normalize-limits
  fi

}

# run with source ./analyze_volume_results.sh in order that the command conda is recognized
conda activate rl-plasticity-experiments

cd src || exit

#### VOLUME COMPUTATION FOR DISCRIMINATION ####

g=1.0
rpe=3
normalize_limits="true"

echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 ppo2 length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 ppo2 dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 ppo2 force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='


echo "++++++++++++++ 3 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction, masspole **************"
compute_volume CartPole-v1 ppo2 length,cart_friction,masspole $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction,masspole $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction,masspole $g $rpe 2.0$normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length, mass **************"
compute_volume Pendulum-v0 ppo2 dt,length,mass $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length,mass $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length,mass $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity, goal_velocity **************"
compute_volume MountainCar-v0 ppo2 force,gravity,goal_velocity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity,goal_velocity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity,goal_velocity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='

g=2.0 # to save time
echo "++++++++++++++ 4 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction, masspole, masscart **************"
compute_volume CartPole-v1 ppo2 length,cart_friction,masspole,masscart $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction,masspole,masscart $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction,masspole,masscart $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2, link_mass_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='






#### VOLUME COMPUTATION VARYING GRANULARITY ####

normalize_limits="false"
g=1.0

echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 ppo2 length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 ppo2 dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 ppo2 force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='

g=0.5

echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 ppo2 length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 ppo2 dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 ppo2 force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='

g=2.0

echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 ppo2 length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 ppo2 dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 ppo2 force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='







#### VOLUME COMPUTATION VARYING RPE ####

g=1.0
rpe=5
echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 ppo2 length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits
compute_volume CartPole-v1 dqn length,cart_friction $g $rpe 2.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 ppo2 dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 sac dt,length $g $rpe 1.0 $normalize_limits
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 ppo2 force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits
compute_volume MountainCar-v0 dqn force,gravity $g $rpe 0.01 $normalize_limits
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 ppo2 link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 sac link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits
echo '----------------------------------------------------------------'

echo '==================================================================='








#### VOLUME COMPUTATION WITH FULL TRAINING TIME ####

g=1.0
rpe=3
training_time="full"
echo "++++++++++++++ 2 parameters g = $g, rpe = $rpe, training_time = $training_time ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction **************"
compute_volume CartPole-v1 sac length,cart_friction $g $rpe 2.0 $normalize_limits $training_time
echo '----------------------------------------------------------------'

echo "************** Pendulum-v0 dt, length **************"
compute_volume Pendulum-v0 dqn dt,length $g $rpe 1.0 $normalize_limits $training_time
echo '----------------------------------------------------------------'

echo "************** MountainCar-v0 force, gravity **************"
compute_volume MountainCar-v0 sac force,gravity $g $rpe 0.01 $normalize_limits $training_time
echo '----------------------------------------------------------------'

echo "************** Acrobot-v1 link_length_1, link_com_pos_1 **************"
compute_volume Acrobot-v1 dqn link_length_1,link_com_pos_1 $g $rpe 1.0 $normalize_limits $training_time
echo '----------------------------------------------------------------'

echo '==================================================================='

cd ..
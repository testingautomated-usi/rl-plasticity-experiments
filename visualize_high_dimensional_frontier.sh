#!/bin/bash

algos=('ppo2' 'sac' 'dqn')

function plot_high_dimensional_frontier {
  local env=$1
  local algo=$2
  local param_names=$3 # comma-separated values
  local granularity=$4
  local perplexity=$5

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

  save_dir=~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo

  python analyze_volume_results.py --dir $save_dir \
  --algo_name $algo --grid_granularity_percentage_of_range $granularity --env_name $env \
  --plot_file_path ~/Desktop/rl-experiments-artifacts/cluster/$env/alphatest/$param_names_dir/$algo \
  --param_names $param_names --normalize_limits --perplexity $perplexity --only_tsne --logging_level INFO
  mkdir -p $save_dir/tsne_frontier_dir_p_$perplexity
  mv $save_dir/analyze_volume_results_tsne_* $save_dir/tsne_frontier_dir_p_$perplexity
  mv $save_dir/tsne_frontier_points_iteration_*_p_$perplexity* $save_dir/tsne_frontier_dir_p_$perplexity

}

# run with source ./get_min_max_num_iterations.sh in order that the command conda is recognized
conda activate stable-baselines

cd src || exit

#### VOLUME COMPUTATION FOR DISCRIMINATION ####

g=1.0
p=10

echo "++++++++++++++ 3 parameters g = $g ++++++++++++++"

echo "************** CartPole-v1 length, cart_friction, masspole **************"
plot_high_dimensional_frontier CartPole-v1 ppo2 length,cart_friction,masspole $g $p
plot_high_dimensional_frontier CartPole-v1 sac length,cart_friction,masspole $g $p
p=13
plot_high_dimensional_frontier CartPole-v1 dqn length,cart_friction,masspole $g $p
echo '----------------------------------------------------------------'

p=100
echo "************** Pendulum-v0 dt, length, mass **************"
plot_high_dimensional_frontier Pendulum-v0 ppo2 dt,length,mass $g $p
plot_high_dimensional_frontier Pendulum-v0 sac dt,length,mass $g $p
p=150
plot_high_dimensional_frontier Pendulum-v0 dqn dt,length,mass $g $p
echo '----------------------------------------------------------------'

p=20
echo "************** MountainCar-v0 force, gravity, goal_velocity **************"
plot_high_dimensional_frontier MountainCar-v0 ppo2 force,gravity,goal_velocity $g $p
p=50
plot_high_dimensional_frontier MountainCar-v0 sac force,gravity,goal_velocity $g $p
plot_high_dimensional_frontier MountainCar-v0 dqn force,gravity,goal_velocity $g $p
echo '----------------------------------------------------------------'

p=20
echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2 **************"
plot_high_dimensional_frontier Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2 $g $p
plot_high_dimensional_frontier Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2 $g $p
plot_high_dimensional_frontier Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2 $g $p
echo '----------------------------------------------------------------'

echo '==================================================================='

echo "++++++++++++++ 4 parameters g = $g ++++++++++++++"

p=15
echo "************** CartPole-v1 length, cart_friction, masspole, masscart **************"
plot_high_dimensional_frontier CartPole-v1 ppo2 length,cart_friction,masspole,masscart $g $p
p=20
plot_high_dimensional_frontier CartPole-v1 sac length,cart_friction,masspole,masscart $g $p
p=30
plot_high_dimensional_frontier CartPole-v1 dqn length,cart_friction,masspole,masscart $g $p
echo '----------------------------------------------------------------'

p=50
echo "************** Acrobot-v1 link_length_1, link_com_pos_1, link_mass_2, link_mass_1 **************"
plot_high_dimensional_frontier Acrobot-v1 ppo2 link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $p
p=80
plot_high_dimensional_frontier Acrobot-v1 sac link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $p
p=120
plot_high_dimensional_frontier Acrobot-v1 dqn link_length_1,link_com_pos_1,link_mass_2,link_mass_1 $g $p
echo '----------------------------------------------------------------'

echo '==================================================================='

cd ..
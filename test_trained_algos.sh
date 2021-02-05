#!/bin/bash

function execute_test {
  local env_name=$1
  local algo_name=$2

  tb_log_name=$(echo "$env_name" | cut -d'-' -f 1 | awk '{print tolower($0)}')
  if [[ "$algo_name" == "sac" ]]; then
      python main.py --mode test --env_name "$env_name" --algo_name "$algo_name" \
        --tb_log_name "$tb_log_name" --n_eval_episodes 100
  else
    python main.py --mode test --env_name "$env_name" --algo_name "$algo_name" \
      --discrete_action_space True --tb_log_name "$tb_log_name" --n_eval_episodes 100
  fi

}

# run with source ./test_trained_algos.sh in order that the command conda is recognized
conda activate stable-baselines

cd src || exit

env_name=CartPole-v1
env_names=('CartPole-v1' 'Pendulum-v0' 'MountainCar-v0' 'Acrobot-v1')
#env_names=('CartPole-v1')
algo_names=('ppo2' 'sac' 'dqn')
#algo_names=('ppo2')
for env_name in "${env_names[@]}"; do
  echo "****** Start testing algos for environment: $env_name ******"
  for algo_name in "${algo_names[@]}"; do
    echo "++ Algo: $algo_name ++"
    execute_test "$env_name" "$algo_name"
    echo
  done
  echo "****** End testing algos for environment: $env_name ******"
  echo
done

cd ..

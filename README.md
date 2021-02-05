# Testing the Plasticity of Reinforcement Learning Based Systems
Source code and data for the paper "Testing the Plasticity of Reinforcement Learning Based Systems"

## Experiments artifacts
Experiments artifacts are available for download [at this link](https://drive.google.com/file/d/1lMZ-GOMh-qgBCnfaxfDHPfJpmM2D6q3N/view?usp=sharing).

## 1. Installation

To install the dependencies to run this project (we currently support MacOS and Ubuntu):
1. Download and install [Anaconda](https://www.anaconda.com/)
2. Create directory workspace in your home directory and move there: `mkdir ~/workspace && cd ~/workspace`
3. Clone this repository: `git clone https://github.com/testingautomated-usi/rl-plasticity-experiments`
4. Create the environment with the python dependencies: `conda env create -f rl-plasticity-experiments.yml`

## 2. Run an experiment

To run an experiment with AlphaTest type the following commands:

1. Move to the source code directory `cd ~/workspace/rl-plasticity-experiments`
2. Activate the environment: `conda activate rl-plasticity-experiments`
3. Run the experiment:
``` sh
python experiments.py --algo_name ppo2 \
  --env_name CartPole-v1 \
  --num_iterations 13 \
  --param_names length,cart_friction \
  --runs_for_probability_estimation 3 \
  --num_search_iterations 1 \
  --logging_level INFO \
  --search_type alphatest
```
          
The previous command...


## 4. Experiments artifacts
Experiments artifacts are available for download [at this link](https://drive.google.com/file/d/1lMZ-GOMh-qgBCnfaxfDHPfJpmM2D6q3N/view?usp=sharing).

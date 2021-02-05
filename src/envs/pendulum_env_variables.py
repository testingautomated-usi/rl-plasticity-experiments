import json
from typing import Dict, List

import numpy as np

from envs.env_variables import EnvVariables, load_env_params
from param import Param


class PendulumEnvVariables(EnvVariables):
    def __init__(
        self,
        algo_name,
        dt=None,
        mass=None,
        length=None,
        discrete_action_space: bool = False,
        param_names: List[str] = None,
        model_suffix: str = None,
    ):
        # FIXME may be not that efficient loading from disk every time the object is instantiated
        params_mass = load_env_params(algo_name=algo_name, env_name="pendulum", param_name="mass", model_suffix=model_suffix)
        params_length = load_env_params(
            algo_name=algo_name, env_name="pendulum", param_name="length", model_suffix=model_suffix
        )
        params_dt = load_env_params(algo_name=algo_name, env_name="pendulum", param_name="dt", model_suffix=model_suffix)

        self.mass = Param(**params_mass, current_value=mass, id=0, name="mass")
        self.length = Param(**params_length, current_value=length, id=1, name="length")
        self.dt = Param(**params_dt, current_value=dt, id=2, name="dt")

        if param_names:
            self.params = []
            for param_name in param_names:
                if self.mass.get_name() == param_name:
                    self.params.append(self.mass)
                elif self.length.get_name() == param_name:
                    self.params.append(self.length)
                elif self.dt.get_name() == param_name:
                    self.params.append(self.dt)

            assert len(self.params) > 1, "num of params should be at least 2: {}".format(len(self.params))
        else:
            self.params = [self.mass, self.length, self.dt]

        self.discrete_action_space = discrete_action_space
        self.check_range()

    def instantiate_env(self) -> Dict:
        result = dict()
        result["discrete_action_space"] = self.discrete_action_space
        result["mass"] = self.mass.get_current_value()
        result["length"] = self.length.get_current_value()
        result["dt"] = self.dt.get_current_value()

        return result

    def get_params(self) -> List[Param]:
        return self.params

    def get_algo_related_params(self) -> Dict:
        return dict()

    def get_params_string(self) -> str:
        result = dict()
        result["mass"] = self.mass.get_current_value()
        result["length"] = self.length.get_current_value()
        result["dt"] = self.dt.get_current_value()

        return json.dumps(result)

    def get_lower_limit_reward(self) -> float:
        raise -np.inf

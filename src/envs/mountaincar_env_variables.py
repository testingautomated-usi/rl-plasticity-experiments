import json
from typing import Dict, List

from envs.env_variables import EnvVariables, load_env_params
from param import Param


class MountainCarEnvVariables(EnvVariables):
    def __init__(
        self,
        algo_name,
        discrete_action_space: bool = False,
        goal_velocity: float = 0.0,
        force: float = 0.001,
        gravity: float = 0.0025,
        param_names: List[str] = None,
        model_suffix: str = None,
    ):
        # FIXME may be not that efficient loading from disk every time the object is instantiated
        param_goal_velocity = load_env_params(
            algo_name=algo_name, env_name="mountaincar", param_name="goal_velocity", model_suffix=model_suffix
        )
        params_force = load_env_params(
            algo_name=algo_name, env_name="mountaincar", param_name="force", model_suffix=model_suffix
        )
        params_gravity = load_env_params(
            algo_name=algo_name, env_name="mountaincar", param_name="gravity", model_suffix=model_suffix
        )

        self.goal_velocity = Param(**param_goal_velocity, current_value=goal_velocity, id=0, name="goal_velocity")
        self.force = Param(**params_force, current_value=force, id=1, name="force")
        self.gravity = Param(**params_gravity, current_value=gravity, id=2, name="gravity")

        self.algo_name = algo_name

        if param_names:
            self.params = []
            for param_name in param_names:
                if self.goal_velocity.get_name() == param_name:
                    self.params.append(self.goal_velocity)
                elif self.force.get_name() == param_name:
                    self.params.append(self.force)
                elif self.gravity.get_name() == param_name:
                    self.params.append(self.gravity)

            assert len(self.params) > 1, "num of params should be at least 2: {}".format(len(self.params))
        else:
            self.params = [self.goal_velocity, self.force, self.gravity]

        self.discrete_action_space = discrete_action_space
        self.check_range()

    def instantiate_env(self) -> Dict:
        result = dict()
        result["goal_velocity"] = self.goal_velocity.get_current_value()
        result["force"] = self.force.get_current_value()
        result["gravity"] = self.gravity.get_current_value()
        result["discrete_action_space"] = self.discrete_action_space
        return result

    def get_params(self) -> List[Param]:
        return self.params

    def get_algo_related_params(self) -> Dict:
        return dict()

    def get_params_string(self) -> str:
        result = dict()
        result["goal_velocity"] = self.goal_velocity.get_current_value()
        result["force"] = self.force.get_current_value()
        result["gravity"] = self.gravity.get_current_value()

        return json.dumps(result)

    def get_lower_limit_reward(self) -> float:
        if self.algo_name != "sac":
            return -200.0
        return -999.0

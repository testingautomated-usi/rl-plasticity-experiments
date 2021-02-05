import json
from typing import Dict, List

from envs.env_variables import EnvVariables, load_env_params
from param import Param


class CartPoleEnvVariables(EnvVariables):
    def __init__(
        self,
        algo_name,
        masscart=None,
        masspole=None,
        length=None,
        cart_friction=None,
        discrete_action_space: bool = False,
        param_names: List[str] = None,
        model_suffix: str = None,
    ):
        # FIXME may be not that efficient loading from disk every time the object is instantiated
        params_masscart = load_env_params(
            algo_name=algo_name, env_name="cartpole", param_name="masscart", model_suffix=model_suffix
        )
        params_masspole = load_env_params(
            algo_name=algo_name, env_name="cartpole", param_name="masspole", model_suffix=model_suffix
        )
        params_length = load_env_params(
            algo_name=algo_name, env_name="cartpole", param_name="length", model_suffix=model_suffix
        )
        params_cart_friction = load_env_params(
            algo_name=algo_name, env_name="cartpole", param_name="cart_friction", model_suffix=model_suffix
        )

        self.masscart = Param(**params_masscart, current_value=masscart, id=0, name="masscart")
        self.masspole = Param(**params_masspole, current_value=masspole, id=1, name="masspole")
        self.length = Param(**params_length, current_value=length, id=2, name="length")
        self.cart_friction = Param(**params_cart_friction, current_value=cart_friction, id=3, name="cart_friction")

        if param_names:
            self.params = []
            for param_name in param_names:
                if self.masscart.get_name() == param_name:
                    self.params.append(self.masscart)
                elif self.masspole.get_name() == param_name:
                    self.params.append(self.masspole)
                elif self.length.get_name() == param_name:
                    self.params.append(self.length)
                elif self.cart_friction.get_name() == param_name:
                    self.params.append(self.cart_friction)

            assert len(self.params) > 1, "num of params should be at least 2: {}".format(len(self.params))
        else:
            self.params = [self.masscart, self.masspole, self.length, self.cart_friction]

        self.discrete_action_space = discrete_action_space
        self.check_range()

    def instantiate_env(self) -> Dict:
        result = dict()
        result["discrete_action_space"] = self.discrete_action_space
        result["masscart"] = self.masscart.get_current_value()
        result["masspole"] = self.masspole.get_current_value()
        result["length"] = self.length.get_current_value()
        result["cart_friction"] = self.cart_friction.get_current_value()

        return result

    def get_params(self) -> List[Param]:
        return self.params

    def get_algo_related_params(self) -> Dict:
        return dict()

    def get_params_string(self) -> str:
        result = dict()
        result["masscart"] = self.masscart.get_current_value()
        result["masspole"] = self.masspole.get_current_value()
        result["length"] = self.length.get_current_value()
        result["cart_friction"] = self.cart_friction.get_current_value()

        return json.dumps(result)

    def get_lower_limit_reward(self) -> float:
        return 0.0

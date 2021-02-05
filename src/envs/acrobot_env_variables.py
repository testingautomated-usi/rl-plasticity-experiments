from typing import Dict, List
from envs.env_variables import EnvVariables, load_env_params
from param import Param
import json


class AcrobotEnvVariables(EnvVariables):

    def __init__(self, algo_name,
                 discrete_action_space: bool = False,
                 link_length_1: float = 1.0,
                 link_mass_1: float = 1.0,
                 link_mass_2: float = 1.0,
                 link_com_pos_1: float = 0.5,
                 link_com_pos_2: float = 0.5,
                 link_moi: float = 1.0,
                 param_names: List[str] = None,
                 model_suffix: str = None):
        # FIXME may be not that efficient loading from disk every time the object is instantiated
        params_link_length_1 = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_length_1', model_suffix=model_suffix)
        params_link_mass_1 = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_mass_1', model_suffix=model_suffix)
        params_link_mass_2 = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_mass_2', model_suffix=model_suffix)
        params_link_com_pos_1 = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_com_pos_1', model_suffix=model_suffix)
        params_link_com_pos_2 = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_com_pos_2', model_suffix=model_suffix)
        params_link_moi = load_env_params(algo_name=algo_name, env_name='acrobot', param_name='link_moi', model_suffix=model_suffix)

        self.link_length_1 = Param(**params_link_length_1, current_value=link_length_1, id=0, name='link_length_1')
        self.link_mass_1 = Param(**params_link_mass_1, current_value=link_mass_1, id=1, name='link_mass_1')
        self.link_mass_2 = Param(**params_link_mass_2, current_value=link_mass_2, id=2, name='link_mass_2')
        self.link_com_pos_1 = Param(**params_link_com_pos_1, current_value=link_com_pos_1, id=3, name='link_com_pos_1')
        self.link_com_pos_2 = Param(**params_link_com_pos_2, current_value=link_com_pos_2, id=4, name='link_com_pos_2')
        self.link_moi = Param(**params_link_moi, current_value=link_moi, id=5, name='link_moi')

        self.algo_name = algo_name

        if param_names:
            self.params = []
            for param_name in param_names:
                if self.link_length_1.get_name() == param_name:
                    self.params.append(self.link_length_1)
                elif self.link_mass_1.get_name() == param_name:
                    self.params.append(self.link_mass_1)
                elif self.link_mass_2.get_name() == param_name:
                    self.params.append(self.link_mass_2)
                elif self.link_com_pos_1.get_name() == param_name:
                    self.params.append(self.link_com_pos_1)
                elif self.link_com_pos_2.get_name() == param_name:
                    self.params.append(self.link_com_pos_2)
                elif self.link_moi.get_name() == param_name:
                    self.params.append(self.link_moi)

            assert len(self.params) > 1, 'num of params should be at least 2: {}'.format(len(self.params))
        else:
            self.params = [self.link_length_1, self.link_mass_1, self.link_mass_2,
                           self.link_com_pos_1, self.link_com_pos_2, self.link_moi]

        self.discrete_action_space = discrete_action_space
        self.check_range()

    def instantiate_env(self) -> Dict:
        result = dict()
        result["link_length_1"] = self.link_length_1.get_current_value()
        result["link_mass_1"] = self.link_mass_1.get_current_value()
        result["link_mass_2"] = self.link_mass_2.get_current_value()
        result["link_com_pos_1"] = self.link_com_pos_1.get_current_value()
        result["link_com_pos_2"] = self.link_com_pos_2.get_current_value()
        result["link_moi"] = self.link_moi.get_current_value()
        result["discrete_action_space"] = self.discrete_action_space
        return result

    def get_params(self) -> List[Param]:
        return self.params

    def get_algo_related_params(self) -> Dict:
        return dict()

    def get_params_string(self) -> str:
        result = dict()
        result["link_length_1"] = self.link_length_1.get_current_value()
        result["link_mass_1"] = self.link_mass_1.get_current_value()
        result["link_mass_2"] = self.link_mass_2.get_current_value()
        result["link_com_pos_1"] = self.link_com_pos_1.get_current_value()
        result["link_com_pos_2"] = self.link_com_pos_2.get_current_value()
        result["link_moi"] = self.link_moi.get_current_value()

        return json.dumps(result)

    def get_lower_limit_reward(self) -> float:
        return -500.0

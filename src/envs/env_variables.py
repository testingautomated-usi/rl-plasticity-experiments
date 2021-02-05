import math
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import yaml

from algo.env_exec_details import EnvExecDetails
from log import Log
from param import Param
# I DID NOT CREATE AN INIT METHOD FOR INSTANTIATING LOG BECAUSE THIS OBJECT IS DEEP_COPIED DURING SEARCH
# AND DEEP_COPY FAILS IF THERE IS AN INSTANCE OF THE LOGGER OBJECT TO BE COPIED
from utilities import CONTINUAL_LEARNING_RANGE_MULTIPLIER, HOME

logger = Log("EnvVariables")


def load_env_params(algo_name=None, env_name=None, param_name=None, model_suffix=None):
    # Load parameters from yaml file
    abs_params_dir = os.path.abspath(HOME + "/env_params")
    filename = (
        abs_params_dir + "/{}/{}.yml".format(env_name, algo_name)
        if not model_suffix
        else abs_params_dir + "/{}/{}_{}.yml".format(env_name, algo_name, model_suffix)
    )
    with open(filename, "r") as f:
        params_dict = yaml.safe_load(f)
        if param_name in list(params_dict.keys()):
            return params_dict[param_name]
        else:
            raise ValueError("Parameters not found for {}-{}".format(env_name, param_name))


class EnvVariables(ABC):
    @abstractmethod
    def instantiate_env(self) -> Dict:
        pass

    @abstractmethod
    def get_algo_related_params(self) -> Dict:
        pass

    @abstractmethod
    def get_params(self) -> List[Param]:
        pass

    @abstractmethod
    def get_params_string(self) -> str:
        pass

    @abstractmethod
    def get_lower_limit_reward(self) -> float:
        pass

    def check_range(self) -> None:
        for param in self.get_params():
            value = param.get_default_value() if param.get_default_value() != 0 else param.get_starting_value_if_zero()
            if param.get_direction() == "positive" and param.get_starting_multiplier() > 1:
                expected_limit = param.get_starting_multiplier() * value * CONTINUAL_LEARNING_RANGE_MULTIPLIER
                assert math.isclose(
                    expected_limit, param.get_high_limit(), abs_tol=1e-3
                ), "Param {} does not respect high limit that should be {} but is {}".format(
                    param.get_name(), expected_limit, param.get_high_limit()
                )
            elif param.get_direction() == "positive" and 0 < param.get_starting_multiplier() < 1:
                expected_limit = param.get_starting_multiplier() * value / CONTINUAL_LEARNING_RANGE_MULTIPLIER
                assert math.isclose(
                    expected_limit, param.get_low_limit(), abs_tol=1e-3
                ), "Param {} does not respect low limit that should be {} but is {}".format(
                    param.get_name(), expected_limit, param.get_low_limit()
                )
            elif param.get_direction() == "negative" and param.get_starting_multiplier() > 1:
                expected_limit = param.get_starting_multiplier() * value * CONTINUAL_LEARNING_RANGE_MULTIPLIER
                assert math.isclose(
                    expected_limit, param.get_low_limit(), abs_tol=1e-3
                ), "Param {} does not respect low limit that should be {} but is {}".format(
                    param.get_name(), expected_limit, param.get_low_limit()
                )
            elif param.get_direction() == "negative" and 0 < param.get_starting_multiplier() < 1:
                expected_limit = param.get_starting_multiplier() * value / CONTINUAL_LEARNING_RANGE_MULTIPLIER
                assert math.isclose(
                    expected_limit, param.get_high_limit(), abs_tol=1e-3
                ), "Param {} does not respect low limit that should be {} but is {}".format(
                    param.get_name(), expected_limit, param.get_high_limit()
                )

    def get_values(self) -> List:
        values = []
        for param in self.get_params():
            values.append(param.get_current_value())
        return values

    def mutate_params(self, candidates: List[EnvExecDetails], exp_search_guidance: bool = False) -> int:
        if exp_search_guidance:
            per_drops = [candidate.get_per_drop() for candidate in candidates]
            index = random.choices(population=range(len(candidates)), weights=per_drops, k=1)[0]
        else:
            index = random.choices(population=range(len(candidates)), k=1)[0]

        env_values_to_run = candidates[index].get_env_values()
        for i, value in enumerate(env_values_to_run):
            self.set_param(index=i, new_value=value)
        return index

    def mutate_params_randomly(self) -> bool:
        num_params_to_mutate = random.randint(1, len(self.get_params()))
        indices_used = []
        for i in range(num_params_to_mutate):
            while True:
                index = random.randint(0, len(self.get_params()) - 1)
                if index not in indices_used:
                    break
            indices_used.append(index)
            param = self.get_params()[index]
            logger.debug("Mutating param {}".format(param.get_name()))
            param.mutate_random()
        return True

    def mutate_param(self, exp_search_guidance: bool = False, halve_or_double: str = None) -> bool:
        percentage_drops = self.get_percentage_drops()
        percentage_drops_greater_than_zero = True
        for percentage_drop in percentage_drops:
            if percentage_drop == 0.0:
                percentage_drops_greater_than_zero = False
                break

        if exp_search_guidance:
            assert percentage_drops_greater_than_zero, "percentage_drops_greater_than_zero should have a value: {}".format(
                percentage_drops_greater_than_zero
            )

        if percentage_drops_greater_than_zero and exp_search_guidance:
            index = random.choices(population=range(len(self.get_params())), weights=percentage_drops)[0]
        else:
            index = random.randint(0, len(self.get_params()) - 1)
        param = self.get_params()[index]
        logger.debug("Mutating param {}".format(param.get_name()))
        return param.mutate(halve_or_double=halve_or_double)

    def get_percentage_drops(self) -> List[float]:
        result = []
        for param in self.get_params():
            result.append(param.get_percentage_drop())
        return result

    def get_param(self, index: int) -> Param:
        assert 0 <= index <= len(self.get_params()) - 1, "index [0, {}]: {}".format(len(self.get_params()) - 1, index)
        param = self.get_params()[index]
        # logger.debug("get_param param: {}".format(param.get_name()))
        return param

    def set_param(self, index: int, new_value, do_not_check_for_limits: bool = False) -> bool:
        assert 0 <= index <= len(self.get_params()) - 1, "index [0, {}]: {}".format(len(self.get_params()) - 1, index)
        param = self.get_params()[index]
        if not do_not_check_for_limits:
            # logger.debug("set_param param {} to value {}".format(param.get_name(), new_value))
            if param.get_low_limit() <= new_value <= param.get_high_limit():
                param.current_value = new_value
                return True
            raise ValueError(
                "new_value {} for param {} above limits [{}, {}]".format(
                    new_value, param.get_name(), param.get_low_limit(), param.get_high_limit()
                )
            )
        else:
            if param.get_low_limit() <= new_value <= param.get_high_limit():
                pass
            else:
                logger.warn("Setting value {} that is beyond limits of param {}".format(new_value, param.get_name()))
            param.current_value = new_value

    def is_equal(self, other) -> bool:
        if not isinstance(other, EnvVariables):
            # don't attempt to compare against unrelated types
            return NotImplemented

        params = self.get_params()
        other_params = other.get_params()
        for param in params:
            other_params_with_same_id = list(filter(lambda other_param: other_param.id == param.id, other_params))
            assert len(other_params_with_same_id) == 1, "num other params with same id == 1: {}".format(
                len(other_params_with_same_id)
            )
            if other_params_with_same_id[0] != param:
                return False
        return True

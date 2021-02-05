import copy
from typing import List


class EnvExecDetails:
    def __init__(
        self,
        env_values: List,
        executed: bool,
        per_drop: float,
        env_id: int,
        init_env_variables,
        predicate: bool,
        pass_probability: float,
        regression_probability: float,
    ):
        self.env_values = env_values
        self.executed = executed
        self.per_drop = per_drop
        self.env_id = env_id
        self.init_env_variables = copy.deepcopy(init_env_variables)
        self.predicate = predicate
        self.pass_probability = pass_probability
        self.regression_probability = regression_probability

    def is_executed(self) -> bool:
        return self.executed

    def get_env_values(self) -> List:
        return self.env_values

    def get_per_drop(self) -> float:
        return self.per_drop

    def set_per_drop(self, per_drop: float) -> None:
        self.per_drop = per_drop

    def set_executed(self, executed: bool) -> None:
        self.executed = executed

    def set_predicate(self, predicate: bool) -> None:
        self.predicate = predicate

    def is_predicate(self) -> bool:
        return self.predicate

    def set_regression_probability(self, regression_probability: float) -> None:
        self.regression_probability = regression_probability

    def get_regression_probability(self) -> float:
        assert self.regression_probability != -1.0, "regression probability not set"
        return self.regression_probability

    def set_pass_probability(self, pass_probability: float) -> None:
        self.pass_probability = pass_probability

    def get_pass_probability(self) -> float:
        assert self.pass_probability != -1.0, "pass probability not set"
        return self.pass_probability

    def __str__(self):
        return f"({self.env_values}, {self.executed}, {self.per_drop})"

    def __eq__(self, other):
        if not isinstance(other, EnvExecDetails):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.env_id == other.env_id

    def dominates(self, other) -> bool:
        if not isinstance(other, EnvExecDetails):
            # don't attempt to compare against unrelated types
            return NotImplemented

        for i in range(len(self.env_values)):
            direction = self.init_env_variables.get_param(i).get_direction()
            starting_multiplier = self.init_env_variables.get_param(i).get_starting_multiplier()
            assert direction == "positive" or direction == "negative", "unknown direction is not supported"
            env_value = self.env_values[i]
            other_env_value = other.env_values[i]
            if (direction == "positive" and starting_multiplier > 1.0) or (
                direction == "negative" and starting_multiplier < 1.0
            ):
                if env_value < other_env_value:
                    return False
            elif (direction == "positive" and starting_multiplier < 1.0) or (
                direction == "negative" and starting_multiplier > 1.0
            ):
                if env_value > other_env_value:
                    return False

        return True

import copy
import itertools
from typing import Dict, List, Union

import numpy as np

from algo.env_exec_details import EnvExecDetails
from envs.env_variables import EnvVariables
from log import Log


class EnvPredicatePair:
    def __init__(
        self,
        env_variables: EnvVariables,
        predicate: bool = False,
        regression: bool = True,
        execution_info: Dict = None,
        probability_estimation_runs: List[bool] = None,
        regression_estimation_runs: List[bool] = None,
        pass_probability: float = None,
        regression_probability: float = None,
        model_dirs: List[str] = None,
    ):
        self.env_variables = copy.deepcopy(env_variables)
        self.predicate = predicate
        self.regression = regression
        if not execution_info:
            execution_info = {}
        self.execution_info = execution_info
        self.probability_estimation_runs = probability_estimation_runs
        self.regression_estimation_runs = regression_estimation_runs
        self.pass_probability = pass_probability
        self.regression_probability = regression_probability
        self.model_dirs = model_dirs

    def is_predicate(self) -> bool:
        # majority wins
        if self.probability_estimation_runs:
            assert (
                len(self.probability_estimation_runs) % 2 != 0
            ), "num probability_estimation_runs should be divisible by 2: {}".format(len(self.probability_estimation_runs))
            pass_probability = self.compute_pass_probability()
            return pass_probability > 0.5
        return self.predicate

    def is_regression(self) -> Union[bool, None]:
        if self.regression_estimation_runs:
            regression_probability = self.compute_regression_probability()
            if len(self.regression_estimation_runs) % 2 != 0:
                return 1 - regression_probability > 0.5
            return None
        return self.regression

    def get_env_variables(self) -> EnvVariables:
        return self.env_variables

    def get_model_dirs(self) -> List[str]:
        return self.model_dirs

    def set_model_dirs(self, model_dirs: List[str]) -> None:
        self.model_dirs = model_dirs

    def get_execution_info(self) -> Dict:
        return self.execution_info

    def get_probability_estimation_runs(self) -> List[bool]:
        return self.probability_estimation_runs

    def get_regression_estimation_runs(self) -> List[bool]:
        return self.regression_estimation_runs

    def get_pass_probability(self) -> float:
        return self.pass_probability

    def get_regression_probability(self) -> float:
        return self.regression_probability

    def compute_regression_probability(self) -> float:
        assert (
            self.regression_estimation_runs or self.regression_probability
        ), "either regression_estimation_runs or regression_probability should have a value: {}, {}. Env {}".format(
            self.regression_estimation_runs, self.regression_probability, self.env_variables.get_params_string()
        )
        count = 0
        for regression_run in self.regression_estimation_runs:
            if regression_run:
                count += 1
        if len(self.regression_estimation_runs) > 0:
            return round(count / len(self.regression_estimation_runs), 2)
        return self.regression_probability

    def compute_pass_probability(self) -> float:
        assert (
            self.probability_estimation_runs or self.pass_probability
        ), "either probability_estimation_runs or pass_probability should have a value: {}, {}. Env {}".format(
            self.probability_estimation_runs, self.pass_probability, self.env_variables.get_params_string()
        )
        if self.probability_estimation_runs:
            count = 0
            for probability_run in self.probability_estimation_runs:
                if probability_run:
                    count += 1
            return round(count / len(self.probability_estimation_runs), 2)
        return self.pass_probability


class BufferItem:
    def __init__(
        self,
        env_values: Dict,
        pass_probability: float,
        predicate: bool,
        regression_probability: float,
        probability_estimation_runs: List[bool],
        regression_estimation_runs: List[bool],
        model_dirs: List[str],
    ):
        self.env_values = env_values
        self.pass_probability = pass_probability
        self.predicate = predicate
        self.regression_probability = regression_probability
        self.probability_estimation_runs = probability_estimation_runs
        self.regression_estimation_runs = regression_estimation_runs
        self.model_dirs = model_dirs

    def get_env_values(self) -> Dict:
        return self.env_values

    def get_pass_probability(self) -> float:
        return self.pass_probability

    def is_predicate(self) -> bool:
        return self.predicate

    def get_regression_probability(self) -> float:
        return self.regression_probability

    def get_model_dirs(self) -> List[str]:
        return self.model_dirs

    def get_probability_estimation_runs(self) -> List[bool]:
        return self.probability_estimation_runs

    def get_regression_estimation_runs(self) -> List[bool]:
        return self.regression_estimation_runs


def read_saved_buffer(buffer_file: str) -> List[BufferItem]:
    result = []
    with open(buffer_file, "r", encoding="utf-8") as f:
        for line in f.readlines():

            # deal with '-' sign in the dictionary containing values which is the same as the separator
            index_open_curly_brace = line.index("{")
            index_closed_curly_brace = line.index("}")
            env_values = eval(line[index_open_curly_brace : index_closed_curly_brace + 1])
            line_excluded_dict_values = line[index_closed_curly_brace + 2 :]
            items = line_excluded_dict_values.split("-")
            pass_probability = eval(items[1])
            predicate = eval(items[2])
            if len(items) > 3:
                regression_probability = eval(items[3])
            else:
                regression_probability = 1.0

            if len(items) > 4:
                probability_estimation_runs = eval(items[4])
            else:
                probability_estimation_runs = []

            if len(items) > 5:
                regression_estimation_runs = eval(items[5])
            else:
                regression_estimation_runs = []

            if len(items) > 6:
                model_dirs = eval(items[6])
            else:
                model_dirs = []

            result.append(
                BufferItem(
                    env_values=env_values,
                    pass_probability=pass_probability,
                    predicate=predicate,
                    regression_probability=regression_probability,
                    probability_estimation_runs=probability_estimation_runs,
                    regression_estimation_runs=regression_estimation_runs,
                    model_dirs=model_dirs,
                )
            )

    return result


def _find_closest_env(possible_envs_dict: Dict, env_to_search: EnvVariables) -> EnvExecDetails:
    min_distance = np.inf
    closest_env = None
    for possible_env in list(itertools.chain(*possible_envs_dict.values())):
        dist = np.linalg.norm(np.asarray(possible_env.get_env_values()) - np.asarray(env_to_search.get_values()))
        if dist < min_distance:
            closest_env = copy.deepcopy(possible_env)
            min_distance = dist
    return closest_env


class BufferEnvPredicatePairs:
    def __init__(self, save_dir: str = None):
        assert save_dir, "save_dir should have a value: {}".format(save_dir)
        self.save_dir = save_dir
        self.env_predicate_pairs: List[EnvPredicatePair] = []
        self.logger = Log("BufferEnvPredicatePairs")

    def append(self, env_predicate_pair: EnvPredicatePair):
        if self.is_already_evaluated(env_predicate_pair.get_env_variables()):
            raise BufferError("Env {} already evaluated".format(env_predicate_pair.get_env_variables().get_params_string()))
        self.logger.debug(
            "Adding env {} that evaluates to {}".format(
                env_predicate_pair.get_env_variables().get_params_string(), env_predicate_pair.is_predicate(),
            )
        )
        self.env_predicate_pairs.append(env_predicate_pair)

    def is_already_evaluated(self, candidate_env_variables: EnvVariables) -> bool:
        for env_predicate_pair in self.env_predicate_pairs:
            evaluated_env_variables = env_predicate_pair.get_env_variables()
            if evaluated_env_variables.is_equal(candidate_env_variables):
                self.logger.debug("Env {} was already evaluated".format(candidate_env_variables.get_params_string()))
                return True
        return False

    def get_predicate_of_evaluated_env(self, evaluated_env: EnvVariables) -> bool:
        for env_predicate_pair in self.env_predicate_pairs:
            evaluated_env_variables = env_predicate_pair.get_env_variables()
            if evaluated_env_variables.is_equal(evaluated_env):
                return env_predicate_pair.is_predicate()
        raise AttributeError("{} must be evaluated".format(evaluated_env.get_params_string()))

    def dominance_analysis(
        self, candidate_env_variables: EnvVariables, predicate_to_consider: bool = True
    ) -> Union[EnvPredicatePair, None]:
        assert not self.is_already_evaluated(
            candidate_env_variables=candidate_env_variables
        ), "Env {} must not be evaluated".format(candidate_env_variables.get_params_string())

        executed_env_dominate = None
        if predicate_to_consider:
            # searching for an executed env that evaluates to True that dominates the env passed as parameter
            for env_predicate_pair in self.env_predicate_pairs:
                predicate = env_predicate_pair.is_predicate()
                if predicate:
                    dominates = True
                    for i in range(len(env_predicate_pair.get_env_variables().get_params())):
                        direction = env_predicate_pair.get_env_variables().get_param(index=i).get_direction()
                        starting_multiplier = (
                            env_predicate_pair.get_env_variables().get_param(index=i).get_starting_multiplier()
                        )
                        assert direction == "positive", "unknown and negative direction is not supported"
                        env_value = env_predicate_pair.get_env_variables().get_param(index=i).get_current_value()
                        other_env_value = candidate_env_variables.get_param(index=i).get_current_value()
                        if direction == "positive" and starting_multiplier > 1.0:
                            if env_value < other_env_value:
                                dominates = False
                        elif direction == "positive" and starting_multiplier < 1.0:
                            if env_value > other_env_value:
                                dominates = False
                    if dominates:
                        executed_env_dominate = env_predicate_pair
                        self.logger.debug(
                            "candidate {} dominated by executed env {} that evaluates to {}".format(
                                candidate_env_variables.get_params_string(),
                                env_predicate_pair.get_env_variables().get_params_string(),
                                predicate,
                            )
                        )
        else:
            # searching for an executed env that evaluates to False that is dominated by the env passed as parameter
            for env_predicate_pair in self.env_predicate_pairs:
                predicate = env_predicate_pair.is_predicate()
                if not predicate:
                    is_dominated = True
                    for i in range(len(env_predicate_pair.get_env_variables().get_params())):
                        direction = env_predicate_pair.get_env_variables().get_param(index=i).get_direction()
                        starting_multiplier = (
                            env_predicate_pair.get_env_variables().get_param(index=i).get_starting_multiplier()
                        )
                        assert direction == "positive", "unknown and negative direction is not supported"
                        env_value = env_predicate_pair.get_env_variables().get_param(index=i).get_current_value()
                        other_env_value = candidate_env_variables.get_param(index=i).get_current_value()
                        if direction == "positive" and starting_multiplier > 1.0:
                            if other_env_value < env_value:
                                is_dominated = False
                        elif direction == "positive" and starting_multiplier < 1.0:
                            if other_env_value > env_value:
                                is_dominated = False
                    if is_dominated:
                        executed_env_dominate = env_predicate_pair
                        self.logger.debug(
                            "candidate {} dominates executed env {} that evaluates to {}".format(
                                candidate_env_variables.get_params_string(),
                                env_predicate_pair.get_env_variables().get_params_string(),
                                not predicate,
                            )
                        )

        return executed_env_dominate

    def merge_with_other_buffer(self, other_buffer) -> None:
        if isinstance(other_buffer, BufferEnvPredicatePairs):
            for env_predicate_pair in other_buffer.get_buffer():
                self.append(env_predicate_pair)

    def save(self, current_iteration: int, resampling: bool = False) -> None:
        self.logger.debug("Saving buffer of env predicate pairs at iteration {}".format(current_iteration))
        filename = (
            self.save_dir + "/buffer_predicate_pairs_" + str(current_iteration) + ".txt"
            if not resampling
            else self.save_dir + "/buffer_predicate_pairs_" + str(current_iteration) + "_resampling.txt"
        )
        with open(filename, "w+", encoding="utf-8") as f:
            for env_predicate_pair in self.env_predicate_pairs:
                if env_predicate_pair.get_probability_estimation_runs():
                    pass_probability = env_predicate_pair.compute_pass_probability()
                    probability_estimation_runs = env_predicate_pair.get_probability_estimation_runs()
                elif env_predicate_pair.get_pass_probability():
                    pass_probability = env_predicate_pair.get_pass_probability()
                    probability_estimation_runs = []
                else:
                    pass_probability = 1.0 if env_predicate_pair.is_predicate() else 0.0
                    probability_estimation_runs = []

                if env_predicate_pair.get_regression_estimation_runs():
                    regression_probability = env_predicate_pair.compute_regression_probability()
                    regression_estimation_runs = env_predicate_pair.get_regression_estimation_runs()
                else:
                    regression_probability = 0.0 if env_predicate_pair.get_pass_probability() else 1.0
                    regression_estimation_runs = []

                f.write(
                    env_predicate_pair.get_env_variables().get_params_string()
                    + " - "
                    + str(pass_probability)
                    + " - "
                    + str(env_predicate_pair.is_predicate())
                    + " - "
                    + str(regression_probability)
                    + " - "
                    + str(probability_estimation_runs)
                    + " - "
                    + str(regression_estimation_runs)
                    + " - "
                    + str(env_predicate_pair.get_model_dirs())
                    + "\n"
                )

    def get_buffer(self) -> List[EnvPredicatePair]:
        return self.env_predicate_pairs

    def get_indexes_most_recent_param_changed(self, exp_search_counter: int) -> List[int]:
        # depending on the context I may need to take into account first-last envs or last_but_one-last envs
        if exp_search_counter > 0:
            diff_lastbutone_last = self._array_difference(first_index=-2, second_index=-1)
            self.logger.debug("Diff last, last_but_one: {}".format(diff_lastbutone_last))
            return list(diff_lastbutone_last.nonzero()[0])
        diff_first_last = self._array_difference(first_index=0, second_index=-1)
        self.logger.debug("Diff first, last: {}".format(diff_first_last))
        return list(diff_first_last.nonzero()[0])

    def _array_difference(self, first_index: int, second_index: int) -> np.ndarray:
        first_index_values = np.asarray(self.env_predicate_pairs[first_index].get_env_variables().get_values())
        second_index_values = np.asarray(self.env_predicate_pairs[second_index].get_env_variables().get_values())
        return abs(first_index_values - second_index_values)

import copy
from typing import Dict, List, Tuple

from envs.env_variables import EnvVariables
from log import Log
from utilities import norm
import numpy as np
import math


class FrontierPair:
    def __init__(self, t_env_variables: EnvVariables, f_env_variables: EnvVariables, best_distance: float = None):
        self.t_env_variables = copy.deepcopy(t_env_variables)
        self.f_env_variables = copy.deepcopy(f_env_variables)
        self.best_distance = best_distance

    def get_t_env_variables(self):
        return self.t_env_variables

    def get_f_env_variables(self):
        return self.f_env_variables

    def get_best_distance(self):
        return self.best_distance

    def is_equal(self, other) -> bool:
        if not isinstance(other, FrontierPair):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.t_env_variables.is_equal(other.t_env_variables) and self.f_env_variables.is_equal(other.f_env_variables)


def read_saved_archive(archive_file: str) -> List[Tuple[Dict, bool]]:
    result = []
    with open(archive_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                continue

            true_or_false_predicate = True
            if line.startswith("t_env:"):
                line = line.replace("t_env:", "")
            elif line.startswith("f_env:"):
                true_or_false_predicate = False
                line = line.replace("f_env:", "")

            # deal with '-' sign in the dictionary containing values which is the same as the separator
            index_open_curly_brace = line.index('{')
            index_closed_curly_brace = line.index('}')
            env_values = eval(line[index_open_curly_brace:index_closed_curly_brace + 1])
            result.append((env_values, true_or_false_predicate))

    return result


def is_in_archive(env_values: Tuple[List[float], float], archive: List[Tuple[Dict, bool]], param_names: List[str]) -> Tuple[bool, str]:
    for i in range(len(archive)):
        archive_item = archive[i]
        archive_item_values = [value for key, value in archive_item[0].items() if key in param_names]
        if env_values[0] == archive_item_values:
            if archive_item[1]:
                return True, str(int(i/2)) + '_1'
            return True, str(int(i/2)) + '_0'
    return False, ''


def is_frontier_pair(t_env_variables: EnvVariables, f_env_variables: EnvVariables, epsilon: float, dist: float = None) -> bool:
    return is_frontier_pair_values(t_env_values=t_env_variables.get_values(),
                                   f_env_values=f_env_variables.get_values(),
                                   epsilon=epsilon,
                                   dist=dist)


def compute_dist(t_env_variables: EnvVariables, f_env_variables: EnvVariables) -> float:
    return compute_dist_values(t_env_values=t_env_variables.get_values(), f_env_values=f_env_variables.get_values())


def is_frontier_pair_values(t_env_values: List, f_env_values: List, epsilon: float, num_params_to_consider: int = None, dist: float = None) -> bool:
    if not dist:
        dist = compute_dist_values(t_env_values=t_env_values,
                                   f_env_values=f_env_values,
                                   num_params_to_consider=num_params_to_consider)
    if math.isclose(dist, epsilon, abs_tol=1e-4):
        return True
    return dist <= epsilon


def compute_dist_values(t_env_values: List, f_env_values: List, num_params_to_consider: int = None) -> float:
    dist = 0
    for i in range(len(t_env_values)):
        t = t_env_values[i]
        f = f_env_values[i]
        dist += abs(t - f) / ((abs(t) + abs(f)) / 2) if (abs(t) + abs(f)) != 0.0 else 0.0
    return dist / len(t_env_values) if not num_params_to_consider else dist / num_params_to_consider


def compute_inverse_dist_random_search(env_variables: EnvVariables, index_param: int, epsilon: float) -> List[float]:
    num_params = len(env_variables.get_params())
    env_value = env_variables.get_param(index=index_param).get_current_value()
    sol1 = (env_value * (-epsilon) * num_params - 2 * env_value) / (epsilon * num_params - 2)
    sol2 = (2 * env_value - env_value * epsilon * num_params) / (epsilon * num_params + 2)
    return [sol1, sol2]


class Archive:
    def __init__(self, save_dir: str = None, epsilon: float = 0.05):
        assert save_dir, 'save_dir should have a value: {}'.format(save_dir)
        self.save_dir = save_dir
        self.frontier_pairs: List[FrontierPair] = []
        self.best_frontier_pairs: List[FrontierPair] = []
        self.logger = Log("Archive")
        self.epsilon = epsilon

    def append_best_frontier_pair(self, t_env_variables: EnvVariables, f_env_variables: EnvVariables, best_distance: float) -> bool:
        candidate_frontier_pair = FrontierPair(t_env_variables, f_env_variables, best_distance=best_distance)
        for frontier_pair in self.frontier_pairs:
            if frontier_pair.is_equal(candidate_frontier_pair):
                return False
        self.best_frontier_pairs.append(candidate_frontier_pair)
        return True

    def append(self, t_env_variables: EnvVariables, f_env_variables: EnvVariables) -> bool:
        assert is_frontier_pair(t_env_variables=t_env_variables, f_env_variables=f_env_variables, epsilon=self.epsilon), \
            'The pair t_env: {} - f_env: {} is not a frontier pair since its distance {} is > {}'.format(
                t_env_variables.get_params_string(),
                f_env_variables.get_params_string(),
                compute_dist(t_env_variables=t_env_variables, f_env_variables=f_env_variables),
                self.epsilon)
        candidate_frontier_pair = FrontierPair(t_env_variables, f_env_variables)
        for frontier_pair in self.frontier_pairs:
            if frontier_pair.is_equal(candidate_frontier_pair):
                return False
        self.logger.info(
            "New frontier pair found. t_env: {}, f_env: {}".format(
                t_env_variables.get_params_string(), f_env_variables.get_params_string()
            )
        )
        self.frontier_pairs.append(candidate_frontier_pair)
        return True

    def is_empty(self) -> bool:
        return len(self.frontier_pairs) == 0

    def save(self, current_iteration, resampling: bool = False):
        if len(self.frontier_pairs) > 0:
            self.logger.debug("Saving archive at iteration {}".format(current_iteration))
            filename = self.save_dir + "/frontier_" + str(current_iteration) + ".txt" if not resampling else \
                self.save_dir + "/frontier_" + str(current_iteration) + "_resampling.txt"
            with open(filename, "w+", encoding="utf-8") as f:
                for frontier_pair in self.frontier_pairs:
                    f.write("t_env:" + frontier_pair.get_t_env_variables().get_params_string() + "\n")
                    f.write("f_env:" + frontier_pair.get_f_env_variables().get_params_string() + "\n")
                    f.write("\n")

        if len(self.best_frontier_pairs) > 0:
            self.logger.debug("Saving archive at iteration {}".format(current_iteration))
            filename = self.save_dir + "/greater_than_epsilon_pairs_" + str(current_iteration) + ".txt" if not resampling else \
                self.save_dir + "/frontier_" + str(current_iteration) + "_resampling.txt"
            with open(filename, "w+", encoding="utf-8") as f:
                for frontier_pair in self.best_frontier_pairs:
                    f.write("t_env:" + frontier_pair.get_t_env_variables().get_params_string() + "\n")
                    f.write("f_env:" + frontier_pair.get_f_env_variables().get_params_string() + "\n")
                    f.write("dist:" + str(frontier_pair.get_best_distance()) + "\n")
                    f.write("\n")

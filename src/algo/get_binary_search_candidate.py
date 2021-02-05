from algo.env_predicate_pair import BufferEnvPredicatePairs
from env_utils import instantiate_env_variables
from envs.env_variables import EnvVariables
import random
import itertools

from log import Log
import copy


def get_binary_search_candidate(
        t_env_variables: EnvVariables,
        f_env_variables: EnvVariables,
        algo_name: str,
        env_name: str,
        param_names,
        discrete_action_space: bool,
        buffer_env_predicate_pairs: BufferEnvPredicatePairs
) -> EnvVariables:
    original_max_iterations = 50
    logger = Log('get_binary_search_candidate')
    max_number_iterations = original_max_iterations

    candidate_new_env_variables = copy.deepcopy(t_env_variables)

    while True:

        # compute all possible combinations of environments
        candidates_dict = dict()
        t_f_env_variables = random.choice([(t_env_variables, True), (f_env_variables, False)])

        for i in range(len(t_env_variables.get_params())):
            new_value = (t_env_variables.get_param(index=i).get_current_value()
                         + f_env_variables.get_param(index=i).get_current_value()) / 2
            if i not in candidates_dict:
                candidates_dict[i] = []
            if t_env_variables.get_param(index=i).get_current_value() != f_env_variables.get_param(
                    index=i).get_current_value():
                candidates_dict[i].append(new_value)
            for index in range(len(t_env_variables.get_params())):
                if index not in candidates_dict:
                    candidates_dict[index] = []
                if index != i:
                    candidates_dict[index].append(t_f_env_variables[0].get_values()[index])

        all_candidates = list(itertools.product(*list(candidates_dict.values())))
        logger.info('t_env: {}, f_env: {}'.format(
            t_env_variables.get_params_string(), f_env_variables.get_params_string())
        )
        logger.info('all candidates binary search: {}'.format(all_candidates))
        all_candidates_env_variables_filtered = []
        all_candidates_env_variables = []
        for candidate_values in all_candidates:
            env_values = dict()
            for i in range(len(t_f_env_variables[0].get_params())):
                param_name = t_f_env_variables[0].get_param(index=i).get_name()
                env_values[param_name] = candidate_values[i]
            candidate_env_variables = instantiate_env_variables(
                algo_name=algo_name,
                discrete_action_space=discrete_action_space,
                env_name=env_name,
                param_names=param_names,
                env_values=env_values
            )
            # do not consider candidate = t_f_env_variables
            if not candidate_env_variables.is_equal(t_env_variables) and not candidate_env_variables.is_equal(f_env_variables):
                if not buffer_env_predicate_pairs.is_already_evaluated(
                        candidate_env_variables=candidate_env_variables):
                    all_candidates_env_variables_filtered.append(candidate_env_variables)
                all_candidates_env_variables.append(candidate_env_variables)

        if len(all_candidates_env_variables_filtered) > 0:
            candidate_new_env_variables = random.choice(all_candidates_env_variables_filtered)
            break
        else:
            assert len(all_candidates) > 0, 'there must be at least one candidate env for binary search'
            candidate_env_variables_already_evaluated = random.choice(all_candidates_env_variables_filtered)
            if t_f_env_variables[1]:
                t_env_variables = copy.deepcopy(candidate_env_variables_already_evaluated)
            else:
                f_env_variables = copy.deepcopy(candidate_env_variables_already_evaluated)

        max_number_iterations -= 1

        if max_number_iterations == 0:
            break

    assert max_number_iterations > 0, \
        "Could not binary mutate any param of envs {} and {} in {} steps".format(
            t_env_variables.get_params_string(),
            f_env_variables.get_params_string(),
            str(original_max_iterations)
        )

    assert not candidate_new_env_variables.is_equal(t_env_variables) \
           and not candidate_new_env_variables.is_equal(f_env_variables), \
        'candidate_env_variables {} must be different than t_env_variables {} and f_env_variables {}'.format(
            candidate_new_env_variables.get_params_string(), t_env_variables.get_params_string(),
            f_env_variables.get_params_string()
        )

    return candidate_new_env_variables

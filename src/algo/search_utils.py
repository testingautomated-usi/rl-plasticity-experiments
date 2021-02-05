from typing import List, Dict, Tuple

from algo.env_predicate_pair import EnvPredicatePair
from log import Log


class BufferExecutionsSkippedItem:

    def __init__(self, env_values_skipped: Dict, predicate: bool,
                 env_values_executed: Dict, search_component: str):
        self.env_values_skipped = env_values_skipped
        self.env_values_executed = env_values_executed
        self.search_component = search_component
        self.predicate = predicate


def read_saved_buffer_executions_skipped(buffer_executions_skipped_file: str) -> List[BufferExecutionsSkippedItem]:
    result = []
    with open(buffer_executions_skipped_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                break

            true_or_false_predicate = True
            if 'False' in line:
                true_or_false_predicate = False

            index_first_open_curly_brace = line.index('{')
            index_dominate = line.index('dominate')
            env_values_skipped = eval(line[index_first_open_curly_brace:index_dominate - 1])
            index_second_close_curly_brace = line.rindex('}')
            index_last_env = line.rindex('env')
            env_values_executed = eval(line[index_last_env + 4:index_second_close_curly_brace + 1])

            search_component = 'exp_search'
            if 'binary_search' in line:
                search_component = 'binary_search'

            result.append(BufferExecutionsSkippedItem(
                env_values_skipped=env_values_skipped,
                predicate=true_or_false_predicate,
                env_values_executed=env_values_executed,
                search_component=search_component)
            )

    return result


class ExecutionSkipped:

    def __init__(self, env_predicate_pair_skipped: EnvPredicatePair,
                 env_predicate_pair_executed: EnvPredicatePair,
                 search_component: str):
        self.env_predicate_pair_skipped = env_predicate_pair_skipped
        self.env_predicate_pair_executed = env_predicate_pair_executed
        self.search_component = search_component


class BufferExecutionsSkipped:

    def __init__(self, save_dir: str = None):
        assert save_dir, 'save_dir should have a value: {}'.format(save_dir)
        self.save_dir = save_dir
        self.executions_skipped: List[ExecutionSkipped] = []
        self.logger = Log("BufferEnvPredicatePairs")

    def append(self, execution_skipped: ExecutionSkipped):
        if not self.is_already_evaluated(other_execution_skipped=execution_skipped):
            self.executions_skipped.append(execution_skipped)

    def is_already_evaluated(self, other_execution_skipped: ExecutionSkipped) -> bool:
        for execution_skipped in self.executions_skipped:
            evaluated_env_variables = execution_skipped.env_predicate_pair_skipped.get_env_variables()
            if evaluated_env_variables.is_equal(other_execution_skipped.env_predicate_pair_skipped.get_env_variables()):
                self.logger.debug("Env {} was already evaluated".format(
                    other_execution_skipped.env_predicate_pair_skipped.get_env_variables().get_params_string()))
                return True
        return False

    def save(self, current_iteration: int) -> None:
        if len(self.executions_skipped) > 0:
            self.logger.debug("Saving buffer of env executions skipped at iteration {}".format(current_iteration))
            filename = self.save_dir + "/executions_skipped_" + str(current_iteration) + ".txt"
            with open(filename, "w+", encoding="utf-8") as f:
                executions_skipped_by_search_component = dict()
                for execution_skipped in self.executions_skipped:
                    if execution_skipped.search_component not in executions_skipped_by_search_component:
                        executions_skipped_by_search_component[execution_skipped.search_component] = 0
                    executions_skipped_by_search_component[execution_skipped.search_component] += 1
                    if execution_skipped.env_predicate_pair_executed.is_predicate():
                        f.write('env {} dominated by True env {} in {} \n'.format(
                            execution_skipped.env_predicate_pair_skipped.get_env_variables().get_params_string(),
                            execution_skipped.env_predicate_pair_executed.get_env_variables().get_params_string(),
                            execution_skipped.search_component
                        ))
                    elif not execution_skipped.env_predicate_pair_executed.is_predicate():
                        f.write('env {} dominates False env {} in {} \n'.format(
                            execution_skipped.env_predicate_pair_skipped.get_env_variables().get_params_string(),
                            execution_skipped.env_predicate_pair_executed.get_env_variables().get_params_string(),
                            execution_skipped.search_component
                        ))

                f.write('\n')
                for key in executions_skipped_by_search_component.keys():
                    f.write('executions skipped by {}: {} \n'.format(key, executions_skipped_by_search_component[key]))


def read_executions_skipped_totals(executions_skipped_file: str) -> Dict:
    executions_skipped_by_search_type = dict()
    with open(executions_skipped_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith('executions skipped by'):
                text = line.split(':')[0]
                search_type = text.replace('executions skipped by ', '')
                if search_type not in executions_skipped_by_search_type:
                    executions_skipped_by_search_type[search_type] = 0
                num = eval(line.split(':')[1])
                executions_skipped_by_search_type[search_type] = num
        return executions_skipped_by_search_type


def read_executions_skipped(executions_skipped_file: str, param_names: List[str]) -> List[Tuple[Dict, Dict]]:
    # tuple in which the first item dominates the second
    result = []
    with open(executions_skipped_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if 'dominated by True' in line:
                index_first_curly = line.index('{')
                index_last_curly = line.index('}')
                env_dict_1 = eval(line[index_first_curly:index_last_curly + 1])
                env_dict_1_filtered = dict()
                for key, value in env_dict_1.items():
                    if key in param_names:
                        env_dict_1_filtered[key] = value
                line = line.replace(line[index_first_curly:index_last_curly + 1], '')

                line = line.replace('dominated by True env', '')
                index_first_curly = line.index('{')
                index_last_curly = line.index('}')
                env_dict_2 = eval(line[index_first_curly:index_last_curly + 1])
                env_dict_2_filtered = dict()
                for key, value in env_dict_2.items():
                    if key in param_names:
                        env_dict_2_filtered[key] = value
                result.append((env_dict_2_filtered, env_dict_1_filtered))
            elif 'dominates False' in line:
                index_first_curly = line.index('{')
                index_last_curly = line.index('}')
                env_dict_1 = eval(line[index_first_curly:index_last_curly + 1])
                env_dict_1_filtered = dict()
                for key, value in env_dict_1.items():
                    if key in param_names:
                        env_dict_1_filtered[key] = value
                line = line.replace(line[index_first_curly:index_last_curly + 1], '')

                line = line.replace('dominates False env', '')
                index_first_curly = line.index('{')
                index_last_curly = line.index('}')
                env_dict_2 = eval(line[index_first_curly:index_last_curly + 1])
                env_dict_2_filtered = dict()
                for key, value in env_dict_2.items():
                    if key in param_names:
                        env_dict_2_filtered[key] = value
                result.append((env_dict_1_filtered, env_dict_2_filtered))
        return result

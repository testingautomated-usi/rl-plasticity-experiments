from typing import List


def save_time_elapsed(save_dir: str, current_iteration: int,
                      execution_times: List[float], regression: bool = False) -> None:
    filename = save_dir + "/time_elapsed_" + str(current_iteration) + ".txt" if not regression \
        else save_dir + "/regression_time_" + str(current_iteration) + ".txt"
    with open(filename, "w+", encoding="utf-8") as f:
        f.write('Execution time: {} s'.format(sum(execution_times)))


def read_time_elapsed(time_elapsed_file: str) -> float:
    with open(time_elapsed_file, "r", encoding="utf-8") as f:
        line = f.read()
        line_split_description = line.split(':')
        line_split_seconds = line_split_description[1].split('s')
        return eval(line_split_seconds[0])

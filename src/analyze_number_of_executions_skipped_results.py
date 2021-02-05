import argparse
import glob
import itertools
import logging
import math
import os
from statistics.wilcoxon import summary

import numpy as np

from algo.archive import read_saved_archive
from algo.env_predicate_pair import read_saved_buffer
from algo.search_utils import (read_executions_skipped,
                               read_executions_skipped_totals)
from algo.time_elapsed_util import read_time_elapsed
from env_utils import instantiate_env_variables
from log import Log
from utilities import (SUPPORTED_ALGOS, SUPPORTED_ENVS, check_file_existence,
                       check_param_names, filter_resampling_artifacts,
                       get_result_dir_iteration_number,
                       get_result_file_iteration_number)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=check_file_existence, required=True)
    parser.add_argument("--param_names", type=check_param_names, required=True)
    parser.add_argument("--algo_name", choices=SUPPORTED_ALGOS, required=True)
    parser.add_argument("--env_name", choices=SUPPORTED_ENVS, required=True)

    args = parser.parse_args()

    logger = Log("analyze_number_of_executions_skipped")
    logging.basicConfig(
        filename=os.path.join(args.dir, "analyze_number_of_executions_skipped.txt"), filemode="w", level=logging.DEBUG
    )

    env_variables = instantiate_env_variables(
        algo_name=args.algo_name,
        discrete_action_space=False,  # I do not care about this parameter in this case
        env_name=args.env_name,
        param_names=args.param_names,
    )

    iterations_dirs = glob.glob(os.path.join(args.dir, "n_iterations_*"))
    iterations_dirs = filter_resampling_artifacts(files=iterations_dirs)
    iterations_dirs_sorted = sorted(iterations_dirs, key=get_result_dir_iteration_number)

    executions_skipped_dict = dict()

    all_search_points = []
    time_elapsed_per_run = []
    time_taken_per_repetition = []
    regression_time_per_repetition = []
    all_frontier_points = []
    executions_skipped = []

    for i, iteration_dir in enumerate(iterations_dirs_sorted):
        iteration_dir = os.path.join(args.dir, iteration_dir)
        logger.info("Analyzing folder {}".format(iteration_dir))

        list_of_buffer_files = glob.glob(os.path.join(iteration_dir, "buffer_predicate_pairs_*.txt"))
        list_of_buffer_files = filter_resampling_artifacts(files=list_of_buffer_files)
        last_buffer_file = max(list_of_buffer_files, key=get_result_file_iteration_number)
        all_search_points.append(len(read_saved_buffer(buffer_file=last_buffer_file)))

        list_of_archive_files = glob.glob(os.path.join(iteration_dir, "frontier_*.txt"))
        list_of_archive_files = filter_resampling_artifacts(files=list_of_archive_files)

        if len(list_of_archive_files) > 0:
            last_archive_file = max(list_of_archive_files, key=get_result_file_iteration_number)
            archive = read_saved_archive(archive_file=last_archive_file)
            all_frontier_points.append(len(archive) / 2)
        else:
            all_frontier_points.append(0)

        list_of_executions_skipped_files = glob.glob(os.path.join(iteration_dir, "executions_skipped_*.txt"))
        last_executions_skipped_file = max(list_of_executions_skipped_files, key=get_result_file_iteration_number)
        executions_skipped_by_search_type_dict = read_executions_skipped_totals(
            executions_skipped_file=last_executions_skipped_file
        )
        executions_skipped_lines = read_executions_skipped(
            executions_skipped_file=last_executions_skipped_file, param_names=args.param_names
        )
        for item in executions_skipped_lines:
            dominates_env = item[0]
            dominated_env = item[1]
            dimension_strictly_greater = False
            for param_name, param_value in dominates_env.items():
                param = list(filter(lambda param_: param_.get_name() == param_name, env_variables.get_params()))[0]
                if param.get_starting_multiplier() > 1.0:
                    assert param_value >= dominated_env[param_name], "Dominates env {} is not >= than dominated env {}".format(
                        dominates_env, dominated_env
                    )
                    if param_value > dominated_env[param_name]:
                        dimension_strictly_greater = True
                elif param.get_starting_multiplier() < 1.0:
                    assert param_value <= dominated_env[param_name], "Dominates env {} is not <= than dominated env {}".format(
                        dominates_env, dominated_env
                    )
                    if param_value < dominated_env[param_name]:
                        dimension_strictly_greater = True
            assert dimension_strictly_greater, "Dominates env {} is not >= than dominated env {}".format(
                dominates_env, dominated_env
            )

        executions_skipped_sum = 0
        for search_type in executions_skipped_by_search_type_dict.keys():
            executions_skipped_sum += executions_skipped_by_search_type_dict[search_type]
        executions_skipped.append(executions_skipped_sum)

        list_of_iterations = glob.glob(os.path.join(iteration_dir, "iteration_*"))
        list_of_iterations = filter_resampling_artifacts(files=list_of_iterations)
        list_of_iterations_sorted = sorted(list_of_iterations, key=get_result_dir_iteration_number)

        time_taken_per_iteration = []
        regression_time_per_iteration = []
        time_elapsed_per_iteration_files = glob.glob(os.path.join(iteration_dir, "time_elapsed_*.txt"))
        time_elapsed_per_iteration_files_sorted = sorted(
            time_elapsed_per_iteration_files, key=get_result_file_iteration_number
        )

        num_runs_probability_estimation = 0
        for time_elapsed_per_iteration_file in time_elapsed_per_iteration_files_sorted:
            time_elapsed_per_iteration_file = os.path.join(iteration_dir, time_elapsed_per_iteration_file)
            time_elapsed_iteration_j = read_time_elapsed(time_elapsed_file=time_elapsed_per_iteration_file)
            j = get_result_file_iteration_number(time_elapsed_per_iteration_file)
            time_taken_per_iteration.append(time_elapsed_iteration_j)
            if math.isclose(time_elapsed_iteration_j, 0.0):
                time_elapsed_per_run.append(0)
            else:
                iteration_js = list(
                    filter(
                        lambda iteration_dir_name: get_result_dir_iteration_number(iteration_dir_name) == j,
                        list_of_iterations_sorted,
                    )
                )
                assert len(iteration_js) == 1, "There should only be one match. Found {}".format(len(iteration_js))
                iteration_j = os.path.join(iteration_dir, iteration_js[0])
                num_of_runs_iteration_j = len(glob.glob(os.path.join(iteration_j, "logs_*")))
                one_run_in_iteration = glob.glob(os.path.join(iteration_j, "logs_*"))[0]
                num_runs_probability_estimation = len(
                    glob.glob(
                        os.path.join(iteration_j, "{}*".format(one_run_in_iteration[: one_run_in_iteration.index("run")]))
                    )
                )
                assert num_of_runs_iteration_j != 0, "In {} there is no run dir".format(iteration_j)
                time_elapsed_per_run.append(time_elapsed_iteration_j / num_of_runs_iteration_j)

        assert num_runs_probability_estimation != 0, "Failed to estimate num of probability estimation runs"
        time_taken_per_repetition.append((np.asarray(time_taken_per_iteration) / num_runs_probability_estimation).sum())

    mean, std, min_, max_ = summary(a=all_search_points)
    logger.info("Search points summary. Mean {}, Std {}, Min {}, Max {}".format(mean, std, min_, max_))
    mean, std, min_, max_ = summary(a=all_frontier_points)
    logger.info("Frontier points summary. Mean {}, Std {}, Min {}, Max {}".format(mean, std, min_, max_))
    mean, std, min_, max_ = summary(a=executions_skipped)
    logger.info("Executions skipped total summary. Mean {}, Std {}, Min {}, Max {}".format(mean, std, min_, max_))
    mean, std, min_, max_ = summary(a=time_elapsed_per_run)
    logger.info("Time elapsed per run [s] summary. Mean {}, Std {}, Min {}, Max {}".format(mean, std, min_, max_))
    mean, std, min_, max_ = summary(a=time_taken_per_repetition)
    logger.info("Time elapsed per repetition [s] summary. Mean {}, Std {}, Min {}, Max {}".format(mean, std, min_, max_))

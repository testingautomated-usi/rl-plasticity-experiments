import argparse
import glob
import itertools
import logging
import os

import numpy as np

from algo.archive import read_saved_archive
from log import Log
from utilities import (check_file_existence, compute_statistics,
                       filter_resampling_artifacts,
                       get_result_dir_iteration_number,
                       get_result_file_iteration_number)

# for comparison between random and guided choice in exponential and binary search
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=check_file_existence, required=True)
    parser.add_argument("--first_mode_dir", type=check_file_existence, required=True)
    parser.add_argument("--second_mode_dir", type=check_file_existence, required=True)
    parser.add_argument("--num_runs_probability_estimation", type=int, required=True, default=3)
    args = parser.parse_args()

    logger = Log("analyze_guidance_results")
    logging.basicConfig(
        filename=os.path.join(args.save_dir, "analyze_guidance_results.txt"), filemode="w", level=logging.DEBUG
    )

    # analysis per iteration, per search type (binary, exp), cumulative
    binary_search_runs_comparison = []
    exp_search_runs_comparison = []
    frontier_points_comparison = []

    dirs_to_analyze = [args.first_mode_dir, args.second_mode_dir]
    for dir_to_analyze in dirs_to_analyze:

        iterations_dirs = glob.glob(os.path.join(dir_to_analyze, "n_iterations_*"))
        iterations_dirs_sorted = sorted(iterations_dirs, key=get_result_dir_iteration_number)

        frontier_points = []
        binary_search_runs = []
        exp_search_runs = []

        for i, iteration_dir in enumerate(iterations_dirs_sorted):
            iteration_dir = os.path.join(dir_to_analyze, iteration_dir)
            logger.info("Analyzing folder {}".format(iteration_dir))

            list_of_archive_files = glob.glob(os.path.join(iteration_dir, "frontier_*.txt"))
            list_of_archive_files = filter_resampling_artifacts(files=list_of_archive_files)
            last_archive_file = max(list_of_archive_files, key=get_result_file_iteration_number)
            archive = read_saved_archive(archive_file=last_archive_file)
            frontier_points.append(len(archive))

            single_iteration_dirs = glob.glob(os.path.join(iteration_dir, "iteration_*"))
            single_iteration_dirs = filter_resampling_artifacts(files=single_iteration_dirs)
            single_iteration_dirs_sorted = sorted(single_iteration_dirs, key=get_result_file_iteration_number)
            binary_search_runs_it = []
            exp_search_runs_it = []
            for j, single_iteration_dir in enumerate(single_iteration_dirs_sorted):
                single_iteration_dir = os.path.join(iteration_dir, single_iteration_dir)
                binary_search_runs_dirs = glob.glob(os.path.join(single_iteration_dir, "*_binary_search_*"))
                exp_search_runs_dirs = glob.glob(os.path.join(single_iteration_dir, "*_exp_search_*"))
                if len(exp_search_runs_dirs) != 0:
                    if len(binary_search_runs_dirs) == 0:
                        logger.warn("!!!!!!!!! Binary search runs are not present. " "Make sure this is correct. !!!!!!!!! ")
                    binary_search_runs_it.append(len(binary_search_runs_dirs) / args.num_runs_probability_estimation)
                    exp_search_runs_it.append(len(exp_search_runs_dirs) / args.num_runs_probability_estimation)

            binary_search_runs.append(binary_search_runs_it)
            exp_search_runs.append(exp_search_runs_it)

        binary_search_runs_comparison.append(binary_search_runs)
        exp_search_runs_comparison.append(exp_search_runs)
        frontier_points_comparison.append(frontier_points)

    binary_search_runs_comparison_0 = binary_search_runs_comparison[0]
    exp_search_runs_comparison_0 = exp_search_runs_comparison[0]
    binary_search_runs_comparison_1 = binary_search_runs_comparison[1]
    exp_search_runs_comparison_1 = exp_search_runs_comparison[1]
    frontier_points_comparison_0 = frontier_points_comparison[0]
    frontier_points_comparison_1 = frontier_points_comparison[1]

    num_iterations_0 = [len(single_iteration_runs) for single_iteration_runs in binary_search_runs_comparison_0]
    num_iterations_1 = [len(single_iteration_runs) for single_iteration_runs in binary_search_runs_comparison_1]
    for i in range(len(num_iterations_0)):
        if num_iterations_0[i] != num_iterations_1[i]:
            diff = num_iterations_0[i] - num_iterations_1[i]
            if diff > 0:
                # remove elements first list
                binary_search_runs_comparison_0[i].pop()
                exp_search_runs_comparison_0[i].pop()
            else:
                # remove elements second list
                binary_search_runs_comparison_1[i].pop()
                exp_search_runs_comparison_1[i].pop()

    iteration_binary_dict_0 = dict()
    iteration_exp_dict_0 = dict()
    iteration_binary_dict_1 = dict()
    iteration_exp_dict_1 = dict()
    for num_search_run in range(len(binary_search_runs_comparison_0)):
        for num_iteration in range(len(binary_search_runs_comparison_0[num_search_run])):
            if num_iteration not in iteration_binary_dict_0:
                iteration_binary_dict_0[num_iteration] = []
                iteration_exp_dict_0[num_iteration] = []
                iteration_binary_dict_1[num_iteration] = []
                iteration_exp_dict_1[num_iteration] = []
            iteration_binary_dict_0[num_iteration].append(binary_search_runs_comparison_0[num_search_run][num_iteration])
            iteration_exp_dict_0[num_iteration].append(exp_search_runs_comparison_0[num_search_run][num_iteration])
            iteration_binary_dict_1[num_iteration].append(binary_search_runs_comparison_1[num_search_run][num_iteration])
            iteration_exp_dict_1[num_iteration].append(exp_search_runs_comparison_1[num_search_run][num_iteration])

    logger.info("******* frontier points comparison *******")
    compute_statistics(a=frontier_points_comparison_0, b=frontier_points_comparison_1, _logger=logger)

    logger.info("******* cumulative comparison *******")
    flatten_binary_search_runs_0 = list(itertools.chain(*binary_search_runs_comparison_0))
    flatten_exp_search_runs_0 = list(itertools.chain(*exp_search_runs_comparison_0))
    flatten_binary_search_runs_1 = list(itertools.chain(*binary_search_runs_comparison_1))
    flatten_exp_search_runs_1 = list(itertools.chain(*exp_search_runs_comparison_1))

    sum_exp_binary_0 = np.asarray(flatten_binary_search_runs_0) + np.asarray(flatten_exp_search_runs_0)
    sum_exp_binary_1 = np.asarray(flatten_binary_search_runs_1) + np.asarray(flatten_exp_search_runs_1)
    compute_statistics(a=sum_exp_binary_0, b=sum_exp_binary_1, _logger=logger)
    logger.info("")

    logger.info("******* per_type_search comparison *******")
    logger.info("-------- exp search --------")
    compute_statistics(a=flatten_exp_search_runs_0, b=flatten_exp_search_runs_1, _logger=logger)
    logger.info("-------- binary search --------")
    compute_statistics(a=flatten_binary_search_runs_0, b=flatten_binary_search_runs_1, _logger=logger)
    logger.info("")

    logger.info("******* cumulative comparison per iteration *******")
    for num_iteration in iteration_binary_dict_0.keys():
        sum_exp_binary_0 = np.asarray(iteration_binary_dict_0[num_iteration]) + np.asarray(iteration_exp_dict_0[num_iteration])
        sum_exp_binary_1 = np.asarray(iteration_binary_dict_1[num_iteration]) + np.asarray(iteration_exp_dict_1[num_iteration])
        logger.info("++++++++ iteration #{} ++++++++ ".format(num_iteration))
        compute_statistics(a=sum_exp_binary_0, b=sum_exp_binary_1)
    logger.info("")

    logger.info("******* per_type_search comparison per iteration *******")
    for num_iteration in iteration_binary_dict_0.keys():
        logger.info("++++++++ iteration #{} ++++++++ ".format(num_iteration))
        logger.info("-------- exp search --------")
        compute_statistics(a=iteration_exp_dict_0[num_iteration], b=iteration_exp_dict_1[num_iteration], _logger=logger)
        logger.info("-------- binary search --------")
        compute_statistics(a=iteration_binary_dict_0[num_iteration], b=iteration_binary_dict_1[num_iteration], _logger=logger)

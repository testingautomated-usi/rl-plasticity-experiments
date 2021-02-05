import argparse
import logging
import os
import glob
import numpy as np
import math

import sklearn
from scipy.spatial.distance import pdist

from algo.archive import read_saved_archive, compute_dist, compute_dist_values
from algo.env_predicate_pair import read_saved_buffer
from algo.time_elapsed_util import read_time_elapsed
from analysis.clustering import cluster_data
from env_utils import instantiate_env_variables
from log import Log
from plot.plot_clusters import plot_clusters
from plot.plot_frontier_points import plot_frontier_points_2D
from utilities import check_file_existence, get_result_dir_iteration_number, get_result_file_iteration_number, \
    compute_statistics, filter_resampling_artifacts, SUPPORTED_ALGOS, check_param_names, SUPPORTED_ENVS

# for comparison between random exploration and alphatest in terms of num of frontier points
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=check_file_existence, required=True)
    parser.add_argument("--env_name", choices=SUPPORTED_ENVS, required=True)
    parser.add_argument("--algo_name", choices=SUPPORTED_ALGOS, required=True)
    parser.add_argument("--first_mode_dir", type=check_file_existence, required=True)
    parser.add_argument("--second_mode_dir", type=check_file_existence, required=True)
    parser.add_argument('--param_names', type=check_param_names, required=True)
    args = parser.parse_args()

    assert len(args.param_names) == 2, 'Cannot compare results with environments that have more than 2 params: {}'\
        .format(len(args.param_names))

    env_variables = instantiate_env_variables(
        algo_name=args.algo_name,
        discrete_action_space=False,  # I do not care about this parameter in this context
        env_name=args.env_name,
        param_names=args.param_names,
    )

    params = env_variables.get_params()
    first_param_limits = [params[0].get_low_limit(), params[0].get_high_limit()]
    second_param_limits = [params[1].get_low_limit(), params[1].get_high_limit()]
    point_1_at_max_distance = [first_param_limits[0], second_param_limits[0]]
    point_2_at_max_distance = [first_param_limits[1], second_param_limits[1]]
    max_distance = math.sqrt(((point_1_at_max_distance[0]-point_2_at_max_distance[0])**2)
                             +((point_1_at_max_distance[1]-point_2_at_max_distance[1])**2))

    logger = Log('analyze_effectiveness_results')
    filename = 'analyze_effectiveness_results_{}.txt'.format(args.algo_name)
    logging.basicConfig(
        filename=os.path.join(args.save_dir, filename),
        filemode='w',
        level=logging.DEBUG
    )

    logger.info('Max distance: {}'.format(max_distance))

    num_frontier_points_comparison = []
    frontier_pairs_comparison = []
    search_points_comparison = []
    number_of_iterations_comparison = []
    avg_pairwise_distances_comparison = []
    time_elapsed_per_run_comparison = []
    regression_time_comparison = []
    time_taken_comparison = []
    dists_comparison = []

    green_xs_comparison = []
    green_ys_comparison = []
    red_xs_comparison = []
    red_ys_comparison = []

    dirs_to_analyze = [args.first_mode_dir, args.second_mode_dir]
    for dir_to_analyze in dirs_to_analyze:

        iterations_dirs = glob.glob(os.path.join(dir_to_analyze, "n_iterations_*"))
        iterations_dirs_sorted = sorted(iterations_dirs, key=get_result_dir_iteration_number)

        all_frontier_points = []
        all_search_points = []
        number_of_iterations = []
        avg_pairwise_distances = []
        time_elapsed_per_run = []
        time_taken_per_repetition = []
        regression_time_per_repetition = []
        dists_per_iteration = []
        green_xs = [[] for i in range(len(iterations_dirs_sorted))]
        green_ys = [[] for i in range(len(iterations_dirs_sorted))]
        red_xs = [[] for i in range(len(iterations_dirs_sorted))]
        red_ys = [[] for i in range(len(iterations_dirs_sorted))]
        frontier_pairs = [[] for i in range(len(iterations_dirs_sorted))]

        for i, iteration_dir in enumerate(iterations_dirs_sorted):

            green_xs_run = []
            green_ys_run = []
            red_xs_run = []
            red_ys_run = []
            frontier_pairs_run = []

            iteration_dir = os.path.join(dir_to_analyze, iteration_dir)
            logger.info('Analyzing folder {}'.format(iteration_dir))

            list_of_iterations = glob.glob(os.path.join(iteration_dir, "iteration_*"))
            list_of_iterations = filter_resampling_artifacts(files=list_of_iterations)
            list_of_iterations_sorted = sorted(list_of_iterations, key=get_result_dir_iteration_number)

            list_of_archive_files = glob.glob(os.path.join(iteration_dir, "frontier_*.txt"))
            list_of_archive_files = filter_resampling_artifacts(files=list_of_archive_files)

            list_of_buffer_files = glob.glob(os.path.join(iteration_dir, "buffer_predicate_pairs_*.txt"))
            list_of_buffer_files = filter_resampling_artifacts(files=list_of_buffer_files)

            first_buffer_file = min(list_of_buffer_files, key=get_result_file_iteration_number)
            last_buffer_file = max(list_of_buffer_files, key=get_result_file_iteration_number)
            all_search_points.append(len(read_saved_buffer(buffer_file=last_buffer_file)))
            if len(list_of_archive_files) > 0:
                last_archive_file = max(list_of_archive_files, key=get_result_file_iteration_number)
                archive = read_saved_archive(archive_file=last_archive_file)

                t_env_values, f_env_values = None, None
                dists = []
                avg_points_in_frontier = []
                for env_values, predicate in archive:

                    env_values_to_consider = [env_values[param.get_name()] for param in params]

                    if predicate:
                        t_env_values = env_values_to_consider
                        green_xs_run.append(t_env_values[0])
                        green_ys_run.append(t_env_values[1])
                    else:
                        f_env_values = env_values_to_consider
                        red_xs_run.append(f_env_values[0])
                        red_ys_run.append(f_env_values[1])
                    if t_env_values and f_env_values:
                        dists.append(compute_dist_values(
                            t_env_values=t_env_values, f_env_values=f_env_values,
                            num_params_to_consider=len(args.param_names)
                        ))
                        avg_point = []
                        for ind, t_value in enumerate(t_env_values):
                            avg_point.append((t_value + f_env_values[ind]) / 2)
                        avg_points_in_frontier.append(avg_point)
                        t_env_values, f_env_values = None, None

                    frontier_pairs_run.append(env_values_to_consider)

                logger.info('dists: {}'.format(dists))
                dists_per_iteration.append(np.asarray(dists).mean())
                matrix_pairwise_distances = sklearn.metrics.pairwise_distances(np.asarray(frontier_pairs_run))  # returns distances as matrix
                max_pairwise_distances = []
                for num_frontier_pair in range(len(frontier_pairs_run)):
                    max_distance_for_frontier_pair = np.max(matrix_pairwise_distances[num_frontier_pair])
                    max_pairwise_distances.append(max_distance_for_frontier_pair)
                    assert max_distance_for_frontier_pair < max_distance, \
                        'Max distance for a frontier pair {} cannot be greater ' \
                        'than the max distance in the parameter space {}'.format(
                            max_distance_for_frontier_pair, max_distance
                        )

                # pairwise_distances = pdist(np.asarray(frontier_pairs_run))  # returns distances as an array
                normalized_pairwise_distances = np.asarray(max_pairwise_distances).mean() / max_distance

                green_xs[i] = green_xs_run
                green_ys[i] = green_ys_run
                red_xs[i] = red_xs_run
                red_ys[i] = red_ys_run
                frontier_pairs[i] = frontier_pairs_run

                avg_pairwise_distances.append(normalized_pairwise_distances.mean())

                all_frontier_points.append(len(archive) / 2)
            else:
                all_frontier_points.append(0)
                avg_pairwise_distances.append(0)

            number_of_iterations.append(len(list_of_iterations))

            time_taken_per_iteration = []
            regression_time_per_iteration = []
            time_elapsed_per_iteration_files = glob.glob(os.path.join(iteration_dir, "time_elapsed_*.txt"))
            time_elapsed_per_iteration_files_sorted = sorted(
                time_elapsed_per_iteration_files, key=get_result_file_iteration_number
            )
            regression_time_per_iteration_files = glob.glob(os.path.join(iteration_dir, "regression_time_*.txt"))
            regression_time_per_iteration_files_sorted = sorted(
                regression_time_per_iteration_files, key=get_result_file_iteration_number
            )

            num_runs_probability_estimation = 0
            for time_elapsed_per_iteration_file in time_elapsed_per_iteration_files_sorted:
                time_elapsed_per_iteration_file = os.path.join(iteration_dir, time_elapsed_per_iteration_file)
                time_elapsed_iteration_j = read_time_elapsed(time_elapsed_file=time_elapsed_per_iteration_file)
                j = get_result_file_iteration_number(time_elapsed_per_iteration_file)
                if len(regression_time_per_iteration_files_sorted) != 0:
                    regression_time_per_iteration_file = os.path.join(
                        iteration_dir, regression_time_per_iteration_files_sorted[j]
                    )
                    regression_time_iteration_j = read_time_elapsed(time_elapsed_file=regression_time_per_iteration_file)
                    assert regression_time_iteration_j <= time_elapsed_iteration_j, \
                        'Regression time {} cannot be greater than time elapsed {}'.format(
                            regression_time_iteration_j, time_elapsed_iteration_j)
                    logger.warn('Subtracting regression time {} from time elapsed: {}'.format(
                        regression_time_iteration_j, time_elapsed_iteration_j))
                    time_elapsed_iteration_j -= regression_time_iteration_j
                    regression_time_per_iteration.append(regression_time_iteration_j)
                time_taken_per_iteration.append(time_elapsed_iteration_j)
                if math.isclose(time_elapsed_iteration_j, 0.0):
                    time_elapsed_per_run.append(0)
                else:
                    iteration_js = list(
                        filter(
                            lambda iteration_dir_name: get_result_dir_iteration_number(iteration_dir_name) == j,
                            list_of_iterations_sorted
                        )
                    )
                    assert len(iteration_js) == 1, 'There should only be one match. Found {}'.format(len(iteration_js))
                    iteration_j = os.path.join(iteration_dir, iteration_js[0])
                    num_of_runs_iteration_j = len(glob.glob(os.path.join(iteration_j, "logs_*")))
                    assert num_of_runs_iteration_j != 0, 'In {} there is no run dir'.format(iteration_j)
                    time_elapsed_per_run.append(time_elapsed_iteration_j / num_of_runs_iteration_j)

                    one_run_in_iteration = glob.glob(os.path.join(iteration_j, "logs_*"))[0]
                    num_runs_probability_estimation = len(
                        glob.glob(
                            os.path.join(
                                iteration_j, "{}*".format(one_run_in_iteration[:one_run_in_iteration.index('run')])
                            )
                        )
                    )

            assert num_runs_probability_estimation != 0, 'Failed to estimate num of probability estimation runs'
            time_taken_per_repetition.append((np.asarray(time_taken_per_iteration) / num_runs_probability_estimation).sum())
            regression_time_per_repetition.append(np.asarray(regression_time_per_iteration).sum())

        num_frontier_points_comparison.append(all_frontier_points)
        search_points_comparison.append(all_search_points)
        number_of_iterations_comparison.append(number_of_iterations)
        avg_pairwise_distances_comparison.append(avg_pairwise_distances)
        time_elapsed_per_run_comparison.append(time_elapsed_per_run)
        time_taken_comparison.append(time_taken_per_repetition)
        dists_comparison.append(dists_per_iteration)
        regression_time_comparison.append(regression_time_per_repetition)
        green_xs_comparison.append(green_xs)
        green_ys_comparison.append(green_ys)
        red_xs_comparison.append(red_xs)
        red_ys_comparison.append(red_ys)
        frontier_pairs_comparison.append(frontier_pairs)

    number_of_iterations_comparison_0 = number_of_iterations_comparison[0]
    number_of_iterations_comparison_1 = number_of_iterations_comparison[1]
    search_points_comparison_0 = search_points_comparison[0]
    search_points_comparison_1 = search_points_comparison[1]
    num_frontier_points_comparison_0 = num_frontier_points_comparison[0]
    num_frontier_points_comparison_1 = num_frontier_points_comparison[1]
    frontier_pairs_comparison_0 = frontier_pairs_comparison[0]
    frontier_pairs_comparison_1 = frontier_pairs_comparison[1]
    avg_pairwise_distances_comparison_0 = avg_pairwise_distances_comparison[0]
    avg_pairwise_distances_comparison_1 = avg_pairwise_distances_comparison[1]
    time_elapsed_per_run_comparison_0 = time_elapsed_per_run_comparison[0]
    time_elapsed_per_run_comparison_1 = time_elapsed_per_run_comparison[1]
    time_taken_comparison_0 = time_taken_comparison[0]
    time_taken_comparison_1 = time_taken_comparison[1]
    dists_comparison_0 = dists_comparison[0]
    dists_comparison_1 = dists_comparison[1]
    regression_time_comparison_0 = regression_time_comparison[0]
    regression_time_comparison_1 = regression_time_comparison[1]

    # num_clusters_comparison = []
    # avg_pairwise_distances_clusters_comparison = []

    for i, frontier_pairs in enumerate(frontier_pairs_comparison):
        # num_clusters = []
        # cluster_centers = []
        # avg_pairwise_distances_clusters = []
        for j, frontier_pairs_run in enumerate(frontier_pairs):
            # if len(frontier_pairs_run) > 2:
            #     max_clusters = len(frontier_pairs_run) - 1
            #     clusterer, labels, centers, optimal_score = \
            #         cluster_data(data=frontier_pairs_run, n_clusters_interval=(2, max_clusters))
            #     if len(frontier_pairs_run) == 4 and optimal_score < 0.8:
            #         clusterer, labels, centers, optimal_score = \
            #             cluster_data(data=frontier_pairs_run, n_clusters_interval=(1, max_clusters))
            #     plot_clusters(clusterer=clusterer, data=frontier_pairs_run, points_size=15,
            #                   points_marker='D', save_or_show=False)
            #     num_clusters.append(len(centers))
            #     cluster_centers.append(list(centers))
            # else:
            #     num_clusters.append(len(frontier_pairs_run) / 2)

            if len(frontier_pairs_run) > 0:
                points_marker = '.' if i == 0 else '*'
                plot_file_path = None
                plot_file_path = 'alphatest_frontier_{}_{}.pdf'.format(args.algo_name, j) if i == 0 \
                    else 'random_frontier_{}_{}.pdf'.format(args.algo_name, j)
                plot_file_path = os.path.join(args.save_dir, plot_file_path)

                plot_frontier_points_2D(green_xs=green_xs_comparison[i][j], green_ys=green_ys_comparison[i][j],
                                        red_xs=red_xs_comparison[i][j], red_ys=red_ys_comparison[i][j],
                                        points_size=20, points_marker=points_marker,
                                        plot_file_path=plot_file_path, param_names=args.param_names,
                                        x_lim=first_param_limits, y_lim=second_param_limits)

        #     if len(cluster_centers) > 0 and len(cluster_centers[-1]) > 1:
        #         cluster_centers_ = []
        #         for cluster_center in cluster_centers[-1]:
        #             cluster_centers_.append(list(cluster_center))
        #         pairwise_distances_clusters = pdist(np.asarray(cluster_centers_))
        #         assert pairwise_distances_clusters.max() <= max_distance, \
        #             'Max pairwise distance at iteration {}, {} > max distance {}'.format(
        #                 j, pairwise_distances_clusters.max(), max_distance
        #             )
        #         normalized_pairwise_distances_clusters = pairwise_distances_clusters / max_distance
        #         avg_pairwise_distances_clusters.append(normalized_pairwise_distances_clusters.mean())
        #     else:
        #         avg_pairwise_distances_clusters.append(0.0)
        #
        # num_clusters_comparison.append(num_clusters)
        # avg_pairwise_distances_clusters_comparison.append(avg_pairwise_distances_clusters)

    # num_clusters_comparison_0 = num_clusters_comparison[0]
    # num_clusters_comparison_1 = num_clusters_comparison[1]
    # avg_pairwise_distances_clusters_comparison_0 = avg_pairwise_distances_clusters_comparison[0]
    # avg_pairwise_distances_clusters_comparison_1 = avg_pairwise_distances_clusters_comparison[1]

    # logger.info('******* number of iterations comparison *******')
    # compute_statistics(a=number_of_iterations_comparison_0, b=number_of_iterations_comparison_1, _logger=logger, only_summary=True)

    logger.info('******* search points comparison *******')
    compute_statistics(a=search_points_comparison_0, b=search_points_comparison_1, _logger=logger, only_summary=True)

    logger.info('******* frontier points comparison *******')
    compute_statistics(a=num_frontier_points_comparison_0,
                       b=num_frontier_points_comparison_1,
                       _logger=logger,
                       the_higher_the_better=True)

    # logger.info('******* frontier clusters comparison *******')
    # compute_statistics(a=num_clusters_comparison[0],
    #                    b=num_clusters_comparison[1],
    #                    _logger=logger,
    #                    the_higher_the_better=True)

    logger.info('******* avg pairwise distances comparison *******')
    compute_statistics(a=avg_pairwise_distances_comparison_0,
                       b=avg_pairwise_distances_comparison_1,
                       _logger=logger,
                       the_higher_the_better=True)

    # logger.info('******* sparseness comparison *******')
    # compute_statistics(a=avg_pairwise_distances_clusters_comparison_0,
    #                    b=avg_pairwise_distances_clusters_comparison_1,
    #                    _logger=logger,
    #                    the_higher_the_better=True)

    logger.info('******* time elapsed per run [s] comparison *******')
    compute_statistics(a=time_elapsed_per_run_comparison_0,
                       b=time_elapsed_per_run_comparison_1,
                       _logger=logger,
                       only_summary=True)

    logger.info('******* time taken per repetition [s] comparison *******')
    compute_statistics(a=time_taken_comparison_0,
                       b=time_taken_comparison_1,
                       _logger=logger)

    logger.info('******* dists comparison *******')
    compute_statistics(a=dists_comparison_0,
                       b=dists_comparison_1,
                       _logger=logger,
                       only_summary=True)

    logger.info('******* regression time statistics *******')
    compute_statistics(a=regression_time_comparison_0,
                       b=regression_time_comparison_1,
                       _logger=logger,
                       only_summary=True)

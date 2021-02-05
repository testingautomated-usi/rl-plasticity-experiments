import argparse
import copy
import glob
import logging
import os
import time

import numpy as np
import yaml

from algo.archive import read_saved_archive
from env_utils import standardize_env_name
from log import Log
from plot.nd_approximator import MultiDimensionalApproximator
from utilities import (HOME, SUPPORTED_ALGOS, SUPPORTED_ENVS,
                       check_file_existence, check_param_names,
                       filter_resampling_artifacts,
                       get_result_dir_iteration_number,
                       get_result_file_iteration_number)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=check_file_existence, required=True)
    parser.add_argument(
        "--algo_name", choices=SUPPORTED_ALGOS, required=True, default="ppo2",
    )
    parser.add_argument("--normalize_limits", action="store_true", default=False)
    parser.add_argument("--grid_granularity_percentage_of_range", type=float, required=True)
    parser.add_argument("--env_name", choices=SUPPORTED_ENVS, required=True, default="CartPole-v1")
    parser.add_argument("--param_names", type=check_param_names, required=True)
    parser.add_argument("--regression_probability", action="store_true", default=False)
    parser.add_argument("--no_approximate_nearest_neighbor", action="store_false")
    parser.add_argument("--show_plot", type=bool, default=False)
    parser.add_argument("--plot_nn", type=bool, default=False)
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--plot_file_path", type=check_file_existence, default=None)
    parser.add_argument("--num_iterations_to_consider", type=int, default=None)
    parser.add_argument("--perplexity", type=int, default=None)
    parser.add_argument("--only_tsne", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--model_suffix", type=str, default=None)
    parser.add_argument("--no_filter_model_architecture", action="store_true")
    parser.add_argument("--plot_only_approximated", action="store_true")
    parser.add_argument("--max_points_x", type=int, default=None)
    parser.add_argument("--skip_points_x", type=int, default=None)
    parser.add_argument("--max_points_y", type=int, default=None)
    parser.add_argument("--skip_points_y", type=int, default=None)
    parser.add_argument("--n_iterations_dim_reduction", type=int, default=10000)

    args, _ = parser.parse_known_args()

    if args.model_suffix == "False":
        args.model_suffix = None

    if args.only_tsne:
        assert args.perplexity, "Parameter perplexity must be set"

    nd_approximator = MultiDimensionalApproximator(
        algo_name=args.algo_name, env_name=args.env_name, model_suffix=args.model_suffix
    )
    adapt_volumes_after_nn = []
    regress_volumes_after_nn = []
    adapt_volumes_after_approximation = []
    regress_volumes_after_approximation = []
    adapt_collisions_percentages = []
    regress_collisions_percentages = []
    adapt_execution_times = []
    regress_execution_times = []
    adapt_frontier_pairs_in_grid = []
    regress_frontier_pairs_in_grid = []
    n_clusters_all = []

    filename = os.path.join(
        args.dir,
        ("analyze_volume_results_" if not args.only_tsne else "analyze_volume_results_tsne_")
        + "adapt_regress_probability"
        + ("_" + str(args.num_iterations_to_consider) if args.num_iterations_to_consider else "")
        + "_g_"
        + str(args.grid_granularity_percentage_of_range)
        + ".txt",
    )

    logger = Log("analyze_volume_results")
    logging.basicConfig(filename=filename, filemode="w", level=logging.DEBUG)
    logger.info("args: {}".format(args))

    limits_dict = dict()
    env_params_dir = os.path.abspath("{}/env_params/{}".format(HOME, standardize_env_name(args.env_name)))
    list_of_config_files = (
        glob.glob(os.path.join(env_params_dir, "*.yml"))
        if args.normalize_limits
        else glob.glob(os.path.join(env_params_dir, "{}.yml").format(args.algo_name))
    )
    low_limits_dict = dict()
    high_limits_dict = dict()
    for config_file in list_of_config_files:
        algo_name = config_file[config_file.rindex("/") + 1 : config_file.rindex(".")]
        if not args.no_filter_model_architecture:
            if "_big" in algo_name or "_small" in algo_name:
                continue
        with open(config_file, "r") as f:
            env_params = yaml.safe_load(f)
            for param_name, values_dict in env_params.items():
                if param_name in args.param_names:
                    if param_name not in limits_dict:
                        limits_dict[param_name] = []

                    if param_name not in low_limits_dict:
                        low_limits_dict[param_name] = []
                    low_limits_dict[param_name].append(values_dict["low_limit"])

                    if param_name not in high_limits_dict:
                        high_limits_dict[param_name] = []
                    high_limits_dict[param_name].append(values_dict["high_limit"])

    for param_name, low_limits in low_limits_dict.items():
        limits_dict[param_name].append(np.min(np.array(low_limits)))
    for param_name, high_limits in high_limits_dict.items():
        limits_dict[param_name].append(np.min(np.array(high_limits)))

    new_limits_dict = copy.deepcopy(limits_dict)
    for param_name, limits in limits_dict.items():
        if limits_dict[param_name][0] > limits_dict[param_name][1]:
            new_limits_dict[param_name] = [limits_dict[param_name][1], limits_dict[param_name][0]]
    limits_dict = copy.deepcopy(new_limits_dict)
    logger.info("Limits: {}".format(limits_dict))
    list_of_iterations = glob.glob(os.path.join(args.dir, "n_iterations_*"))
    sorted_dirs = sorted(list_of_iterations, key=get_result_dir_iteration_number)
    for i, iteration_dir in enumerate(sorted_dirs):
        iteration_dir = os.path.join(args.dir, iteration_dir)
        logger.info("Analyzing folder {}".format(iteration_dir))
        if os.path.isdir(iteration_dir):
            list_of_buffer_files = glob.glob(os.path.join(iteration_dir, "buffer_predicate_pairs_*.txt"))
            buffer_files_resampling = glob.glob(os.path.join(iteration_dir, "buffer_predicate_pairs_*_resampling.txt"))
            buffer_file_resampling = None
            if len(buffer_files_resampling) > 0:
                buffer_file_resampling = buffer_files_resampling[0]
            list_of_buffer_files = filter_resampling_artifacts(files=list_of_buffer_files)
            list_of_archive_files = glob.glob(os.path.join(iteration_dir, "frontier_*.txt"))
            list_of_archive_files = filter_resampling_artifacts(files=list_of_archive_files)
            if args.num_iterations_to_consider:
                # buffer
                matching_files = list(
                    filter(
                        lambda buffer_file: buffer_file.endswith("_" + str(args.num_iterations_to_consider) + ".txt"),
                        list_of_buffer_files,
                    )
                )
                assert len(matching_files) == 1, "There should be only one match. Found {}, {}".format(
                    len(matching_files), matching_files
                )
                last_buffer_file = matching_files[0]
                # archive
                matching_files = list(
                    filter(
                        lambda buffer_file: buffer_file.endswith("_" + str(args.num_iterations_to_consider) + ".txt"),
                        list_of_archive_files,
                    )
                )
                assert len(matching_files) == 1, "There should be only one match. Found {}, {}".format(
                    len(matching_files), matching_files
                )
                last_archive_file = matching_files[0]
            else:
                # last buffer file is the one written the latest in time
                last_buffer_file = max(list_of_buffer_files, key=get_result_file_iteration_number)
                # last archive file is the one written the latest in time
                last_archive_file = max(list_of_archive_files, key=get_result_file_iteration_number)
            logger.info("Taking buffer file: {}".format(last_buffer_file))
            logger.info("Taking archive file: {}".format(last_archive_file))
            suffix = "heatmap_adaptation_probability_iteration_g_{}_".format(args.grid_granularity_percentage_of_range)
            if args.plot_file_path:
                plot_file_path = args.plot_file_path + "/" + suffix + str(i)
                if len(args.param_names) > 2:
                    plot_file_path = args.plot_file_path + "/tsne_frontier_points_iteration_" + str(i)
            else:
                plot_file_path = args.plot_file_path
            start_time = time.time()
            if not buffer_file_resampling:
                buffer_file_resampling = last_buffer_file

            logger.info("++++++++++ Adaptation probability ++++++++++")
            (
                adapt_volume_after_nn,
                adapt_volume_after_approximation,
                adapt_number_of_collisions,
                adapt_frontier_pairs_collided,
                index_frontier_not_adapted,
                index_frontier_not_adapted_appr,
                n_clusters,
            ) = nd_approximator.compute_probability_volume(
                buffer_file=buffer_file_resampling,
                last_buffer_file=last_buffer_file,
                archive_file=last_archive_file,
                grid_granularity_percentage_of_range=args.grid_granularity_percentage_of_range,
                param_names_to_consider=args.param_names,
                regression_probability=False,
                approximate_nearest_neighbor=args.no_approximate_nearest_neighbor,
                plot_only_approximated=args.plot_only_approximated,
                show_plot=args.show_plot,
                plot_file_path=plot_file_path,
                limits_dict=limits_dict,
                plot_nn=args.plot_nn,
                smooth=args.smooth,
                perplexity=args.perplexity,
                only_tsne=args.only_tsne,
                max_points_x=args.max_points_x,
                skip_points_x=args.skip_points_x,
                max_points_y=args.max_points_y,
                skip_points_y=args.skip_points_y,
                n_iterations_dim_reduction=args.n_iterations_dim_reduction,
            )
            n_clusters_all.append(n_clusters)
            adapt_execution_times.append(time.time() - start_time)
            adapt_collisions_percentages.append(adapt_number_of_collisions)
            adapt_frontier_pairs_in_grid.append(
                int(len(read_saved_archive(archive_file=last_archive_file)) / 2) - adapt_frontier_pairs_collided
            )
            if adapt_volumes_after_approximation:
                adapt_volumes_after_approximation.append(adapt_volume_after_approximation)
            adapt_volumes_after_nn.append(adapt_volume_after_nn)

            suffix = "heatmap_regression_probability_iteration_g_{}_".format(args.grid_granularity_percentage_of_range)
            plot_file_path = args.plot_file_path + "/" + suffix + str(i)

            if args.regression_probability and not args.only_tsne:
                start_time = time.time()
                logger.info("++++++++++ Regression probability ++++++++++")
                (
                    regress_volume_after_nn,
                    regress_volume_after_approximation,
                    regress_number_of_collisions,
                    regress_frontier_pairs_collided,
                    _,
                    _,
                    _,
                ) = nd_approximator.compute_probability_volume(
                    buffer_file=buffer_file_resampling,
                    last_buffer_file=last_buffer_file,
                    archive_file=last_archive_file,
                    grid_granularity_percentage_of_range=args.grid_granularity_percentage_of_range,
                    param_names_to_consider=args.param_names,
                    regression_probability=True,
                    approximate_nearest_neighbor=args.no_approximate_nearest_neighbor,
                    plot_only_approximated=args.plot_only_approximated,
                    show_plot=args.show_plot,
                    plot_file_path=plot_file_path,
                    limits_dict=limits_dict,
                    plot_nn=args.plot_nn,
                    smooth=args.smooth,
                    perplexity=args.perplexity,
                    only_tsne=False,
                    max_points_x=args.max_points_x,
                    skip_points_x=args.skip_points_x,
                    max_points_y=args.max_points_y,
                    skip_points_y=args.skip_points_y,
                    indices_frontier_not_adapted=index_frontier_not_adapted,
                    indices_frontier_not_adapted_appr=index_frontier_not_adapted_appr,
                    n_iterations_dim_reduction=args.n_iterations_dim_reduction,
                )
                regress_execution_times.append(time.time() - start_time)
                regress_collisions_percentages.append(regress_number_of_collisions)
                regress_frontier_pairs_in_grid.append(
                    int(len(read_saved_archive(archive_file=last_archive_file)) / 2) - regress_frontier_pairs_collided
                )
                if regress_volumes_after_approximation:
                    regress_volumes_after_approximation.append(regress_volume_after_approximation)
                regress_volumes_after_nn.append(regress_volume_after_nn)
            # break

    if not args.only_tsne:
        logger.info("********** Adaptation probability summary **********")
        logger.info(
            "Adaptation collisions percentages: {} +- {}".format(
                np.mean(np.array(adapt_collisions_percentages)), np.std(np.array(adapt_collisions_percentages))
            )
        )
        logger.info(
            "Adaptation volume after nn: {} +- {}".format(
                np.mean(np.array(adapt_volumes_after_nn)), np.std(np.array(adapt_volumes_after_nn))
            )
        )
        if len(adapt_volumes_after_approximation) > 0:
            logger.info(
                "Adaptation volume after approximation: {} +- {}".format(
                    np.mean(np.array(adapt_volumes_after_approximation)), np.std(np.array(adapt_volumes_after_approximation))
                )
            )
            logger.info(
                "Adaptation volume diff: {} +- {}".format(
                    np.mean(np.abs(np.array(adapt_volumes_after_approximation) - np.array(adapt_volumes_after_nn))),
                    np.std(np.abs(np.array(adapt_volumes_after_approximation) - np.array(adapt_volumes_after_nn))),
                )
            )
        logger.info(
            "Time taken for adaptation volume comp. [s]: {} +- {}".format(
                np.mean(np.array(adapt_execution_times)), np.std(np.array(adapt_execution_times))
            )
        )
        logger.info(
            "Frontier pairs in adaptation grid: {} +- {}".format(
                np.mean(np.array(adapt_frontier_pairs_in_grid)), np.std(np.array(adapt_frontier_pairs_in_grid))
            )
        )

    if args.regression_probability:
        logger.info("********** Regression probability summary **********")
        logger.info(
            "Regression collisions percentages: {} +- {}".format(
                np.mean(np.array(regress_collisions_percentages)), np.std(np.array(regress_collisions_percentages))
            )
        )
        logger.info(
            "Regression volume after nn: {} +- {}".format(
                np.mean(np.array(regress_volumes_after_nn)), np.std(np.array(regress_volumes_after_nn))
            )
        )
        if len(regress_volumes_after_approximation) > 0:
            logger.info(
                "Regression volume after approximation: {} +- {}".format(
                    np.mean(np.array(regress_volumes_after_approximation)),
                    np.std(np.array(regress_volumes_after_approximation)),
                )
            )
            logger.info(
                "Adaptation volume diff: {} +- {}".format(
                    np.mean(np.abs(np.array(regress_volumes_after_approximation) - np.array(regress_volumes_after_nn))),
                    np.std(np.abs(np.array(regress_volumes_after_approximation) - np.array(regress_volumes_after_nn))),
                )
            )
        logger.info(
            "Time taken for regression volume comp. [s]: {} +- {}".format(
                np.mean(np.array(regress_execution_times)), np.std(np.array(regress_execution_times))
            )
        )
        logger.info(
            "Frontier pairs in regression grid: {} +- {}".format(
                np.mean(np.array(regress_frontier_pairs_in_grid)), np.std(np.array(regress_frontier_pairs_in_grid))
            )
        )

    if args.only_tsne:
        logger.info("********** TSNE summary **********")
        logger.info("n_clusters: {} +- {}".format(np.mean(np.array(n_clusters_all)), np.std(np.array(n_clusters_all))))

import argparse
import glob
import os
import shutil
import zipfile

from env_utils import standardize_env_name
from utilities import HOME, SUPPORTED_ENVS, SUPPORTED_ALGOS, check_param_names


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", choices=SUPPORTED_ALGOS, type=str)
    parser.add_argument('--env_name', choices=SUPPORTED_ENVS, required=True)
    parser.add_argument('--experiment_type', choices=['alphatest', 'random'])
    parser.add_argument('--num_iterations', type=int, default=0)
    parser.add_argument('--folders_suffix', type=str, default=None)
    parser.add_argument('--analysis_results', action='store_true')
    parser.add_argument('--param_names', type=check_param_names)
    parser.add_argument('--model_suffix', type=str, default=None)

    args, _ = parser.parse_known_args()
    abs_params_dir = os.path.abspath(HOME)

    if args.analysis_results:
        analysis_folder = os.path.join(abs_params_dir, 'rl-experiments-artifacts')
        env_folder = os.path.join(analysis_folder, args.env_name)
        exp_time_folder = os.path.join(env_folder, args.experiment_type)
        param_names_folder = os.path.join(exp_time_folder, '_'.join(args.param_names))
        list_of_folders = glob.glob(os.path.join(param_names_folder, "{}_cluster".format(args.algo_name)))
        zip_name = 'analysis_{}_{}_{}.zip'.format(standardize_env_name(args.env_name), args.algo_name, '_'.join(args.param_names))
    else:
        experiments_folder = os.path.join(abs_params_dir, args.experiment_type)
        env_folder = os.path.join(experiments_folder, args.env_name)
        algo_folder = os.path.join(env_folder, args.algo_name)

        if args.folders_suffix:
            list_of_folders = glob.glob(os.path.join(algo_folder, "n_iterations_{}_{}_*".format(
                args.folders_suffix, args.num_iterations))) if not args.model_suffix \
                else glob.glob(os.path.join(algo_folder, "n_iterations_{}_{}_{}_*".format(args.model_suffix, args.folders_suffix, args.num_iterations)))
            zip_name = '{}_{}_{}_{}.zip'.format(standardize_env_name(args.env_name), args.algo_name, args.folders_suffix, args.num_iterations) if not args.model_suffix \
                else '{}_{}_{}_{}_{}.zip'.format(standardize_env_name(args.env_name), args.algo_name, args.model_suffix, args.folders_suffix, args.num_iterations)
        else:
            list_of_folders = glob.glob(os.path.join(algo_folder, "n_iterations_{}_*".format(args.num_iterations))) if not args.model_suffix \
                else glob.glob(os.path.join(algo_folder, "n_iterations_{}_{}_*".format(args.num_iterations, args.model_suffix)))
            zip_name = '{}_{}_{}.zip'.format(standardize_env_name(args.env_name), args.algo_name, args.num_iterations) if not args.model_suffix \
                else '{}_{}_{}_{}.zip'.format(standardize_env_name(args.env_name), args.algo_name, args.model_suffix, args.num_iterations)

    # zip folder
    with zipfile.ZipFile(os.path.join(HOME, zip_name), 'w', zipfile.ZIP_DEFLATED) as zip_handler:
        for folder in list_of_folders:
            zipdir(folder, zip_handler)
            if not args.analysis_results:
                shutil.rmtree(folder)


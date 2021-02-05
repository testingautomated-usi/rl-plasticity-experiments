from typing import List, Tuple

from matplotlib.colors import LinearSegmentedColormap
from numpy.linalg import LinAlgError

from envs.env_variables import load_env_params
from param import Param
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pandas as pd
import os


def plot_heatmap(env_name: str, algo_name: str, param_names: List,
                 env_pairs_pass_probabilities: List[Tuple[Tuple, float]],
                 first_param_limits: List[float],
                 second_param_limits: List[float],
                 plot_points: bool = False,
                 save_plot: bool = False,
                 show_plot: bool = False,
                 file_path: str = None,
                 interpolation_function: str = 'linear',
                 smooth: float = 0.0) -> float:

    assert interpolation_function \
           in ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'], \
        'interpolation_function should have one of these values: {}'\
            .format(['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'])

    dist = None

    first_param_values = []
    second_param_values = []
    pass_probabilities = []

    for env_pair_pass_probability in env_pairs_pass_probabilities:
        mutated_pair = env_pair_pass_probability[0]
        pass_probability = env_pair_pass_probability[1]
        first_param_values.append(round(mutated_pair[0], 1))
        second_param_values.append(round(mutated_pair[1], 1))
        pass_probabilities.append(pass_probability)

    dict_for_df = {param_names[0]: first_param_values, param_names[1]: second_param_values,
                   'pass_probability': pass_probabilities}
    df = pd.DataFrame(dict_for_df)
    heatmap_data = pd.pivot_table(df, values='pass_probability', index=[param_names[1]], columns=param_names[0])

    first_param = Param(
        **load_env_params(algo_name=algo_name, env_name=env_name, param_name=param_names[0]), id=0,
        name=param_names[0])
    second_param = Param(
        **load_env_params(algo_name=algo_name, env_name=env_name, param_name=param_names[1]), id=1,
        name=param_names[1])

    min_first_param = first_param_limits[0] if len(first_param_limits) == 2 else first_param.get_low_limit()
    max_first_param = first_param_limits[1] if len(first_param_limits) == 2 else first_param.get_high_limit()

    min_second_param = second_param_limits[0] if len(second_param_limits) == 2 else second_param.get_low_limit()
    max_second_param = second_param_limits[1] if len(second_param_limits) == 2 else second_param.get_high_limit()

    extent = [min_first_param, max_first_param, min_second_param, max_second_param]
    # Create regular grid
    xi, yi = np.linspace(min_first_param, max_first_param, heatmap_data.shape[1]), \
             np.linspace(min_first_param, max_second_param, heatmap_data.shape[0])
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate missing data
    matrix_is_singular_error = True
    while matrix_is_singular_error:
        try:
            rbf = Rbf(first_param_values, second_param_values, pass_probabilities, function=interpolation_function,
                      smooth=smooth)
            matrix_is_singular_error = False
        except LinAlgError:
            smooth += 0.1
            print('Matrix is singular: increasing smooth value to', smooth)
            matrix_is_singular_error = True

    zi = rbf(xi, yi)

    if zi.shape == heatmap_data.shape:
        dist = 1 / (1 + np.linalg.norm(zi - heatmap_data.values))

    if show_plot or save_plot:

        fig = plt.figure()
        fig.suptitle(env_name + ', ' + algo_name)

        ax1 = fig.add_subplot(211)
        cmap = LinearSegmentedColormap.from_list(name='test', colors=['red', 'gold', 'green'])
        sns.heatmap(data=heatmap_data, cmap=cmap, linewidths=0.3, cbar_kws={'label': 'Pass probability'})
        ax1.invert_yaxis()

        ax2 = fig.add_subplot(212)
        hm = ax2.imshow(zi, interpolation='none', cmap=cmap, extent=extent, origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
        if plot_points:
            ax2.scatter(first_param_values, second_param_values)
        cbar = fig.colorbar(hm, ax=ax2)
        cbar.ax.set_ylabel('Pass probability')
        ax2.set_xlabel(param_names[0])
        ax2.set_ylabel(param_names[1])

        fig = plt.gcf()
        fig.set_size_inches(12, 9)

        if show_plot:
            plt.show()

        if save_plot:
            if file_path:
                plt.savefig(file_path, format='pdf')
            else:
                abs_prefix = os.path.abspath('../')
                file_prefix = 0
                file_suffix = 'heatmap_' + interpolation_function + '_' + env_name + '_' + algo_name + '_'
                file_name = file_suffix + str(file_prefix) + '.pdf'
                while os.path.exists(os.path.join(abs_prefix, file_name)):
                    file_prefix += 1
                    file_name = file_suffix + str(file_prefix) + '.pdf'

                plt.savefig(os.path.join(abs_prefix, file_name), format='pdf')

    return dist

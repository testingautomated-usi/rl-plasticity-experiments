import os
from typing import List, Tuple, Dict
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.text import Text

from algo.env_predicate_pair import BufferItem
from env_utils import standardize_env_name
from envs.env_variables import load_env_params
from log import Log
from param import Param
from utilities import get_num_digits_after_floating_point, get_num_digits_before_floating_point


def get_num_significant_digits(labels: List[Text] = None, values: List[float] = None) -> int:
    min_number = float(labels[0].get_text() if labels else values[0])
    min_number_digits_before_float = get_num_digits_before_floating_point(number=min_number)
    max_number = float(labels[-1].get_text() if labels else values[-1])
    max_number_digits_before_float = get_num_digits_before_floating_point(number=max_number)

    if min_number_digits_before_float == max_number_digits_before_float and min_number >= 1.0:
        return 2

    nums = []
    digits_after_float = []
    if labels:
        for label in labels:
            num_to_represent = float(label.get_text())
            num_digits_after_float = get_num_digits_after_floating_point(number=num_to_represent)
            digits_after_float.append(num_digits_after_float)
            nums.append(num_to_represent)
    else:
        for value in values:
            num_to_represent = float(value)
            num_digits_after_float = get_num_digits_after_floating_point(number=num_to_represent)
            digits_after_float.append(num_digits_after_float)
            nums.append(num_to_represent)

    if np.asarray(nums).mean() > 1.0:
        return 2

    return min(int(np.asarray(digits_after_float).mean()), 5)


def plot_heatmap(
        buffer: List[BufferItem],
        grid_indices_with_values: dict,
        grid_components: dict,
        tensor_shape: Tuple,
        env_values_repeated: Dict,
        env_values_dict: Dict,
        regression_probability: bool,
        param_names: List[str],
        approximated_tensor: np.ndarray,
        show_plot: bool,
        plot_only_approximated: bool,
        algo_name: str,
        env_name: str,
        interpolation_function: str,
        plot_file_path: str = None,
        plot_nn: bool = False,
        model_suffix: str = None,
        max_points_x: int = None,
        skip_points_x: int = None,
        max_points_y: int = None,
        skip_points_y: int = None,
        indices_frontier_not_adapted_appr: List[Tuple] = None
):
    logger = Log('plot_heatmap')

    env_values_not_resampling = []
    for buffer_item in buffer:
        env_values = []
        for key in buffer_item.get_env_values().keys():
            if key in param_names:
                env_values.append(buffer_item.get_env_values()[key])
        env_values_not_resampling.append(tuple(env_values))

    first_param_values_step, second_param_values_step, probabilities_step = [], [], []
    first_param_components = grid_components[param_names[0]]
    second_param_components = grid_components[param_names[1]]
    values_set = 0
    for i in range(tensor_shape[0]):
        for j in range(tensor_shape[1]):
            mutated_pair = (first_param_components[i], second_param_components[j])
            pass_probability = np.nan
            if (i, j) in grid_indices_with_values:
                pass_probability = grid_indices_with_values[(i, j)][1]
                values_set += 1
            first_param_values_step.append(mutated_pair[0])
            second_param_values_step.append(mutated_pair[1])
            probabilities_step.append(pass_probability)
    assert values_set == len(grid_indices_with_values), '{} != {}'.format(values_set, len(grid_indices_with_values))

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    if plot_only_approximated:
        SMALL_SIZE = 14
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    lsts_values_repeated = list(env_values_repeated.values())
    first_param_values_repeated = lsts_values_repeated[0]
    second_param_values_repeated = lsts_values_repeated[1]

    min_first_param = np.min(np.array(first_param_values_repeated))
    max_first_param = np.max(np.array(first_param_values_repeated))

    min_second_param = np.min(np.array(second_param_values_repeated))
    max_second_param = np.max(np.array(second_param_values_repeated))

    if not plot_only_approximated:
        fig = plt.figure(figsize=(22, 15))
        _ = fig.add_subplot(211)
    else:
        fig = plt.figure(figsize=(9, 8))

    apprx_ax = plt.gca()

    colors = ['red', 'gold', 'green'] if not regression_probability else ['green', 'gold', 'red']
    cmap = LinearSegmentedColormap.from_list(name='test', colors=colors)

    dict_for_df = {
        param_names[0]: first_param_values_step,
        param_names[1]: second_param_values_step,
        'pass_probability': probabilities_step
    }

    df = pd.DataFrame(dict_for_df)
    heatmap_data = pd.pivot_table(df, dropna=False, values='pass_probability',
                                  index=param_names[1], columns=param_names[0])

    first_param = Param(**load_env_params(
        algo_name=algo_name,
        env_name=standardize_env_name(env_name=env_name),
        param_name=param_names[0], model_suffix=model_suffix), id=0, name=param_names[0])
    second_param = Param(**load_env_params(
        algo_name=algo_name,
        env_name=standardize_env_name(env_name=env_name),
        param_name=param_names[1], model_suffix=model_suffix), id=1, name=param_names[0])

    # first_param = x, second_param = y
    direction_x = 'ltr' if first_param.get_starting_multiplier() > 1 and first_param.get_direction() == 'positive' else 'rtl'
    direction_y = 'btt' if second_param.get_starting_multiplier() > 1 and second_param.get_direction() == 'positive' else 'ttb'

    if not plot_only_approximated:

        grid_ax = sns.heatmap(
            data=heatmap_data,
            linewidths=0.2,
            square=False,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Pass probability' if not regression_probability else 'Regression probability'}
        )

        xticks_rounded = []
        num_significant_digits = get_num_significant_digits(labels=grid_ax.get_xticklabels())
        for colorbar_label in grid_ax.get_xticklabels():
            num_to_represent = float(colorbar_label.get_text())
            if abs(num_to_represent) > 1:
                round_label = round(num_to_represent, num_significant_digits)
                xticks_rounded.append(Text(x=colorbar_label._x, y=colorbar_label._y, text=str(round_label)))
            else:
                round_label = round(num_to_represent, num_significant_digits)
                xticks_rounded.append(Text(x=colorbar_label._x, y=colorbar_label._y, text=str(round_label)))

        num_significant_digits = get_num_significant_digits(labels=grid_ax.get_yticklabels())
        yticks_rounded = []
        for colorbar_label in grid_ax.get_yticklabels():
            num_to_represent = float(colorbar_label.get_text())
            if abs(num_to_represent) > 1:
                round_label = round(num_to_represent, num_significant_digits)
                yticks_rounded.append(Text(x=colorbar_label._x, y=colorbar_label._y, text=str(round_label)))
            else:
                round_label = round(num_to_represent, num_significant_digits)
                yticks_rounded.append(Text(x=colorbar_label._x, y=colorbar_label._y, text=str(round_label)))

        if direction_x == 'ltr' and direction_y == 'btt':
            grid_ax.invert_yaxis()
        elif direction_x == 'rtl' and direction_y == 'btt':
            grid_ax.invert_yaxis()
            grid_ax.invert_xaxis()
        elif direction_x == 'ltr' and direction_y == 'ttb':
            raise NotImplementedError()
        elif direction_x == 'rtl' and direction_y == 'ttb':
            raise NotImplementedError()

        grid_ax.set_xticklabels(xticks_rounded, rotation=90)
        grid_ax.set_yticklabels(yticks_rounded)

        apprx_ax = fig.add_subplot(212)

    if plot_nn:
        # suppose to plot filled heatmap
        pass

    lsts_values = list(env_values_dict.values())
    limits_first_param = [grid_components[param_names[0]][0], grid_components[param_names[0]][-1]]
    limits_second_param = [grid_components[param_names[1]][0], grid_components[param_names[1]][-1]]
    first_param_values = []
    second_param_values = []
    first_param_values_resampling = []
    second_param_values_resampling = []
    for i in range(len(lsts_values[0])):
        value_first_param = lsts_values[0][i]
        value_second_param = lsts_values[1][i]
        min_limit_first_param = round(limits_first_param[0], 5)
        max_limit_first_param = round(limits_first_param[1], 5)
        min_limit_second_param = round(limits_second_param[0], 5)
        max_limit_second_param = round(limits_second_param[1], 5)
        if (min_limit_first_param <= value_first_param <= max_limit_first_param) \
                and (min_limit_second_param <= value_second_param <= max_limit_second_param):

            if (value_first_param, value_second_param) not in env_values_not_resampling:
                first_param_values_resampling.append(value_first_param)
                second_param_values_resampling.append(value_second_param)
            else:
                first_param_values.append(value_first_param)
                second_param_values.append(value_second_param)
        else:
            logger.warn('Discarding pair {} from scatterplot because beyond limits [{}. {}], [{}, {}]'.format(
                (value_first_param, value_second_param), min_limit_first_param, max_limit_first_param,
                min_limit_second_param, max_limit_second_param))

    if regression_probability and indices_frontier_not_adapted_appr:
        cmap.colorbar_extend = 'min'
        cmap.set_under('gray')
        for index_frontier_not_adapted_appr in indices_frontier_not_adapted_appr:
            approximated_tensor[tuple(index_frontier_not_adapted_appr)] = -1.0

    extent = [min_first_param, max_first_param, min_second_param, max_second_param]
    hm = apprx_ax.imshow(approximated_tensor, interpolation='none', cmap=cmap, extent=extent, aspect='auto',
                         origin='lower', vmin=0.0, vmax=1.0)

    max_num_points_x = 100 if not max_points_x else max_points_x
    # num_points_to_skip = (len(heatmap_data.axes[1].values) + max_num_points_x // 2) // max_num_points_x
    if skip_points_x is not None:
        num_points_to_skip = skip_points_x
    else:
        num_points_to_skip = 4
    xticks_rounded = []
    num_significant_digits = get_num_significant_digits(values=list(heatmap_data.axes[1].values))
    if num_points_to_skip > 0:
        for i in range(0, len(heatmap_data.axes[1].values), num_points_to_skip):
            xticks_rounded.append(round(heatmap_data.axes[1].values[i], num_significant_digits))
    else:
        for value in heatmap_data.axes[1].values:
            xticks_rounded.append(round(value, num_significant_digits))

    max_num_points_y = 30 if not max_points_y else max_points_y
    # num_points_to_skip = (len(heatmap_data.axes[0].values) + max_num_points_y // 2) // max_num_points_y
    if skip_points_y is not None:
        num_points_to_skip = skip_points_y
    else:
        num_points_to_skip = 4
    yticks_rounded = []
    num_significant_digits = get_num_significant_digits(values=list(heatmap_data.axes[0].values))
    if num_points_to_skip > 0:
        for i in range(0, len(heatmap_data.axes[0].values), num_points_to_skip):
            yticks_rounded.append(round(heatmap_data.axes[0].values[i], num_significant_digits))
    else:
        for value in heatmap_data.axes[0].values:
            yticks_rounded.append(round(value, num_significant_digits))

    apprx_ax.set_xticks(xticks_rounded)
    apprx_ax.set_yticks(yticks_rounded)

    for tick in apprx_ax.get_xticklabels():
        tick.set_rotation(90)

    if direction_x == 'ltr' and direction_y == 'btt':
        pass
    elif direction_x == 'rtl' and direction_y == 'btt':
        apprx_ax.invert_xaxis()
    elif direction_x == 'ltr' and direction_y == 'ttb':
        raise NotImplementedError()
    elif direction_x == 'rtl' and direction_y == 'ttb':
        raise NotImplementedError()

    apprx_ax.scatter(first_param_values, second_param_values, s=50, c='black')
    if len(first_param_values_resampling) > 0:
        apprx_ax.scatter(first_param_values_resampling, second_param_values_resampling, s=100, marker='*', c='black')

    # determine points in the adaptation frontier if regression
    if regression_probability:
        pass

    cbar = fig.colorbar(hm, ax=apprx_ax)
    cbar.ax.set_ylabel('Adaptation probability' if not regression_probability else 'Regression probability')
    apprx_ax.set_xlabel(param_names[0])
    apprx_ax.set_ylabel(param_names[1])

    if show_plot:
        plot_title = 'heatmap_' + interpolation_function + '_' + env_name + '_' + algo_name
        fig.canvas.set_window_title(plot_title)
        plt.show()
    else:
        if plot_file_path:
            plt.savefig(plot_file_path + '.pdf', format='pdf')
        else:
            abs_prefix = os.path.abspath('../')
            file_prefix = 0
            file_suffix = 'heatmap_' + interpolation_function + '_' + env_name + '_' + algo_name + '_'
            file_name = file_suffix + str(file_prefix) + '.pdf'
            while os.path.exists(os.path.join(abs_prefix, file_name)):
                file_prefix += 1
                file_name = file_suffix + str(file_prefix) + '.pdf'

            plt.savefig(os.path.join(abs_prefix, file_name), format='pdf')




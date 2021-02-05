from typing import Tuple, List, Dict

from numba import njit, prange, objmode
from scipy.interpolate import Rbf

from algo.archive import read_saved_archive, compute_dist_values, is_in_archive
from algo.env_predicate_pair import read_saved_buffer, BufferItem
import numpy as np
import time
import math
import copy

from log import Log
from plot.heatmap import plot_heatmap
from plot.plot_frontier_points import plot_frontier_points_high_dim


def count_digits(number: float) -> Tuple[int, int]:
    number = abs(number)
    return str(number).find('.'), str(number)[::-1].find('.')


def get_env_values_probabilities_archive(
        archive: List[Tuple[Dict, bool]],
        param_names: List[str]
) -> List[Tuple[List, float]]:
    env_values_probabilities = []
    for i, archive_item in enumerate(archive):
        tuples_param_name_value = archive_item[0].items()
        env_values = []
        for tuple_param_name_value in tuples_param_name_value:
            if tuple_param_name_value[0] in param_names:
                env_values.append(tuple_param_name_value[1])
        probability = archive_item[1]
        env_values_probabilities.append((env_values, probability))
    return env_values_probabilities


def get_env_values_probabilities(
        buffer: List[BufferItem],
        param_names: List[str],
        regression_probability: bool,
) -> List[Tuple[List, float]]:
    env_values_probabilities = []
    for i, buffer_item in enumerate(buffer):
        tuples_param_name_value = buffer_item.get_env_values().items()
        env_values = []
        for tuple_param_name_value in tuples_param_name_value:
            if tuple_param_name_value[0] in param_names:
                env_values.append(tuple_param_name_value[1])
        if regression_probability:
            probability = buffer_item.get_regression_probability()
        else:
            probability = buffer_item.get_pass_probability()
        # discard first search point since it is the original env and regression_probability is not meaningful;
        # and discard as well points whose predicate is False for the same reason
        if regression_probability:
            if i == 0:
                env_values_probabilities.append((env_values, 0.0))
            elif buffer_item.is_predicate():
                env_values_probabilities.append((env_values, probability))
        else:
            env_values_probabilities.append((env_values, probability))
    return env_values_probabilities


def get_env_values_dict(env_values_probabilities: List[Tuple[List, float]], param_names: List[str]) -> Dict:
    env_values_dict = dict()
    for env_value_probability_step in env_values_probabilities:
        env_values = env_value_probability_step[0]
        for i, env_value in enumerate(env_values):
            param_name = param_names[i]
            if param_name not in env_values_dict:
                env_values_dict[param_name] = []
            env_values_dict[param_name].append(env_value)
    return env_values_dict


def compute_grid(env_values_dict: Dict, grid_granularity_percentage_of_range: float, limits_dict: Dict) -> Tuple[
    Dict, Tuple]:
    grid_dims = []
    grid = dict()
    for key, values in env_values_dict.items():
        values_set = set(values)
        unique_values = list(values_set)
        if limits_dict:
            max_value = limits_dict[key][1]
        else:
            max_value = np.max(np.array(unique_values))

        if limits_dict:
            min_value = limits_dict[key][0]
        else:
            min_value = np.min(np.array(unique_values))

        step_size = (max_value - min_value) * grid_granularity_percentage_of_range / 100
        if key not in grid:
            grid[key] = []
        values_in_grid = list(np.arange(min_value, max_value + step_size, step_size))
        grid[key] = values_in_grid
        grid_dims.append(len(values_in_grid))
    return grid, tuple(grid_dims)


def fill_grid_with_search_values(
        grid_indices_with_values: Dict,
        tensor: np.ndarray,
) -> np.ndarray:
    sum_probabilities_step = 0.0

    for grid_index in grid_indices_with_values.keys():
        assert len(grid_index) == len(tensor.shape), \
            'Grid index does not have the same num of dimensions as the grid. {} != {}'.format(
                len(grid_index), len(tensor.shape))
        sum_probabilities_step += grid_indices_with_values[grid_index][1]
        tensor[grid_index] = grid_indices_with_values[grid_index][1]

    sum_tensor = np.nansum(tensor)
    assert math.isclose(sum_tensor, sum_probabilities_step, abs_tol=0.1), 'The two sums must match: {}, {}' \
        .format(sum_tensor, sum_probabilities_step)

    return tensor

def above_threshold(p1: float, p2: float) -> bool:
    if p1 > 0.5 and p2 > 0.5:
        return True
    return False


def above_or_below_threshold(p1: float, p2: float) -> bool:
    if p1 > 0.5 and p2 > 0.5:
        return True
    if p1 < 0.5 and p2 < 0.5:
        return True
    return False


def adjacent_points(index1: Tuple[int, int], index2: Tuple[int, int], dim: int) -> bool:
    return abs(index1[dim] - index2[dim]) == 1


def resampling(
        tensor: np.ndarray,
        grid_indices_with_values: Dict,
        grid_components: Dict,
        param_names: List[str],
) -> np.ndarray:
    tensor = np.array(tensor)

    boolean_mask_finite = np.isfinite(tensor)
    nd_indices_finite = np.where(boolean_mask_finite)
    del boolean_mask_finite
    indices_finite = np.asarray(list(zip(*nd_indices_finite)))
    del nd_indices_finite
    num_dimensions = len(tensor.shape)
    indices_per_dimension = dict()

    for num_dim in range(num_dimensions):
        indices_per_dimension[num_dim] = []

    for i, index_finite in enumerate(indices_finite):
        for num_dim in range(num_dimensions):
            start_dim_index = index_finite[num_dim]
            for j in range(i + 1, len(indices_finite)):
                next_index = indices_finite[j]
                others_all_equals = True
                for k in range(len(index_finite)):
                    if k != num_dim:
                        if index_finite[k] != next_index[k]:
                            others_all_equals = False
                            break
                if next_index[num_dim] - start_dim_index > 0 and others_all_equals:
                    indices_per_dimension[num_dim].append([tuple(index_finite), tuple(next_index)])

    del indices_finite

    indices_to_update = dict()
    for key in indices_per_dimension.keys():
        indices_dim = indices_per_dimension[key]
        for indices_pair in indices_dim:
            p1 = tensor[tuple(indices_pair[0])]
            p2 = tensor[tuple(indices_pair[1])]
            if not adjacent_points(index1=indices_pair[0], index2=indices_pair[1], dim=key) \
                    and above_or_below_threshold(p1=p1, p2=p2):
                index_down = indices_pair[0][key]
                index_up = indices_pair[1][key]
                index_middle = math.ceil((index_up + index_down) / 2)
                new_index = list(copy.deepcopy(indices_pair[0]))
                new_index[key] = index_middle
                if tuple(new_index) not in indices_to_update:
                    if np.isnan(tensor[tuple(new_index)]):
                        # tensor[tuple(new_index)] = (p1 + p2) / 2
                        indices_to_update[tuple(new_index)] = (p1 + p2) / 2
    del indices_per_dimension

    for key in indices_to_update.keys():
        tensor[key] = indices_to_update[key]
        env_values = []
        for index in range(len(key)):
            env_values.append(grid_components[param_names[index]])
        grid_indices_with_values[tuple(key)] = (tuple(env_values), indices_to_update[key])

    del indices_to_update

    return tensor


@njit(nopython=True)
def cartesian_product(arrays: List[Tuple]):
    n = 1
    for x in arrays:
        n *= len(x)
    out = np.zeros((n, len(arrays)), dtype=np.int32)

    for i in range(len(arrays)):
        m = int(n / len(arrays[i]))
        out[:n, i] = np.repeat(arrays[i], m)
        n //= len(arrays[i])

    n = len(arrays[-1])
    for k in range(len(arrays)-2, -1, -1):
        n *= len(arrays[k])
        m = int(n / len(arrays[k]))
        for j in range(1, len(arrays[k])):
            out[j*m:(j+1)*m, k+1:] = out[0:m, k+1:]
    return out


@njit(nopython=True)
def compute_neighbourhood_values(index, tensor_shape, tensor) -> Tuple[List[int], List[float]]:
    neighbors = []
    index_nan = list(index)
    plus_minus = []
    for i in range(len(index_nan)):
        d = index_nan[i]
        if d == 0:
            plus_minus.append([d, d + 1])
        elif d == tensor_shape[i] - 1:
            plus_minus.append([d, d - 1])
        else:
            plus_minus.append([d, d + 1, d - 1])

    combinations = list(cartesian_product(plus_minus))
    combinations.pop(0)

    for i in range(len(combinations)):
        combination = combinations[i]
        # if not np.isnan(tensor[tuple(combination)]):
        #     neighbors.append(tensor[tuple(combination)])

        # numba does not support tuple
        if len(tensor_shape) == 2:
            if not np.isnan(tensor[(combination[0], combination[1])]):
                neighbors.append(tensor[(combination[0], combination[1])])
        elif len(tensor_shape) == 3:
            if not np.isnan(tensor[(combination[0], combination[1], combination[2])]):
                neighbors.append(tensor[(combination[0], combination[1], combination[2])])
        elif len(tensor_shape) == 4:
            if not np.isnan(tensor[(combination[0], combination[1], combination[2], combination[3])]):
                neighbors.append(tensor[(combination[0], combination[1], combination[2], combination[3])])

    return index_nan, neighbors


@njit(nopython=True)
def compute_neighbourhood_indices(index, tensor_shape, tensor) -> Tuple[List[int], List[List[int]]]:
    index_nan = list(index)
    plus_minus = []
    for i in range(len(index_nan)):
        d = index_nan[i]
        if d == 0:
            plus_minus.append([d, d + 1])
        elif d == tensor_shape[i] - 1:
            plus_minus.append([d, d - 1])
        else:
            plus_minus.append([d, d + 1, d - 1])

    combinations = list(cartesian_product(plus_minus))
    combinations.pop(0)

    filtered_combinations = []
    for i in range(len(combinations)):
        combination = combinations[i]
        # if np.isnan(tensor[tuple(combination)]):
        #     filtered_combinations.append(combination)

        # numba does not support tuple
        if len(tensor_shape) == 2:
            if np.isnan(tensor[(combination[0], combination[1])]):
                filtered_combinations.append(combination)
        elif len(tensor_shape) == 3:
            if np.isnan(tensor[(combination[0], combination[1], combination[2])]):
                filtered_combinations.append(combination)
        elif len(tensor_shape) == 4:
            if np.isnan(tensor[(combination[0], combination[1], combination[2], combination[3])]):
                filtered_combinations.append(combination)

    return index_nan, filtered_combinations


@njit(parallel=True, nopython=True)
def nearest_neighbor_parallel(
        tensor: np.ndarray,
        tensor_shape: Tuple,
) -> np.ndarray:

    assert len(tensor_shape) <= 4, 'Dimensions greater than 4 not supported since numba does not support tuple'

    is_nan_present = True
    num_iterations = 0
    while is_nan_present:
        is_nan_present = False

        boolean_mask_nan = np.isnan(tensor)
        nd_indices_nan = np.where(boolean_mask_nan)
        indices_nan = list(zip(*nd_indices_nan))

        if len(indices_nan) > 0:
            is_nan_present = True
            batches = np.zeros(shape=tensor.shape)
            batches[...] = np.nan

            for i in prange(len(indices_nan)):
                index_nan = indices_nan[i]
                index, neighbors = compute_neighbourhood_values(
                    index=index_nan, tensor_shape=tensor_shape, tensor=tensor
                )
                mean = np.nan
                sum_ = 0.0
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        sum_ += neighbor
                    mean = sum_ / len(neighbors)
                if len(tensor_shape) == 2:
                    batches[(index[0], index[1])] = mean
                elif len(tensor_shape) == 3:
                    batches[(index[0], index[1], index[2])] = mean
                elif len(tensor_shape) == 4:
                    batches[(index[0], index[1], index[2], index[3])] = mean

            # update in batch
            # print('batches: {}, {}'.format(len(batches), batches[0]))

            boolean_mask_not_nan = ~np.isnan(batches)
            nd_indices_not_nan = np.where(boolean_mask_not_nan)
            indices_not_nan = list(zip(*nd_indices_not_nan))

            for i in prange(len(indices_not_nan)):
                index_not_nan = indices_not_nan[i]
                if len(tensor_shape) == 2:
                    tensor[(index_not_nan[0], index_not_nan[1])] = batches[(index_not_nan[0], index_not_nan[1])]
                elif len(tensor_shape) == 3:
                    tensor[(index_not_nan[0], index_not_nan[1], index_not_nan[2])] \
                        = batches[(index_not_nan[0], index_not_nan[1], index_not_nan[2])]
                elif len(tensor_shape) == 4:
                    tensor[(index_not_nan[0], index_not_nan[1], index_not_nan[2], index_not_nan[3])] \
                        = batches[(index_not_nan[0], index_not_nan[1], index_not_nan[2], index_not_nan[3])]

        num_iterations += 1

    return tensor


@njit(nopython=True)
def nearest_neighbor(
        tensor: np.ndarray,
        tensor_shape: Tuple,
) -> np.ndarray:

    assert len(tensor_shape) <= 4, 'Dimensions greater than 4 not supported since numba does not support tuple'

    is_nan_present = True
    num_iterations = 0
    while is_nan_present:
        is_nan_present = False

        boolean_mask_nan = np.isnan(tensor)
        nd_indices_nan = np.where(boolean_mask_nan)
        indices_nan = list(zip(*nd_indices_nan))

        if len(indices_nan) > 0:
            is_nan_present = True
            batches = np.zeros(shape=tensor.shape)
            batches[...] = np.nan

            for i in range(len(indices_nan)):
                index_nan = indices_nan[i]
                index, neighbors = compute_neighbourhood_values(
                    index=index_nan, tensor_shape=tensor_shape, tensor=tensor
                )
                mean = np.nan
                sum_ = 0.0
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        sum_ += neighbor
                    mean = sum_ / len(neighbors)
                if len(tensor_shape) == 2:
                    batches[(index[0], index[1])] = mean
                elif len(tensor_shape) == 3:
                    batches[(index[0], index[1], index[2])] = mean
                elif len(tensor_shape) == 4:
                    batches[(index[0], index[1], index[2], index[3])] = mean

            boolean_mask_not_nan = ~np.isnan(batches)
            nd_indices_not_nan = np.where(boolean_mask_not_nan)
            indices_not_nan = list(zip(*nd_indices_not_nan))

            for i in range(len(indices_not_nan)):
                index_not_nan = indices_not_nan[i]
                if len(tensor_shape) == 2:
                    tensor[(index_not_nan[0], index_not_nan[1])] = batches[(index_not_nan[0], index_not_nan[1])]
                elif len(tensor_shape) == 3:
                    tensor[(index_not_nan[0], index_not_nan[1], index_not_nan[2])] \
                        = batches[(index_not_nan[0], index_not_nan[1], index_not_nan[2])]
                elif len(tensor_shape) == 4:
                    tensor[(index_not_nan[0], index_not_nan[1], index_not_nan[2], index_not_nan[3])] \
                        = batches[(index_not_nan[0], index_not_nan[1], index_not_nan[2], index_not_nan[3])]

        num_iterations += 1

    return tensor


INTERPOLATION_FUNCTION = 'linear'


def approximate(
        grid_components: Dict,
        tensor: np.ndarray,
        tensor_shape: Tuple,
        plot_nn: False,
        smooth: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    lists_of_env_values = list(grid_components.values())
    meshgrid_components = np.meshgrid(*lists_of_env_values)
    assert np.multiply.reduce([len(values) for values in lists_of_env_values]) == len(tensor.flatten()), \
        'The two dimensions must match: {}, {}' \
            .format(np.multiply.reduce([len(values) for values in lists_of_env_values]), len(tensor.flatten()))

    it = np.nditer(tensor, flags=['multi_index'])
    env_values_repeated = dict()
    while not it.finished:
        indices = it.multi_index
        for index_param in range(len(tensor_shape)):
            if index_param not in env_values_repeated:
                env_values_repeated[index_param] = []
            env_values_repeated[index_param].append(lists_of_env_values[index_param][indices[index_param]])
        it.iternext()

    for len_values in [len(values) for values in env_values_repeated.values()]:
        assert len_values == len(tensor.flatten()), 'The two dimensions must match: {}, {}' \
            .format(len_values, len(tensor.flatten()))

    if plot_nn:
        rbf_tensor = np.ones(shape=tensor_shape)
    else:
        rbf = Rbf(*env_values_repeated.values(), tensor.flatten(), smooth=smooth, function=INTERPOLATION_FUNCTION)
        rbf_tensor = rbf(*meshgrid_components)

    return rbf_tensor, env_values_repeated


class MultiDimensionalApproximator:

    def __init__(self, algo_name: str, env_name: str, model_suffix: str = None):
        self.logger = Log('MultiDimensionalApproximator')
        self.algo_name = algo_name
        self.env_name = env_name
        self.model_suffix = model_suffix

    def adapting_search_values_to_grid(
            self,
            env_values_probabilities: List[Tuple[List, float]],
            grid_components: Dict,
            param_names: List[str],
            archive: List[Tuple[Dict, bool]],
    ) -> Tuple[dict, int, int]:

        indices_with_value = dict()
        start_time = time.time()
        frontier_pairs_collided = 0

        original_sum_of_probabilities = 0.0
        for env_values_probability in env_values_probabilities:
            env_values = env_values_probability[0]
            probability = env_values_probability[1]
            original_sum_of_probabilities += probability
            index = []
            for i, env_value in enumerate(env_values):
                grid_ith_dimension = grid_components[param_names[i]]
                if round(grid_ith_dimension[0], 5) <= env_value <= round(grid_ith_dimension[-1], 5):
                    index_ith_dimension = int(round(
                        (len(grid_ith_dimension) - 1) * (env_value - grid_ith_dimension[0]) / (
                                grid_ith_dimension[-1] - grid_ith_dimension[0])))
                    index.append(index_ith_dimension)
                else:
                    self.logger.warn('Env value {} for param {} is beyond limits [{}, {}]'
                                     .format(env_value, param_names[i], round(grid_ith_dimension[0], 5),
                                             round(grid_ith_dimension[-1], 5)))
                    index.clear()
            if len(index) > 1 and len(index) == len(param_names):
                if tuple(index) not in indices_with_value:
                    indices_with_value[tuple(index)] = []
                indices_with_value[tuple(index)].append((env_values, probability))
        self.logger.debug('time elapsed to map values into grid: {}s'.format(time.time() - start_time))

        number_of_collisions = 0
        new_indices_with_value = dict()
        approximated_sum_of_probabilities = 0.0
        frontier_pairs_collided_dict = dict()
        for index, values in indices_with_value.items():
            if len(values) > 1:
                self.logger.debug('collision in index: {}, values: {}'.format(index, values))
                number_of_collisions += len(values) - 1
                probabilities = np.array([probability for env_values, probability in values])

                for env_values in values:
                    result, pos = is_in_archive(env_values=env_values, archive=archive, param_names=param_names)
                    if result:
                        if pos not in frontier_pairs_collided_dict:
                            frontier_pairs_collided_dict[pos] = False
                        self.logger.debug('frontier pair collided: {}, {}'.format(env_values, pos))
                        frontier_pairs_collided_dict[pos] = True

                # for i in range(0, len(values) - 1, 2):
                #     values_1, values_2 = values[i], values[i + 1]
                #     if (values_1[1] > 0.5 and values_2[1] < 0.5) or (values_1[1] < 0.5 and values_2[1] > 0.5):
                #         dist = compute_dist_values(t_env_values=values_1[0], f_env_values=values_2[0])
                #         if dist <= 0.05:
                #             frontier_pairs_collided += 1
                #             self.logger.debug('frontier pair collided: {}, {}'.format(values_1, values_2))

                mean_probability = np.mean(probabilities)
                # taking the mean of probabilities of the collided env_values
                self.logger.debug('taking mean probability: {}, considering probabilities: {}'
                                  .format(mean_probability, probabilities))
                new_indices_with_value[index] = (values[0][0], mean_probability)
                approximated_sum_of_probabilities += mean_probability
            else:
                new_indices_with_value[index] = (values[0][0], values[0][1])
                approximated_sum_of_probabilities += values[0][1]

        unique_keys = set()
        for key in frontier_pairs_collided_dict.keys():
            unique_keys.add(key.split('_')[0])
        self.logger.debug('Num frontier pairs collided: {} / {}'.format(len(unique_keys), len(archive) / 2))
        frontier_pairs_collided = len(unique_keys)

        self.logger.debug('Original probabilities sum: {}; Approximated: {}'.format(original_sum_of_probabilities,
                                                                                    approximated_sum_of_probabilities))
        return new_indices_with_value, number_of_collisions, frontier_pairs_collided

    def compute_probability_volume(
            self,
            buffer_file: str,
            last_buffer_file: str,
            archive_file: str,
            grid_granularity_percentage_of_range: float,
            param_names_to_consider: List[str],
            regression_probability: bool = False,
            approximate_nearest_neighbor: bool = False,
            show_plot: bool = False,
            plot_only_approximated: bool = False,
            plot_file_path: str = None,
            limits_dict: Dict = None,
            plot_nn: bool = False,
            smooth: float = 1.0,
            perplexity: int = 0.0,
            only_tsne: bool = False,
            max_points_x: int = None,
            skip_points_x: int = None,
            max_points_y: int = None,
            skip_points_y: int = None,
            indices_frontier_not_adapted: List[Tuple] = None,
            indices_frontier_not_adapted_appr: List[Tuple] = None,
            n_iterations_dim_reduction: int = 10000,
    ) -> Tuple[float, float, float, float, List[Tuple], List[Tuple], int]:

        _indices_frontier_not_adapted = None
        _indices_frontier_not_adapted_appr = None

        buffer_resampling = read_saved_buffer(buffer_file=buffer_file)
        buffer = read_saved_buffer(buffer_file=last_buffer_file)
        archive = read_saved_archive(archive_file=archive_file)

        param_names = list(buffer_resampling[0].get_env_values().keys())
        if param_names_to_consider and len(param_names_to_consider) > 0:
            param_names = list(filter(lambda pn: pn in param_names_to_consider, param_names))

        assert len(param_names) > 1, 'At least two parameters must be considered. Found {}'.format(len(param_names))

        env_values_probabilities = get_env_values_probabilities(
            buffer=buffer_resampling, param_names=param_names,
            regression_probability=regression_probability,
        )
        total_number_of_search_points = len(env_values_probabilities)
        env_values_probabilities_archive = get_env_values_probabilities_archive(
            archive=archive, param_names=param_names,
        )
        env_values_dict = get_env_values_dict(
            env_values_probabilities=env_values_probabilities,
            param_names=param_names
        )

        self.logger.debug('param_names: {}'.format(param_names))
        self.logger.debug('env_values_dict: {}'.format(env_values_dict))
        self.logger.debug('limits_dict: {}'.format(limits_dict))

        # sort_key_fn = lambda tup: tuple([tup[0][_index] for _index in range(len(env_values_probabilities[0][0]))])
        # env_values_probabilities = sorted(env_values_probabilities, key=sort_key_fn)
        start_time = time.time()
        grid_components, tensor_shape = compute_grid(
            env_values_dict=env_values_dict,
            grid_granularity_percentage_of_range=grid_granularity_percentage_of_range,
            limits_dict=limits_dict
        )
        self.logger.debug('compute grid time elapsed: {}s'.format(time.time() - start_time))

        self.logger.debug('tensor shape: {}'.format(tensor_shape))
        tensor = np.zeros(shape=tensor_shape, dtype=np.float32)
        tensor[...] = np.nan

        start_time = time.time()
        grid_indices_with_values, number_of_collisions, frontier_pairs_collided = self.adapting_search_values_to_grid(
            env_values_probabilities=env_values_probabilities,
            grid_components=grid_components,
            param_names=param_names,
            archive=archive
        )

        self.logger.info('total number of collisions: {}'.format(number_of_collisions))
        self.logger.info('total number of search points: {}'.format(total_number_of_search_points))
        self.logger.info('collision %: {}'.format((number_of_collisions / total_number_of_search_points) * 100))
        self.logger.info('time elapsed inserting search values in grid: {}s'.format(round(time.time() - start_time, 2)))

        start_time = time.time()
        tensor = fill_grid_with_search_values(
            grid_indices_with_values=grid_indices_with_values,
            tensor=tensor
        )

        self.logger.info('time elapsed filling the grid: {}s'.format(round(time.time() - start_time, 2)))
        n_clusters = None

        if len(param_names) > 2 and not regression_probability and only_tsne:
            n_clusters = plot_frontier_points_high_dim(
                perplexity=perplexity,
                env_values_probabilities=env_values_probabilities,
                env_values_probabilities_archive=env_values_probabilities_archive,
                plot_file_path=plot_file_path,
                param_names=param_names,
                n_iterations=n_iterations_dim_reduction,
            )

        if not only_tsne:

            if not regression_probability:
                start_time = time.time()
                start_time_resampling = time.time()
                for _ in range(3):

                    tensor = resampling(tensor=tensor,
                                        grid_components=grid_components,
                                        param_names=param_names,
                                        grid_indices_with_values=grid_indices_with_values)

                self.logger.info('time elapsed performing resampling in grid: {}s'.format(
                    round(time.time() - start_time_resampling, 2)))

            elif regression_probability and indices_frontier_not_adapted:
                # setting values of non-important indices so that the nearest neighbor does not need to do useless work
                for index_frontier_not_adapted in indices_frontier_not_adapted:
                    tensor[index_frontier_not_adapted] = 0.0

            boolean_mask_nan = np.isnan(tensor)
            nd_indices_nan = np.where(boolean_mask_nan)
            indices_nan = list(zip(*nd_indices_nan))
            len_indices_nan = len(indices_nan)
            del indices_nan
            if len_indices_nan > 1e6:
                tensor = nearest_neighbor_parallel(tensor=tensor, tensor_shape=tensor_shape)
            else:
                tensor = nearest_neighbor(tensor=tensor, tensor_shape=tensor_shape)

            self.logger.info('time elapsed performing nn in grid: {}s'.format(round(time.time() - start_time, 2)))

            max_volume_tensor = np.ones(shape=tensor_shape, dtype=np.float32)
            indices_adapted = [tuple(index) for index in
                               list(zip(*np.where(np.logical_and(max_volume_tensor > 0.5, max_volume_tensor <= 1.0))))]
            max_volume = len(indices_adapted) / np.multiply.reduce(tensor_shape)
            min_volume = 0.0

            min_value = np.min(tensor)
            max_value = np.max(tensor)
            if min_value != max_value:
                tensor = (tensor - min_value) / (max_value - min_value)

            indices_frontier = None
            if regression_probability:
                # since in this case 1.0 means 100% regression (not desired behaviour)
                # while in the nominal case 1.0 means 100% pass probability (desired behaviour)
                reversed_tensor = abs(tensor - 1.0)
                assert indices_frontier_not_adapted, 'Adaptation volume needs to be computed first to determine where regression volume is defined'
                # here compute only the volume of the part where regression probability is defined, i.e. within the adaptation frontier
                for index_frontier_not_adapted in indices_frontier_not_adapted:
                    reversed_tensor[index_frontier_not_adapted] = np.nan
                    max_volume_tensor[index_frontier_not_adapted] = np.nan

                indices_not_nan = [tuple(index) for index in list(zip(*np.where(~np.isnan(reversed_tensor))))]
                indices_not_regressed = [index for index in indices_not_nan
                                         if 0.5 < max_volume_tensor[tuple(index)] <= 1.0]
                reversed_tensor = reversed_tensor[~np.isnan(reversed_tensor)]
                max_volume = len(indices_not_regressed) / len(indices_not_nan)
                indices_not_regressed = [tuple(index) for index in
                                         list(zip(*np.where(reversed_tensor > 0.5)))]
                volume = len(indices_not_regressed) / len(indices_not_nan)
            else:
                indices_adapted = [tuple(index) for index in
                                   list(zip(*np.where(tensor > 0.5)))]
                volume = len(indices_adapted) / np.multiply.reduce(tensor_shape)
                self.logger.info('Volume: {}, volume max: {}'.format(volume, max_volume))
                _indices_frontier_not_adapted = [tuple(index) for index in
                                                 list(zip(*np.where(np.logical_and(tensor >= 0.0, tensor <= 0.5))))]

            volume = np.interp(volume, [min_volume, max_volume], [0.0, 1.0])

            self.logger.info('volume after interpolating nan values: {}'.format(volume))
            volume_after_nn = volume

            volume_after_approximation = None
            if approximate_nearest_neighbor and len(list(tensor_shape)) == 2:
                start_time = time.time()
                approximated_tensor, env_values_repeated = approximate(
                    grid_components=grid_components,
                    tensor=tensor,
                    tensor_shape=tensor_shape,
                    plot_nn=plot_nn,
                    smooth=smooth
                )

                # don't know why but there could values > 1 and < 0 resulting from the approximation
                approximated_tensor[approximated_tensor < 0.0] = 0.0
                approximated_tensor[approximated_tensor > 1.0] = 1.0

                self.logger.info('time elapsed approximation after nn: {}s'.format(round(time.time() - start_time, 2)))
                if regression_probability:
                    # since in this case 1.0 means 100% regression (not desired behaviour)
                    # while in the nominal case 1.0 means 100% pass probability (desired behaviour)
                    reversed_approximated_tensor = abs(approximated_tensor - 1.0)
                    # here compute only the volume of the part where regression probability is defined, i.e. within the adaptation frontier
                    for index_frontier_not_adapted in indices_frontier_not_adapted_appr:
                        reversed_approximated_tensor[index_frontier_not_adapted] = np.nan

                    indices_not_nan = [tuple(index) for index in
                                       list(zip(*np.where(~np.isnan(reversed_approximated_tensor))))]
                    volume = np.sqrt(np.nansum(np.square(reversed_approximated_tensor))) / len(indices_not_nan)
                else:
                    indices_adapted = [tuple(index) for index in
                                       list(zip(*np.where(np.logical_and(tensor > 0.5, tensor <= 1.0))))]
                    volume = len(indices_adapted) / np.multiply.reduce(tensor_shape)
                    _indices_frontier_not_adapted_appr = [tuple(index) for index in list(zip(*np.where(
                        np.logical_and(approximated_tensor >= 0.0, approximated_tensor <= 0.59))))]

                volume = np.interp(volume, [min_volume, max_volume], [0.0, 1.0])

                volume_after_approximation = volume
                self.logger.info('volume after linear interpolation: {}'.format(volume))

            if plot_nn:
                approximated_tensor = tensor

            if len(list(tensor_shape)) == 2 and approximate_nearest_neighbor:
                plot_heatmap(
                    buffer=buffer,
                    grid_indices_with_values=grid_indices_with_values,
                    grid_components=grid_components,
                    tensor_shape=tensor_shape,
                    env_values_repeated=env_values_repeated,
                    env_values_dict=env_values_dict,
                    regression_probability=regression_probability,
                    param_names=param_names,
                    approximated_tensor=approximated_tensor,
                    show_plot=show_plot,
                    plot_only_approximated=plot_only_approximated,
                    algo_name=self.algo_name,
                    env_name=self.env_name,
                    interpolation_function=INTERPOLATION_FUNCTION,
                    plot_file_path=plot_file_path,
                    plot_nn=plot_nn,
                    model_suffix=self.model_suffix,
                    max_points_x=max_points_x,
                    skip_points_x=skip_points_x,
                    max_points_y=max_points_y,
                    skip_points_y=skip_points_y,
                    indices_frontier_not_adapted_appr=indices_frontier_not_adapted_appr,
                )
            return volume_after_nn, volume_after_approximation, \
                   (number_of_collisions / total_number_of_search_points) * 100, \
                   frontier_pairs_collided, _indices_frontier_not_adapted, \
                   _indices_frontier_not_adapted_appr, n_clusters

        return 0.0, 0.0, (number_of_collisions / total_number_of_search_points) * 100, \
               frontier_pairs_collided, None, None, n_clusters

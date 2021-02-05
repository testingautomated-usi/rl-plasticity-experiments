import time
from typing import List, Tuple

# from adjustText import adjust_text
from sklearn.manifold import TSNE
from sklearn import tree

import matplotlib.pyplot as plt
import os
import graphviz

from analysis.clustering import cluster_data
from log import Log
from plot.plot_clusters import plot_clusters


def plot_frontier_points_high_dim(
        perplexity: int,
        env_values_probabilities: List[Tuple[List, float]],
        env_values_probabilities_archive: List[Tuple[List, float]],
        plot_file_path: str,
        param_names: List[str],
        n_iterations: int = 30000) -> int:
    logger = Log('plot_frontier_points_high_dim')

    env_values = [env_values_probability[0] for env_values_probability in env_values_probabilities_archive]
    env_values.insert(0, env_values_probabilities[0][0])
    pass_probabilities = [env_values_probability[1] for env_values_probability in env_values_probabilities_archive]
    pass_probabilities.insert(0, env_values_probabilities[0][1])

    logger.info('computing TSNE, points: {}'.format(len(env_values_probabilities_archive)))
    start_time_tsne = time.time()
    env_values_embedded = TSNE(
        n_components=2, perplexity=perplexity, n_jobs=-1, n_iter=n_iterations
    ).fit_transform(env_values)
    end_time_tsne = time.time()
    logger.info('TSNE time elapsed: {}s'.format(end_time_tsne - start_time_tsne))

    _ = plt.figure(figsize=(15, 14))

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

    plt.scatter(env_values_embedded[0][0], env_values_embedded[0][1], s=200, color='blue')
    red_xs, red_ys = [], []
    green_xs, green_ys = [], []
    for i, env_value_pair in enumerate(env_values_embedded):
        if i != 0 and i % 2 != 0:
            green_xs.append(env_value_pair[0])
            green_ys.append(env_value_pair[1])
        elif i != 0:
            red_xs.append(env_value_pair[0])
            red_ys.append(env_value_pair[1])

    plot_frontier_points_2D(green_xs=green_xs, green_ys=green_ys, red_xs=red_xs, red_ys=red_ys,
                            points_size=100, save_or_show=False)

    # texts = []
    # for i in range(len(env_values_embedded)):
    #     tuple_txt = '('
    #     for x in env_values[i]:
    #         tuple_txt += str(round(x, 2)) + ','
    #     tuple_txt = tuple_txt[:len(tuple_txt) - 1] + ')'
    #     texts.append(plt.annotate(tuple_txt, (env_values_embedded[i][0], env_values_embedded[i][1])))

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    clusterer, labels, centers, _ = cluster_data(data=env_values_embedded, n_clusters_interval=(2, 30))
    logger.info('Number of clusters: {}'.format(len(centers)))
    cluster_plot_save_path = plot_file_path + '_p_' + str(perplexity) + '.pdf'
    plot_clusters(clusterer=clusterer, data=env_values_embedded, points_size=300, points_marker='*', save_or_show=False)
    # adjust_text(texts)

    start_time_dt = time.time()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X=env_values, y=labels)
    logger.info('Time elapsed decision trees: {}s'.format(time.time() - start_time_dt))
    start_time_dot = time.time()
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=param_names,
                                    class_names=[str(_class) for _class in clf.classes_])
    graph = graphviz.Source(dot_data)
    logger.info('Time elapsed dot: {}s'.format(time.time() - start_time_dot))

    if plot_file_path:
        plt.savefig(cluster_plot_save_path, format='pdf')
        dt_plot_save_path = plot_file_path + '_p_' + str(perplexity) + '_dt'
        graph.render(filename=dt_plot_save_path, format='pdf')
    else:
        plt.show()

    return len(centers)


def plot_frontier_points_2D(green_xs, green_ys, red_xs, red_ys, points_size: int = 20, points_marker: str = '.',
                            save_or_show: bool = True, plot_file_path: str = None, param_names: List[str] = None,
                            x_lim: List[float] = None, y_lim: List[float] = None) -> None:
    ax = plt.gca()
    plt.scatter(green_xs, green_ys, c='green', marker=points_marker, s=points_size)
    plt.scatter(red_xs, red_ys, c='red', marker=points_marker, s=points_size)
    for green_x, red_x, green_y, red_y in zip(green_xs, red_xs, green_ys, red_ys):
        plt.plot([green_x, red_x], [green_y, red_y], color='gray', linestyle='dashed')

    if param_names:
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if save_or_show:
        if plot_file_path:
            plt.savefig(plot_file_path, format='pdf')
            plt.clf()
        else:
            plt.show()


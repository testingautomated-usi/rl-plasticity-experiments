from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from log import Log


def cluster_data(data, n_clusters_interval: Tuple) -> Tuple:
    logger = Log("cluster_data")
    optimal_n_clusters = 1
    optimal_score = -1
    if n_clusters_interval[0] != 1:
        range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
        optimal_score = -1
        optimal_n_clusters = -1
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(data)
            try:
                silhouette_avg = silhouette_score(data, cluster_labels)
                logger.debug("For n_clusters = {}, the average silhouette score is: {}".format(n_clusters, silhouette_avg))
                if silhouette_avg > optimal_score:
                    optimal_score = silhouette_avg
                    optimal_n_clusters = n_clusters
            except ValueError:
                break

        assert optimal_n_clusters != -1, "Error in silhouette analysis"
        logger.debug("Best score is {} for n_cluster = {}".format(optimal_score, optimal_n_clusters))

    clusterer = KMeans(n_clusters=optimal_n_clusters).fit(data)
    labels = clusterer.labels_
    centers = clusterer.cluster_centers_
    return clusterer, labels, centers, optimal_score

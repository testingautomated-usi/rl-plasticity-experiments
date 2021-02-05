import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(clusterer, data, points_size: int = 30, points_marker: str = 'D', plot_file_path: str = None, save_or_show: bool = True) -> plt:
    centers = clusterer.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='magenta', marker=points_marker, s=points_size, alpha=0.5)
    for i in range(len(centers)):
        plt.annotate('class-' + str(i), (centers[i][0], centers[i][1]))
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = np.asarray(data)[:, 0].min() - 1, np.asarray(data)[:, 0].max() + 1
    y_min, y_max = np.asarray(data)[:, 1].min() - 1, np.asarray(data)[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect="auto", origin="lower", alpha=0.5)
    if save_or_show:
        if plot_file_path:
            plt.savefig(plot_file_path, format='pdf')
            plt.clf()
        else:
            plt.show()

    return plt

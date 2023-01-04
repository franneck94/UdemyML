import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import linalg

from mpl_toolkits.mplot3d import Axes3D


colors = ["yellow", "purple"]


def plot_results(
    X: np.ndarray,
    Y: np.ndarray,
    Y_: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    index: int,
    title: str,
) -> None:
    _ = plt.figure(figsize=(12, 12))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=colors[i], s=20)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            mean, v[0], v[1], 180.0 + angle, color=colors[i]
        )
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-5.0, 5.0)
    plt.ylim(-5.0, 5.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

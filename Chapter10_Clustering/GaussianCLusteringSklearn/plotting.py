from typing import Any

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


colors = ["yellow", "purple"]


def plot_results(
    X: np.ndarray, Y: np.ndarray, Y_: np.ndarray, means: np.ndarray, covariances: np.ndarray, index: int, title: Any
) -> None:
    _ = plt.figure(figsize=(12, 12))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar) in enumerate(zip(
            means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=colors[i], s=20)

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=colors[i])
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-5., 5.)
    plt.ylim(-5., 5.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

from typing import Any

import numpy as np


np.random.seed(42)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF", "#AAFFAA"])


def make_meshgrid(
    x: np.ndarray,
    y: np.ndarray,
    h: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(
    ax: plt.Axes,
    clf: KMeans,
    xx: np.ndarray,
    yy: np.ndarray,
    **params: Any,
) -> Any:
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # noqa: N806
    Z = Z.reshape(xx.shape)  # noqa: N806
    return ax.contourf(xx, yy, Z, **params)


def plot(
    x: np.ndarray,
    y_pred: np.ndarray,
    centroids: np.ndarray,
    kmeans: np.ndarray,
) -> None:
    _, ax = plt.subplots()
    # Decision Border
    X0, X1 = x[:, 0], x[:, 1]  # noqa: N806
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, kmeans, xx, yy, cmap=cmap_light, alpha=0.8)

    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="red",
        s=250,
        marker="*",
    )
    plt.show()

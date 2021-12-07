from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sized
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.colors import ListedColormap
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures


DEFAULT_COLOR_MAP = ListedColormap(
    [color for color in plt.get_cmap("Set1").colors]
    + [color for color in plt.get_cmap("Set2").colors]
)

DEFAULT_COLORS = np.array(
    [
        (color[0] - 0.01, color[1] - 0.01, color[2] - 0.01)
        for color in plt.get_cmap("Set1").colors
    ]
    + [
        (color[0] - 0.01, color[1] - 0.01, color[2] - 0.01)
        for color in plt.get_cmap("Set2").colors
    ]
)


############
### DATA ###
############


def plot_dataset_1d_clf(
    x: np.ndarray, y: np.ndarray, colors: Optional[np.ndarray] = None,
) -> None:
    assert x.shape[1] == 1
    assert len(y.shape) == 1
    num_classes = len(np.unique(y))
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]
    plt.scatter(x, 0, color=colors[y])
    plt.show()


def plot_dataset_1d_regr(x: np.ndarray, y: np.ndarray) -> None:
    assert x.shape[1] == 1
    assert len(y.shape) == 1
    plt.scatter(x, y)
    plt.show()


def plot_dataset_2d_clf(
    x: np.ndarray, y: np.ndarray, colors: Optional[np.ndarray] = None,
) -> None:
    assert x.shape[1] == 2
    assert len(y.shape) == 1
    num_classes = len(np.unique(y))
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]
    plt.scatter(x[:, 0], x[:, 1], c=colors[y])
    plt.show()


############
### HELP ###
############


def make_meshgrid(
    x: np.ndarray, y: np.ndarray, h: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def contour_face(
    ax: plt.Axes,
    clf: sklearn.base.ClassifierMixin,
    xx: np.ndarray,
    yy: np.ndarray,
    **params: Any,
) -> None:
    Z: np.ndarray = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    _ = ax.contourf(xx, yy, Z, **params)


############
### CLFS ###
############


def plot_decision_border(
    clf: sklearn.base.ClassifierMixin,
    x: np.ndarray,
    y: np.ndarray,
    cmap: Optional[list] = None,
    colors: Optional[np.ndarray] = None,
    **scatter_kwargs: Any,
) -> None:
    assert x.shape[1] <= 2
    assert len(y.shape) == 1
    num_classes = len(np.unique(y))
    if cmap is None:
        cmap = ListedColormap(
            [color for color in DEFAULT_COLOR_MAP.colors[:num_classes]]
        )
    if colors is None:
        colors = DEFAULT_COLORS[:num_classes]
    _, ax = plt.subplots()
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    contour_face(ax, clf, xx, yy, cmap=cmap, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=colors[y], **scatter_kwargs)
    plt.show()


############
### REGR ###
############


def plot_regressor(
    regr: sklearn.base.RegressorMixin,
    x: np.ndarray,
    y: np.ndarray,
    **scatter_kwargs: Any,
) -> None:
    assert x.shape[1] == 1
    assert len(y.shape) == 1
    offset = 1.0
    x1 = np.min(x) - offset
    x2 = np.max(x) + offset
    x_arange = np.arange(start=x1, stop=x2, step=0.05).reshape((-1, 1))
    y_arange = regr.predict(x_arange)
    plt.plot(y_arange, y_arange, color="red")
    plt.scatter(x, y, **scatter_kwargs)
    plt.show()


def plot_residuals(
    regr: sklearn.base.RegressorMixin,
    x: np.ndarray,
    y: np.ndarray,
    **scatter_kwargs: Any,
) -> None:
    assert x.shape[1] == 1
    assert len(y.shape) == 1
    y_pred = regr.predict(x)

    offset = 1.0
    min_val = min(np.min(y), np.min(y_pred)) - offset
    max_val = max(np.max(y), np.max(y_pred)) + offset

    plt.scatter(y, y_pred - y, **scatter_kwargs)
    plt.hlines(y=0, xmin=min_val, xmax=max_val)
    plt.show()


def plot_poly_reg(
    regr: sklearn.base.RegressorMixin,
    pf: PolynomialFeatures,
    x: np.ndarray,
    y: np.ndarray,
    **scatter_kwargs: Any,
) -> None:
    assert x.shape[1] == 1
    assert len(y.shape) == 1
    offset = 1.0
    x1 = np.min(x) - offset
    x2 = np.max(x) + offset
    x_arange = np.arange(start=x1, stop=x2, step=0.05).reshape((-1, 1))
    x_arange_transformed = pf.transform(x_arange)
    y_arange = regr.predict(x_arange_transformed)
    _ = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color="white", s=10, marker="o", label="Dataset", **scatter_kwargs)
    plt.plot(x_arange, y_arange)
    plt.show()


############
### CLUS ###
############


def plot_kmeans(
    kmeans: KMeans, x: np.ndarray, cmap: Optional[list] = None, **scatter_kwargs: Any,
) -> None:
    assert x.shape[1] <= 2
    if cmap is None:
        cmap = DEFAULT_COLOR_MAP
    y_pred = kmeans.predict(x)
    centroids = kmeans.cluster_centers_
    _, ax = plt.subplots()
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    contour_face(ax, kmeans, xx, yy, cmap=cmap, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, **scatter_kwargs)
    plt.scatter(
        centroids[:, 0], centroids[:, 1], color="red", s=250, marker="*"
    )
    plt.show()


def plot_gmm(
    model: GaussianMixture, x: np.ndarray, colors: Optional[np.ndarray] = None,
) -> None:
    assert x.shape[1] <= 2
    if colors is None:
        colors = DEFAULT_COLORS
    y_pred = model.predict(x)
    means, covariances = model.means_, model.covariances_
    _, ax = plt.subplots()
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(y_pred == i):
            continue
        plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], color=colors[i], s=20)

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(
            mean, v[0], v[1], 180.0 + angle, color=colors[i]
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


############
### MISC ###
############


def plot_crossval(scores: List[float]) -> None:
    assert len(scores) > 0
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    plt.plot(range(len(scores)), scores, color="blue")
    plt.fill_between(
        [0, len(scores) - 1],
        mean_score - std_score,
        mean_score + std_score,
        alpha=0.2,
        color="lightblue",
    )
    plt.axhline(mean_score, linestyle="-", color="red")
    plt.legend(["Validation Scores", "Mean Score"])
    plt.show()


def plot_validation_curve(
    train_scores: List[float], test_scores: List[float], param_range: Sized,
) -> None:
    assert len(train_scores) > 0
    assert len(test_scores) > 0
    assert len(param_range) > 0
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.ylabel("Score")
    lw = 2
    plt.plot(
        param_range,
        train_scores_mean,
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        param_range,
        test_scores_mean,
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )

    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="lightblue",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(
    train_sizes: Sized, train_scores: List[float], test_scores: List[float]
) -> None:
    assert len(train_sizes) > 0
    assert len(train_scores) > 0
    assert len(test_scores) > 0
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="lightgreen",
    )
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="g",
        label="Cross-validation score",
    )
    plt.legend(loc="best")
    plt.show()

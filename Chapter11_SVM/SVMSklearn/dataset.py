from typing import Tuple

import numpy as np
np.random.seed(42)


def generate_dataset(num_class1: int = 50, num_class2: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    cov1 = np.array([[1, 0], [0, 2]])
    mean1 = np.array([0, 0])
    cov2 = np.array([[2, 0], [0, 1]])
    mean2 = np.array([2, 2])
    data1 = np.random.multivariate_normal(
        mean1, cov1, num_class1
    )
    data2 = np.random.multivariate_normal(
        mean2, cov2, num_class2
    )
    data = np.concatenate((data1, data2), axis=0)
    classes = np.array(
        [0 for i in range(num_class1)] +
        [1 for i in range(num_class2)]
    )
    return data, classes


def cond(x: np.ndarray) -> np.ndarray:
    return (
        (np.abs(x[:, 0]) < 1.0) &
        (np.abs(x[:, 1]) < 1.0)
    )


def filter_cond(x: np.ndarray) -> np.ndarray:
    return (
        ((np.abs(x[:, 0]) < 1.0) |
         (np.abs(x[:, 0]) > 1.75)) &
        ((np.abs(x[:, 1]) < 1.0) |
         (np.abs(x[:, 1]) > 1.75))
    )


def generate_kernel_dataset() -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.diag([5.0, 4.0]), size=200)
    x = x[filter_cond(x)]
    y = cond(x).astype(np.float32)
    return x, y

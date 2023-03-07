from typing import Tuple

import numpy as np


np.random.seed(42)


def generate_dataset(
    num_class_1: int = 50, num_class_2: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    cov_1 = np.array([[1, 0], [0, 2]])
    mean_1 = np.array([0, 0])
    cov_2 = np.array([[2, 0], [0, 1]])
    mean_2 = np.array([2, 2])
    data_1 = np.random.multivariate_normal(mean_1, cov_1, num_class_1)
    data_2 = np.random.multivariate_normal(mean_2, cov_2, num_class_2)
    data = np.concatenate((data_1, data_2), axis=0)
    classes = np.array(
        [0 for i in range(num_class_1)] + [1 for i in range(num_class_2)]
    )
    return data, classes

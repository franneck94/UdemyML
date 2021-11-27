from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris


def generate_dataset() -> Tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    right_classes_idx = [idx for idx in range(iris.data.shape[0]) if iris.target[idx] in [0, 1]]

    x = iris.data[right_classes_idx, :2]
    y = iris.target[right_classes_idx]
    return x, y

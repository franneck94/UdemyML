from __future__ import annotations

import math
from typing import Optional
from typing import SupportsFloat

import numpy as np


def split_func(
    sample: np.ndarray,
    feature_i: int,
    threshold: float
) -> bool:
    return sample[feature_i] >= threshold


def divide_on_feature(
    x: np.ndarray,
    feature_i: int,
    threshold: float
) -> np.ndarray:
    """Divide dataset based on if sample value on feature index is larger than
    the given threshold.
    """
    X_1 = np.array(
        [sample for sample in x if split_func(sample, feature_i, threshold)]
    )
    X_2 = np.array(
        [sample for sample in x if not split_func(sample, feature_i, threshold)]
    )
    return np.array([X_1, X_2])


def log2(x: SupportsFloat) -> float:
    return math.log(x) / math.log(2)


def calculate_entropy(y: np.ndarray) -> float:
    unique_labels = np.unique(y)
    entropy = 0.0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_information_gain(
    y: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray
) -> float:
    p = len(y1) / len(y)
    entropy = calculate_entropy(y)
    info_gain = entropy - p * \
        calculate_entropy(y1) - (1 - p) * \
        calculate_entropy(y2)
    return info_gain


def majority_vote(y: np.ndarray) -> float:
    most_common = 0.0
    max_count = 0
    for label in np.unique(y):
        count = len(y[y == label])
        if count > max_count:
            most_common = label
            max_count = count
    return most_common


class DecisionNode:
    """Class that represents a decision node or leaf in the decision tree.
    """

    def __init__(
        self,
        feature_i: int = None,
        threshold: float = None,
        value: float = None,
        true_branch: DecisionNode = None,
        false_branch: DecisionNode = None
    ) -> None:
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class ClassificationTree:
    """Class of ClassificationTree.
    """

    def __init__(
        self,
        min_samples_split: int = 2,
        min_impurity: float = 1e-7,
        max_depth: float = np.inf
    ) -> None:
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray
    ):
        self.root = self._build_tree(x, y)

    def _build_tree(
        self,
        x: np.ndarray,
        y: np.ndarray,
        current_depth: int = 0
    ):
        """Recursive method which builds out the decision tree and splits
        x and respective y on the feature of x which (based on impurity)
        best separates the data.
        """
        largest_impurity = 0.0
        best_criteria = {}
        best_sets = {}

        # Add y as last column of x
        Xy = np.concatenate((x, y), axis=1)
        n_samples, n_features = np.shape(x)

        if (n_samples >= self.min_samples_split and
                current_depth <= self.max_depth):
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = x[:, feature_i]
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide x and y depending on if the feature value
                    # of x at index feature_i meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = calculate_information_gain(y, y1, y2)

                        # If this threshold resulted in a higher information
                        # gain than previously recorded save the threshold value
                        # and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i,
                                "threshold": threshold
                            }
                            best_sets = {
                                "x_left": Xy1[:, :n_features],
                                "y_left": Xy1[:, n_features:],
                                "x_right": Xy2[:, :n_features],
                                "y_right": Xy2[:, n_features:]
                            }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(
                best_sets["x_left"],
                best_sets["y_left"],
                current_depth + 1
            )
            false_branch = self._build_tree(
                best_sets["x_right"],
                best_sets["y_right"],
                current_depth + 1
            )
            return DecisionNode(
                feature_i=best_criteria["feature_i"],
                threshold=best_criteria["threshold"],
                true_branch=true_branch,
                false_branch=false_branch
            )

        # We're at leaf => determine value
        leaf_value = majority_vote(y)
        return DecisionNode(value=leaf_value)

    def predict_value(
        self,
        x: np.ndarray,
        tree: Optional[DecisionNode] = None
    ):
        """Do a recursive search down the tree and make a prediction of
        the data sample by the value of the leaf that we end up at.
        """
        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the pred
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if feature_value >= tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(
        self,
        x: np.ndarray
    ):
        y_pred = np.array([self.predict_value(xi) for xi in x])
        return y_pred

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray
    ):
        y_pred = self.predict(x)
        true_pred = np.sum(
            [y_pred_i == y_i for y_pred_i, y_i in zip(y_pred, y)]
        )
        n = len(y)
        accuracy = true_pred / n
        return accuracy


if __name__ == "__main__":
    x_train = np.random.uniform(-2, 2, size=(5, 3))
    y_train = np.random.randint(0, 2, size=(5, 1))

    clf = ClassificationTree()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    score = clf.score(x_train, y_train)
    print(f"Accuracy: {score}")

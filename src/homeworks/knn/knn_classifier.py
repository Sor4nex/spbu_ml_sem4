from typing import Any

import numpy as np

from src.homeworks.knn.kd_tree import KDTree


class KNNClassifier:
    def __init__(self, n_neighbours: int = 3, kd_tree_leaf_size: int = 5) -> None:
        self.n_neighbours = n_neighbours
        self.kd_tree_leaf_size = kd_tree_leaf_size
        self.kd_tree: KDTree | None = None
        self.targets: dict = {}

    def fit(self, train_data: np.ndarray, targets: np.ndarray) -> None:
        if len(train_data) == 0:
            raise ValueError("empty array given")
        if len(train_data) != len(targets):
            raise ValueError("number of points doesnt match the number of classes")

        self.kd_tree = KDTree(train_data, self.kd_tree_leaf_size)
        for i in range(len(train_data)):
            self.targets[tuple(train_data[i])] = targets[i]

    def predict_proba(self, prediction_data: np.ndarray) -> list[dict[Any, float]]:
        if self.kd_tree is None:
            raise ValueError("knn classifier was not fit with data")

        res = []
        k_nearest_neighbours = self.kd_tree.query(prediction_data, self.n_neighbours)
        for points_neighbours in k_nearest_neighbours:
            classes_counts: dict = {}
            for neighbour in points_neighbours:
                neighbour_class = self.targets.get(tuple(neighbour))
                if neighbour_class in classes_counts:
                    classes_counts[neighbour_class] += 1
                else:
                    classes_counts[neighbour_class] = 1
            for class_count in classes_counts.keys():
                classes_counts[class_count] /= self.n_neighbours
            res.append(classes_counts)
        return res

    def predict(self, prediction_data: np.ndarray) -> list[Any]:
        proba_predictions = self.predict_proba(prediction_data)
        res = [
            max(prediction.items(), key=lambda x: x[1])[0]
            for prediction in proba_predictions
        ]
        return res

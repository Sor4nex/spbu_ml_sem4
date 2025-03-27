import numpy as np
import pytest

from src.homeworks.knn.kd_tree import *


def find_nearest_neighbours_stupid(
    x: np.ndarray, all_points: np.ndarray, n_nearest: int = 1
) -> np.ndarray:
    res = []
    for point in x:
        neighbours = all_points.copy()
        sorted_indices = np.argsort(np.linalg.norm(point - neighbours, axis=1))
        res.append(np.array(neighbours[sorted_indices[:n_nearest]]))
    return np.array(res)


def test_kd_tree_query() -> None:
    for _ in range(100):
        dim = np.random.randint(1, 20)
        n_near = np.random.randint(1, 20)
        x_train = np.array(
            [
                np.random.uniform(-200, 200, size=dim)
                for _ in range(np.random.randint(100, 200))
            ]
        )
        x_test = np.array([np.random.uniform(-200, 200, size=dim) for _ in range(30)])

        tree = KDTree(x_train.copy(), np.random.randint(1, 50))
        res1 = tree.query(x_test.copy(), n_near)
        res2 = find_nearest_neighbours_stupid(x_test.copy(), x_train.copy(), n_near)

        assert res1.shape == res2.shape
        for i in range(len(x_test)):
            for j in range(n_near):
                assert np.linalg.norm(x_test[i] - res1[i][j]) == np.linalg.norm(
                    x_test[i] - res2[i][j]
                )

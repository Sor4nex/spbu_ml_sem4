import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class Node:
    split_axis: int | None = None
    median: float | None = None
    left_subtree: Optional[Union["Node", np.ndarray]] = None
    right_subtree: Optional[Union["Node", np.ndarray]] = None


class KDTree:
    def __init__(self, x: np.ndarray, leaf_size: int) -> None:
        self.root = self._build_kd_tree(x.copy(), leaf_size)

    def _build_kd_tree(self, x: np.ndarray, leaf_size: int) -> Node | np.ndarray:
        if len(x) <= leaf_size:
            return x

        node = Node()
        # get axis, whose values has the biggest range
        node.split_axis = int(
            np.argmax([max(x[:, i]) - min(x[:, i]) for i in range(len(x[0]))])
        )

        # sort array by axis, whose values has the biggest range
        x = x[np.argsort(x[:, node.split_axis])]
        node.median = (
            x[(len(x) - 1) // 2][node.split_axis]
            if len(x) % 2 == 0
            else (x[len(x) // 2][node.split_axis] + x[len(x) // 2 - 1][node.split_axis])
            / 2
        )
        node.left_subtree = self._build_kd_tree(x[: len(x) // 2], leaf_size)
        node.right_subtree = self._build_kd_tree(x[len(x) // 2 :], leaf_size)

        return node

    def query(self, x: np.ndarray, n_nearest: int = 1) -> np.ndarray:
        def _search_n_nearest(
            point: np.ndarray, node: Node | np.ndarray, n_nearest: int = 1
        ) -> tuple[list, float]:
            # if we reached leaf, then we return array of points, sorted by euclidian norm with target
            if isinstance(node, np.ndarray):
                res: list = [(elem, np.linalg.norm(point - elem)) for elem in node]
                res.sort(key=lambda el: el[1])
                if len(res) > n_nearest:
                    res = res[:n_nearest]
                max_norm = float(res[-1][1])
                return res, max_norm

            # find the closest points recursively, by checking left or right side of hyperplane with respect to median
            if node.left_subtree is not None and node.right_subtree is not None:
                res1 = _search_n_nearest(
                    point,
                    node.right_subtree
                    if point[node.split_axis] >= node.median
                    else node.left_subtree,
                    n_nearest,
                )

            # we should also check other side if there`s not enough points or distance to splitting plane less then
            # maximum distance between founded neighbours and target
            if (
                abs(point[node.split_axis] - node.median) < res1[1]
                or len(res1[0]) < n_nearest
            ):
                if node.left_subtree is not None and node.right_subtree is not None:
                    res2 = _search_n_nearest(
                        point,
                        node.left_subtree
                        if point[node.split_axis] >= node.median
                        else node.right_subtree,
                        n_nearest,
                    )
                res = res1[0] + res2[0]
                res.sort(key=lambda el: el[1])
                if len(res) > n_nearest:
                    res = res[:n_nearest]
                max_norm = float(res[-1][1])
                return res, max_norm
            return res1

        return np.array(
            [
                np.array(
                    [
                        neighbour[0]
                        for neighbour in _search_n_nearest(point, self.root, n_nearest)[
                            0
                        ]
                    ]
                )
                for point in x
            ]
        )

    def __str__(self) -> str:
        def _make_str_recursively(node: Node | np.ndarray, depth: int = 0) -> str:
            if isinstance(node, Node):
                return f"""{"\t" * depth}med: {node.median};
{"\t" * depth}split: {node.split_axis};
{"\t" * depth}left subtree: (
{_make_str_recursively(node.left_subtree, depth + 1) if node.left_subtree is not None else None}
{"\t" * depth})
{"\t" * depth}right subtree: (
{_make_str_recursively(node.right_subtree, depth + 1) if node.right_subtree is not None else None}
{"\t" * depth})"""
            return f"{'\t' * depth}{str(node)}"

        return _make_str_recursively(self.root)

import abc
from math import floor
from typing import Union

import numpy as np


def train_test_split(
    data: np.ndarray, targets: np.ndarray, test_size: float = 0.2
) -> tuple:
    if len(data) != len(targets):
        raise ValueError("number of points doesnt match the number of classes")

    if test_size == 1.0:
        return [], [], data, targets
    elif test_size == 0.0:
        return data, targets, [], []

    ind = floor(len(data) * (1 - test_size))
    # return x_train, y_train, x_test, y_test
    return data[:ind], targets[:ind], data[ind:], targets[ind:]


class Scaler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, train_data: np.ndarray) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, transform_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def fit_transform(
        self, train_data: np.ndarray, transform_data: np.ndarray
    ) -> np.ndarray:
        self.fit(train_data)
        return self.transform(transform_data)


class MinMaxScaler(Scaler):
    def __init__(self):
        self.x_min: Union[np.ndarray, None] = None
        self.x_max: Union[np.ndarray, None] = None

    def fit(self, train_data: np.ndarray) -> None:
        mins, maxs = [], []
        for column in range(len(train_data[0])):
            column_min = float(np.min(train_data[:, column]))
            column_max = float(np.max(train_data[:, column]))
            mins.append(column_min)
            maxs.append(column_max)

        self.x_min = np.array(mins)
        self.x_max = np.array(maxs)

    def transform(self, transform_data: np.ndarray) -> np.ndarray:
        if self.x_min is None or self.x_max is None:
            raise AttributeError("scaler was not fit with data")

        transform_data = transform_data.copy()
        for point in transform_data:
            for coordinate in range(len(point)):
                if self.x_max[coordinate] == self.x_min[coordinate]:
                    point[coordinate] = 0.0
                    continue
                point[coordinate] = (point[coordinate] - self.x_min[coordinate]) / (
                    self.x_max[coordinate] - self.x_min[coordinate]
                )

        return transform_data


class MaxAbsScaler(Scaler):
    def __init__(self):
        self.x_max_abs: Union[np.ndarray, None] = None

    def fit(self, train_data: np.ndarray) -> None:
        max_abs = []
        for column in range(len(train_data[0])):
            column_max = float(np.max(np.absolute(train_data[:, column])))
            max_abs.append(abs(column_max))

        self.x_max_abs = np.array(max_abs)

    def transform(self, transform_data: np.ndarray) -> np.ndarray:
        if self.x_max_abs is None:
            raise AttributeError("scaler was not fit with data")

        transform_data = transform_data.copy()
        for point in transform_data:
            for coordinate in range(len(point)):
                if self.x_max_abs[coordinate] == 0:
                    point[coordinate] = 0.0
                    continue
                point[coordinate] = point[coordinate] / self.x_max_abs[coordinate]

        return transform_data


class Metrics:
    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError(
                "length of prediction vector doesnt match the length of true values vector"
            )

        tp = 0
        for i in range(len(y_pred)):
            tp += int(y_pred[i] == y_true[i])

        return tp / len(y_pred)

    @staticmethod
    def f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if len(y_pred) != len(y_true):
            raise ValueError(
                "length of prediction vector doesnt match the length of true values vector"
            )

        tp = fp = fn = 0
        for i in range(len(y_pred)):
            if y_true[i] == 1:
                tp += int(y_true[i] == y_pred[i])
                fn += int(y_true[i] != y_pred[i])
                continue
            fp += int(y_true[i] != y_pred[i])

        return (2 * tp) / (2 * tp + fp + fn)

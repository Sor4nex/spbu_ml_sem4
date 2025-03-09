import pytest


import numpy as np
import pytest
from src.homeworks.knn.processing import *

# Тесты для функции accuracy
@pytest.mark.parametrize("y_pred, y_true, expected", [
    # Идеальное совпадение
    (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), 1.0),
    # Все предсказания неверны
    (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),
    # Часть предсказаний верна
    (np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]), 0.75),
])
def test_accuracy(y_pred, y_true, expected):
    assert Metrics.accuracy(y_pred, y_true) == expected

# Тест на ошибку при разных длинах массивов
def test_accuracy_value_error():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError):
        Metrics.accuracy(y_pred, y_true)

# Тесты для функции f1_score
@pytest.mark.parametrize("y_pred, y_true, expected", [
    # Идеальное совпадение
    (np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), 1.0),
    # Все предсказания неверны
    (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),
    # Часть предсказаний верна
    (np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]), pytest.approx(0.6666666667)),
    # Все предсказания положительные, но не все верны
    (np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0]), pytest.approx(0.6666666667)),
    # Все предсказания отрицательные, но не все верны
    (np.array([0, 0, 0, 0]), np.array([1, 0, 1, 0]), 0.0),
])
def test_f1_score(y_pred, y_true, expected):
    assert Metrics.f1_score(y_pred, y_true) == expected

# Тест на ошибку при разных длинах массивов
def test_f1_score_value_error():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError):
        Metrics.f1_score(y_pred, y_true)


# ----------------------------
# Тесты для MinMaxScaler
# ----------------------------

@pytest.mark.parametrize("train_data, transform_data, expected", [
    # Простой случай
    (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[2.0, 3.0]]),
            np.array([[0.5, 0.5]])
    ),
    # Отрицательные значения
    (
            np.array([[-1.0, 0.0], [1.0, 2.0]]),
            np.array([[0.0, 1.0]]),
            np.array([[0.5, 0.5]])
    ),
    # Одна колонка с одинаковыми значениями (вызовет деление на 0)
    (
            np.array([[2.0, 5.0], [2.0, 6.0]]),
            np.array([[2.0, 5.0]]),
            np.array([[0.0, 0.0]])  # Ожидаем ошибку, но пример для демонстрации
    ),
])
def test_minmax_scaler(train_data, transform_data, expected):
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    result = scaler.transform(transform_data)

    # Проверяем с точностью до 1e-6 из-за float
    assert np.allclose(result, expected, atol=1e-6)


# Тест на ошибку при transform до fit
def test_minmax_scaler_error():
    scaler = MinMaxScaler()
    with pytest.raises(AttributeError):
        scaler.transform(np.array([[1.0, 2.0]]))


# ----------------------------
# Тесты для MaxAbsScaler
# ----------------------------

@pytest.mark.parametrize("train_data, transform_data, expected", [
    # Простой случай
    (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[3.0, 4.0]]),
            np.array([[1.0, 1.0]])
    ),
    # Отрицательные значения
    (
            np.array([[-2.0, 3.0], [4.0, -6.0]]),
            np.array([[-2.0, 3.0]]),
            np.array([[-0.5, 0.5]])
    ),
    # Нулевые значения
    (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]])  # Вызовет деление на 0
    ),
])
def test_maxabs_scaler(train_data, transform_data, expected):
    scaler = MaxAbsScaler()
    scaler.fit(train_data)
    result = scaler.transform(transform_data)
    assert np.allclose(result, expected, atol=1e-6)


# Тест на ошибку при transform до fit
def test_maxabs_scaler_error():
    scaler = MaxAbsScaler()
    with pytest.raises(AttributeError):
        scaler.transform(np.array([[1.0, 2.0]]))


# Тесты для корректного разделения данных
@pytest.mark.parametrize("data, targets, coef, expected", [
    # Простой случай
    (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),  # data
            np.array([0, 1, 0, 1]),  # targets
            0.25,  # coef
            (
                    np.array([[1, 2], [3, 4], [5, 6]]),  # x_train
                    np.array([0, 1, 0]),  # y_train
                    np.array([[7, 8]]),  # x_test
                    np.array([1])  # y_test
            )
    ),
    # coef = 0 (вся выборка становится тренировочной)
    (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            0.0,
            (
                    np.array([[1, 2], [3, 4], [5, 6]]),
                    np.array([0, 1, 0]),
                    np.array([]),  # x_test пустой
                    np.array([])  # y_test пустой
            )
    ),
    # coef = 1 (вся выборка становится тестовой)
    (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            1.0,
            (
                    np.array([]),  # x_train пустой
                    np.array([]),  # y_train пустой
                    np.array([[1, 2], [3, 4], [5, 6]]),
                    np.array([0, 1, 0])
            )
    ),
    # coef = 0.5 (ровно половина данных)
    (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            np.array([0, 1, 0, 1]),
            0.5,
            (
                    np.array([[1, 2], [3, 4]]),
                    np.array([0, 1]),
                    np.array([[5, 6], [7, 8]]),
                    np.array([0, 1])
            )
    ),
])
def test_train_test_split(data, targets, coef, expected):
    x_train, y_train, x_test, y_test = train_test_split(data, targets, coef)

    # Проверяем тренировочные данные
    assert np.array_equal(x_train, expected[0])
    assert np.array_equal(y_train, expected[1])

    # Проверяем тестовые данные
    assert np.array_equal(x_test, expected[2])
    assert np.array_equal(y_test, expected[3])


# Тест на ошибку при несовпадении длин data и targets
def test_train_test_split_value_error():
    data = np.array([[1, 2], [3, 4]])
    targets = np.array([0])  # Длина не совпадает
    with pytest.raises(ValueError):
        train_test_split(data, targets)


# Тест на пограничный случай: пустые данные
def test_train_test_split_empty_data():
    data = np.array([])
    targets = np.array([])
    x_train, y_train, x_test, y_test = train_test_split(data, targets, 0.2)

    # Ожидаем, что все выборки будут пустыми
    assert x_train.size == 0
    assert y_train.size == 0
    assert x_test.size == 0
    assert y_test.size == 0

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from src.homeworks.knn.knn_classifier import KNNClassifier

# Тесты для predict и predict_proba
@pytest.mark.parametrize("train_data, train_targets, test_data, n_neighbours", [
    # Простой случай
    (
        np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]),  # train_data
        np.array([0, 0, 0, 1, 1, 1]),  # train_targets
        np.array([[2.5, 3.5], [7.5, 8.5]]),  # test_data
        3  # n_neighbours
    ),
    # Один сосед
    (
        np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]),
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([[2.5, 3.5], [7.5, 8.5]]),
        1  # n_neighbours
    ),
    # Все точки в одном классе
    (
        np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]),
        np.array([0, 0, 0, 0, 0, 0]),  # Все точки в классе 0
        np.array([[2.5, 3.5], [7.5, 8.5]]),
        3
    ),
])
def test_knn_classifier(train_data, train_targets, test_data, n_neighbours):
    # Обучаем ваш классификатор
    custom_knn = KNNClassifier(n_neighbours=n_neighbours)
    custom_knn.fit(train_data, train_targets)

    # Обучаем sklearn KNN
    sklearn_knn = KNeighborsClassifier(n_neighbors=n_neighbours)
    sklearn_knn.fit(train_data, train_targets)

    # Сравниваем predict
    custom_predictions = custom_knn.predict(test_data)
    sklearn_predictions = sklearn_knn.predict(test_data)
    assert np.array_equal(custom_predictions, sklearn_predictions)

    # Сравниваем predict_proba
    custom_proba = custom_knn.predict_proba(test_data)
    sklearn_proba = sklearn_knn.predict_proba(test_data)

    # Получаем уникальные классы
    unique_classes = np.unique(train_targets)

    # Проверяем, что вероятности совпадают с точностью до 1e-6
    for i in range(len(test_data)):
        for j, class_label in enumerate(unique_classes):
            custom_prob = custom_proba[i].get(class_label, 0.0)
            sklearn_prob = sklearn_proba[i][j]
            assert pytest.approx(custom_prob, abs=1e-6) == sklearn_prob

# Тест на ошибку при несовпадении длин данных и меток
def test_knn_classifier_value_error():
    train_data = np.array([[1, 2], [2, 3]])
    train_targets = np.array([0])  # Длина не совпадает
    custom_knn = KNNClassifier()
    with pytest.raises(ValueError):
        custom_knn.fit(train_data, train_targets)

# Тест на пустые данные
def test_knn_classifier_empty_data():
    train_data = np.array([])
    train_targets = np.array([])
    custom_knn = KNNClassifier()
    with pytest.raises(ValueError):
        custom_knn.fit(train_data, train_targets)

def test_knn_classifier_random_data():
    # Генерация случайных данных
    n_samples = 100  # Количество точек
    n_features = 5  # Количество признаков
    n_classes = 3  # Количество классов

    # Генерация случайных данных и меток
    train_data = np.random.rand(n_samples, n_features)  # Данные для обучения
    train_targets = np.random.randint(0, n_classes, size=n_samples)  # Метки для обучения
    test_data = np.random.rand(n_samples // 2, n_features)  # Данные для тестирования

    # Обучаем ваш классификатор
    custom_knn = KNNClassifier(n_neighbours=5)
    custom_knn.fit(train_data, train_targets)

    # Обучаем sklearn KNN
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(train_data, train_targets)

    # Сравниваем predict_proba
    custom_proba = custom_knn.predict_proba(test_data)
    sklearn_proba = sklearn_knn.predict_proba(test_data)

    # Получаем уникальные классы
    unique_classes = np.unique(train_targets)

    # Проверяем, что вероятности совпадают с точностью до 1e-6
    for i in range(len(test_data)):
        for j, class_label in enumerate(unique_classes):
            custom_prob = custom_proba[i].get(class_label, 0.0)
            sklearn_prob = sklearn_proba[i][j]
            assert pytest.approx(custom_prob, abs=1e-6) == sklearn_prob

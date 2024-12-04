import numpy as np


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 5 * (x[1] - x[0] ** 2) ** 2


def levy(x):
    """
    Функция Леви для оптимизации в двумерном пространстве.

    Аргументы:
    x : ndarray
        Массив из двух значений [x, y].

    Возвращает:
    float
        Значение функции Леви в точке (x, y).
    """
    w1 = x[0] - 1
    w2 = x[1] - 1
    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 ** 2) * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 ** 2) * (1 + np.sin(2 * np.pi * w2) ** 2)
    return term1 + term2 + term3


def rastrigin(x):
    """
    Функция Растригина для оптимизации в двумерном пространстве.

    Аргументы:
    x : ndarray
        Массив из двух значений [x, y].

    Возвращает:
    float
        Значение функции Растригина в точке (x, y).
    """
    A = 10
    return A * len(x) + sum([xi ** 2 - A * np.cos(2 * np.pi * xi) for xi in x])

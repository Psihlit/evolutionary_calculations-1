import numpy as np


def nelder_mead(f, x_start, tol=1e-6, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=500):
    """
    Реализация алгоритма деформирующихся многогранников (Нелдера-Мида).

    Параметры:
    f: Целевая функция для минимизации.
    x_start: Начальная точка (или массив точек).
    tol: Допустимая ошибка для остановки (критерий сходимости).
    alpha: Коэффициент отражения.
    gamma: Коэффициент растяжения.
    rho: Коэффициент сжатия.
    sigma: Коэффициент редукции.
    max_iter: Максимальное количество итераций.
    """

    # Создаем начальный симплекс на основе начальной точки
    n = len(x_start)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x_start
    for i in range(1, n + 1):
        x = np.copy(x_start)
        x[i - 1] += 0.05  # Добавляем небольшой шаг для каждой координаты
        simplex[i] = x

    # Оцениваем значения функции в вершинах симплекса
    f_values = np.array([f(x) for x in simplex])

    num_iter = 0
    while num_iter < max_iter:
        # Сортируем симплекс и значения функций по возрастанию
        indices = np.argsort(f_values)
        simplex = simplex[indices]
        f_values = f_values[indices]

        # Вычисляем центр тяжести всех точек, кроме наихудшей
        x_centroid = np.mean(simplex[:-1], axis=0)

        # Операция рефлексии (отражение)
        x_reflect = x_centroid + alpha * (x_centroid - simplex[-1])
        f_reflect = f(x_reflect)

        if f_values[0] <= f_reflect < f_values[-2]:
            simplex[-1] = x_reflect
            f_values[-1] = f_reflect
        elif f_reflect < f_values[0]:
            # Операция растяжения
            x_expand = x_centroid + gamma * (x_reflect - x_centroid)
            f_expand = f(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                f_values[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                f_values[-1] = f_reflect
        else:
            # Операция сжатия
            x_contract = x_centroid + rho * (simplex[-1] - x_centroid)
            f_contract = f(x_contract)
            if f_contract < f_values[-1]:
                simplex[-1] = x_contract
                f_values[-1] = f_contract
            else:
                # Операция редукции
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                f_values = np.array([f(x) for x in simplex])

        # Проверка критерия сходимости
        if np.max(np.abs(f_values - f_values[0])) < tol:
            break

        num_iter += 1

    return simplex[0], f_values[0], num_iter



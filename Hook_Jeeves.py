import numpy as np


def hooke_jeeves(f, x_start, step_size=0.5, alpha=2, beta=0.5, tol=1e-6, max_iter=500):
    """
    Реализация алгоритма Хука-Дживса.

    Параметры:
    f: Целевая функция для минимизации.
    x_start: Начальная точка.
    step_size: Начальный шаг поиска.
    alpha: Коэффициент увеличения шага при шаге по направлению.
    beta: Коэффициент уменьшения шага при уменьшении размера шага.
    tol: Допустимая ошибка для остановки (критерий сходимости).
    max_iter: Максимальное количество итераций.
    """

    def explore(x, step_size):
        """
        Поисковый шаг: исследование окрестности точки.
        """
        x_new = x.copy()
        n = len(x)
        for i in range(n):
            # Проверка движения вдоль положительного направления оси
            x_temp = x_new.copy()
            x_temp[i] += step_size
            if f(x_temp) < f(x_new):
                x_new = x_temp
            else:
                # Проверка движения вдоль отрицательного направления оси
                x_temp[i] -= 2 * step_size
                if f(x_temp) < f(x_new):
                    x_new = x_temp
        return x_new

    # Начальная точка
    x_base = x_start
    x_new = explore(x_base, step_size)
    num_iter = 0

    while num_iter < max_iter:
        # Шаг по направлению (если был улучшен поисковый шаг)
        if np.allclose(x_new, x_base, atol=tol):
            step_size *= beta  # Уменьшаем шаг, если нет улучшений
            if step_size < tol:
                break
        else:
            # Шаг по направлению
            x_direction = x_new + alpha * (x_new - x_base)
            if f(x_direction) < f(x_new):
                x_base = x_new
                x_new = explore(x_direction, step_size)
            else:
                x_base = x_new
                x_new = explore(x_base, step_size)

        num_iter += 1

    return x_new, f(x_new), num_iter



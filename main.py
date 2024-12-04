import time
import matplotlib.pyplot as plt
import numpy as np
from Hook_Jeeves import *
from Nelder_Mead import *
from functions import *

# Целевые функции и их имена
functions = [rosenbrock, levy, rastrigin]
function_names = ["Функция Розенброка", "Функция Леви", "Функция Растригина"]

# Начальная точка
x_start = np.array([2.0, 0.0])

# Для сбора данных
results = {
    "Hooke-Jeeves": {"iterations": [], "times": []},
    "Nelder-Mead": {"iterations": [], "times": []},
}

# Тестирование алгоритмов
for func, name in zip(functions, function_names):
    print(f"Testing on {name} function...")

    # Hooke-Jeeves
    start_time = time.time()
    _, _, iter_hj = hooke_jeeves(func, x_start)
    elapsed_hj = time.time() - start_time

    results["Hooke-Jeeves"]["iterations"].append(iter_hj)
    results["Hooke-Jeeves"]["times"].append(elapsed_hj)

    # Nelder-Mead
    start_time = time.time()
    _, _, iter_nm = nelder_mead(func, x_start)
    elapsed_nm = time.time() - start_time

    results["Nelder-Mead"]["iterations"].append(iter_nm)
    results["Nelder-Mead"]["times"].append(elapsed_nm)

# Визуализация
x = np.arange(len(function_names))  # Позиции для баров

# График количества итераций
plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, results["Hooke-Jeeves"]["iterations"], width=0.4, label="Хука-Дживса", color="blue")
plt.bar(x + 0.2, results["Nelder-Mead"]["iterations"], width=0.4, label="Нелдера-Мида", color="green")
plt.xticks(x, function_names)
plt.ylabel("Итерации")
plt.title("Количество итераций")
plt.legend()
plt.grid(axis="y")
plt.show()

# График времени выполнения
plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, results["Hooke-Jeeves"]["times"], width=0.4, label="Хука-Дживса", color="blue")
plt.bar(x + 0.2, results["Nelder-Mead"]["times"], width=0.4, label="Нелдера-Мида", color="green")
plt.xticks(x, function_names)
plt.ylabel("Время выполнения (сек)")
plt.title("Время выполнения")
plt.legend()
plt.grid(axis="y")
plt.show()

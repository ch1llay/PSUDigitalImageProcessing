"""Лабораторная работа 1. Основы NumPy.

Задание: средствами библиотеки NumPy сгенерировать матрицу со случайными
значениями, инвертировать её (найти обратную матрицу) и найти индексы
максимальных элементов в обращённой матрице.
"""

import numpy as np


def generate_random_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Сгенерировать случайную обратимую матрицу n x n.

    Случайная матрица почти всегда обратима, но на всякий случай повторяем
    генерацию, если она оказалась вырожденной (определитель близок к нулю).
    """
    while True:
        matrix = rng.uniform(-10.0, 10.0, size=(n, n))
        if abs(np.linalg.det(matrix)) > 1e-6:
            return matrix


def max_indices(matrix: np.ndarray):
    """Найти индексы максимальных элементов матрицы.

    Возвращает:
        flat_index   - индекс глобального максимума в "развёрнутом" виде;
        global_index - (row, col) глобального максимума;
        rows_argmax  - индекс максимума в каждой строке;
        cols_argmax  - индекс максимума в каждом столбце.
    """
    flat_index = int(np.argmax(matrix))
    global_index = np.unravel_index(flat_index, matrix.shape)
    rows_argmax = np.argmax(matrix, axis=1)   # по строкам
    cols_argmax = np.argmax(matrix, axis=0)   # по столбцам
    return flat_index, global_index, rows_argmax, cols_argmax


def main() -> None:
    rng = np.random.default_rng(seed=42)  # seed для воспроизводимости
    n = 5

    matrix = generate_random_matrix(n, rng)
    inverse = np.linalg.inv(matrix)

    np.set_printoptions(precision=3, suppress=True)
    print("Исходная случайная матрица A:")
    print(matrix)
    print("\nОбратная матрица A^-1:")
    print(inverse)

    # Проверка корректности: A @ A^-1 должно быть близко к единичной матрице.
    print("\nПроверка A @ A^-1 ~ E:", np.allclose(matrix @ inverse, np.eye(n)))

    flat_index, global_index, rows_argmax, cols_argmax = max_indices(inverse)

    print("\n--- Индексы максимальных элементов обращённой матрицы ---")
    print(f"Глобальный максимум: значение = {inverse[global_index]:.3f}, "
          f"индекс (строка, столбец) = {global_index}, "
          f"плоский индекс = {flat_index}")
    print(f"Индексы максимумов по строкам:   {rows_argmax}")
    print(f"Индексы максимумов по столбцам:  {cols_argmax}")


if __name__ == "__main__":
    main()

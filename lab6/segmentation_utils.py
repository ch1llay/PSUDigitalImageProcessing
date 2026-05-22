"""Вспомогательные функции сегментации для лабораторной работы 6."""

import numpy as np


def region_growing(image: np.ndarray, seeds, threshold: float) -> np.ndarray:
    """Наращивание областей (region growing) по 4-связности.

    Начинаем из заданных точек-семян и присоединяем соседние пиксели, если их
    цвет близок к цвету семени (среднее абсолютное отличие по каналам меньше
    порога). Работает на любом 3-канальном изображении (например, HSV).
    """
    h, w = image.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    connectivity = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for seed_x, seed_y in seeds:
        queue = [(seed_x, seed_y)]
        seed_color = image[seed_x, seed_y, :].astype(np.int32)
        while queue:
            x, y = queue.pop(0)
            if x < 0 or x >= h or y < 0 or y >= w:
                continue
            if region_mask[x, y] != 0:
                continue
            similarity = np.mean(np.abs(image[x, y, :].astype(np.int32) - seed_color))
            if similarity < threshold:
                region_mask[x, y] = 255
                for dx, dy in connectivity:
                    queue.append((x + dx, y + dy))

    return region_mask

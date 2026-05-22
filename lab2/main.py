"""Лабораторная работа 2. Эквализация гистограммы.

Задание (одинаковое для всех): самостоятельно реализовать эквализацию
гистограммы изображения, не пользуясь готовой cv2.equalizeHist.

Идея: выравниваем распределение яркостей, растягивая его на весь диапазон
[0, 255] через функцию распределения (CDF) гистограммы.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize(gray: np.ndarray) -> np.ndarray:
    """Ручная эквализация гистограммы полутонового изображения.

    1. Считаем гистограмму яркостей.
    2. Накопленная сумма (CDF) и её нормировка к [0, 1].
    3. Таблица соответствия (LUT) старая яркость -> новая (умножаем на 255).
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    lut = np.round(255 * cdf_normalized).astype(np.uint8)
    return lut[gray]


def main() -> None:
    image = cv2.imread("lenna.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = equalize(gray)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Исходное изображение")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(equalized, cmap="gray")
    plt.title("После эквализации")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.hist(gray.ravel(), 256, range=(0, 255))
    plt.title("Гистограмма исходного")
    plt.xlabel("Яркость")
    plt.ylabel("Частота")

    plt.subplot(2, 2, 4)
    plt.hist(equalized.ravel(), 256, range=(0, 255))
    plt.title("Гистограмма после эквализации")
    plt.xlabel("Яркость")
    plt.ylabel("Частота")

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()

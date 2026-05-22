"""Лабораторная работа 5. Цветовые модели и цветовая сегментация.

Задание: выполнить цветовую сегментацию объекта на изображении, подобрав
пороги вручную. Здесь выделяем апельсины на тёмном фоне (рядом стоит синяя
бутылка, которая не должна попасть в маску).

Подход:
    1. Перевод BGR -> HSV: оранжевый цвет компактно лежит по тону (Hue),
       что удобнее, чем разделять каналы RGB.
    2. cv2.inRange с вручную подобранными порогами по H, S, V.
    3. Морфологические открытие/закрытие убирают мелкий шум и дырки.
    4. Накладываем маску: фон делаем чёрным.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "18d33c106910765_5f9aa887279b1.jpg"

# Пороги для оранжевого в HSV (OpenCV: H в диапазоне 0..179).
# Нижнюю границу по яркости/насыщенности держим невысокой, чтобы захватить
# и затенённые апельсины, но не настолько, чтобы попал тёмный фон.
LOWER = np.array([3, 80, 45], dtype=np.uint8)
UPPER = np.array([22, 255, 255], dtype=np.uint8)


def main() -> None:
    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER, UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    result = img_rgb.copy()
    result[mask == 0] = 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Исходное")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Маска апельсинов")
    axes[2].imshow(result)
    axes[2].set_title("Сегментированные апельсины")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()

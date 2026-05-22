"""Лабораторная работа 4. Удаление горизонтальных линий и бинаризация.

Задание: на изображении с текстом, перечёркнутым горизонтальными линиями,
удалить линии средствами морфологии и восстановить текст, затем выполнить
бинаризацию результата.

Подход:
    1. Перевод в полутон и инверсия (текст и линии становятся светлыми).
    2. Морфологическое открытие широким горизонтальным ядром выделяет именно
       длинные горизонтальные линии (короткие штрихи букв отсеиваются).
    3. cv2.inpaint (алгоритм Telea) "закрашивает" области линий по соседям.
    4. Бинаризация Otsu и адаптивная для подготовки к OCR.
"""

import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "as.png"


def remove_lines(image_path: str):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)

    # Ширину ядра берём заметно больше ширины символа, чтобы поймать линии.
    kernel_len = max(40, img.shape[1] // 20)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # Чуть расширяем маску линий, чтобы перекрыть их края.
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    result = cv2.inpaint(gray, lines, 3, cv2.INPAINT_TELEA)

    _, otsu = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )

    cv2.imwrite("output.png", result)
    cv2.imwrite("output_binary.png", otsu)

    panel = [
        (gray, "Оригинал (grayscale)"),
        (inverted, "Инверсия"),
        (lines, "Найденные линии"),
        (result, "После удаления линий"),
        (otsu, "Бинаризация Otsu"),
        (adaptive, "Адаптивная (Gaussian)"),
    ]
    plt.figure(figsize=(15, 8))
    for i, (image, title) in enumerate(panel, 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    remove_lines(IMAGE_PATH)

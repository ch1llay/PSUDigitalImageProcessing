"""Лабораторная работа 3. Шум и фильтрация.

Задание: добавить к изображению модели шума (соль-перец, гауссов), затем
подавить шум разными фильтрами и оценить качество восстановления метриками
PSNR и SSIM относительно чистого изображения.

Метрики:
    PSNR - пиковое отношение сигнал/шум (чем больше, тем ближе к оригиналу);
    SSIM - индекс структурного сходства из scikit-image (1.0 = идентичны).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

IMAGE_PATH = "b2a1f8f09f5d085eiokk.jpg"


def add_salt_pepper(gray: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Шум соль-перец: часть пикселей делаем белыми, часть чёрными."""
    noisy = gray.copy()
    n = int(amount * gray.size / 2)
    ys = np.random.randint(0, gray.shape[0], n)
    xs = np.random.randint(0, gray.shape[1], n)
    noisy[ys, xs] = 255  # соль
    ys = np.random.randint(0, gray.shape[0], n)
    xs = np.random.randint(0, gray.shape[1], n)
    noisy[ys, xs] = 0    # перец
    return noisy


def add_gaussian(gray: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Аддитивный гауссов шум."""
    noise = np.random.normal(0, sigma, gray.shape)
    noisy = gray.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def metrics(reference: np.ndarray, image: np.ndarray) -> str:
    return f"PSNR = {psnr(reference, image):.2f}\nSSIM = {ssim(reference, image):.3f}"


def show_panel(reference, items, title, filename):
    plt.figure(figsize=(15, 8))
    plt.suptitle(title, fontsize=14, fontweight="bold")
    for i, (img, name) in enumerate(items, 1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)


def main() -> None:
    np.random.seed(42)
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    # --- Шум соль-перец и его подавление ---
    sp = add_salt_pepper(gray)
    sp_panel = [
        (gray, "Оригинал"),
        (sp, f"Шум соль-перец\n{metrics(gray, sp)}"),
        (cv2.medianBlur(sp, 5), f"Медианный 5x5\n{metrics(gray, cv2.medianBlur(sp, 5))}"),
        (cv2.blur(sp, (5, 5)), f"Среднее 5x5\n{metrics(gray, cv2.blur(sp, (5, 5)))}"),
        (cv2.GaussianBlur(sp, (5, 5), 0), f"Гаусс 5x5\n{metrics(gray, cv2.GaussianBlur(sp, (5, 5), 0))}"),
        (cv2.bilateralFilter(sp, 9, 75, 75), f"Билатеральный\n{metrics(gray, cv2.bilateralFilter(sp, 9, 75, 75))}"),
    ]
    show_panel(gray, sp_panel, "Шум соль-перец: медианный фильтр работает лучше всего", "result_salt_pepper.png")

    # --- Гауссов шум и его подавление ---
    gn = add_gaussian(gray)
    gn_panel = [
        (gray, "Оригинал"),
        (gn, f"Гауссов шум\n{metrics(gray, gn)}"),
        (cv2.GaussianBlur(gn, (5, 5), 0), f"Гаусс 5x5\n{metrics(gray, cv2.GaussianBlur(gn, (5, 5), 0))}"),
        (cv2.blur(gn, (5, 5)), f"Среднее 5x5\n{metrics(gray, cv2.blur(gn, (5, 5)))}"),
        (cv2.medianBlur(gn, 5), f"Медианный 5x5\n{metrics(gray, cv2.medianBlur(gn, 5))}"),
        (cv2.bilateralFilter(gn, 9, 75, 75), f"Билатеральный\n{metrics(gray, cv2.bilateralFilter(gn, 9, 75, 75))}"),
    ]
    show_panel(gray, gn_panel, "Гауссов шум: линейные фильтры справляются лучше", "result_gaussian.png")

    plt.show()


if __name__ == "__main__":
    main()

"""Лабораторная работа 6. Сегментация изображений (продвинутые методы).

Задание: сегментировать объекты (апельсины) разными методами и сравнить:
    1. Цветовая маска (HSV) + watershed - разделяет слипшиеся апельсины на
       отдельные объекты (distance transform -> локальные максимумы -> метки).
    2. K-средних по цвету - кластеризация всех пикселей и выбор "оранжевого"
       кластера.
    3. Region growing - наращивание областей из центров найденных апельсинов.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import segmentation_utils

IMAGE_PATH = "18d33c106910765_5f9aa887279b1.jpg"
LOWER = np.array([5, 90, 90], dtype=np.uint8)
UPPER = np.array([25, 255, 255], dtype=np.uint8)


def orange_mask(hsv: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(hsv, LOWER, UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def segment_watershed(image_rgb: np.ndarray, mask: np.ndarray):
    """Разделяет слипшиеся объекты маски на отдельные через watershed."""
    distance = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=40, labels=mask)
    peaks = np.zeros_like(distance, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers = ndimage.label(peaks, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers, mask=mask)

    contoured = image_rgb.copy()
    count = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        obj = np.uint8(labels == label) * 255
        cnts, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            continue
        cv2.drawContours(contoured, [c], -1, (36, 255, 12), 3)
        count += 1
    return labels, contoured, count


def segment_kmeans(image_rgb: np.ndarray, k: int = 4):
    """Кластеризация пикселей по цвету и выбор оранжевого кластера."""
    pixels = np.float32(image_rgb.reshape(-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=4).fit(pixels)
    centers = kmeans.cluster_centers_
    # Оранжевый: высокий R, средний G, низкий B.
    scores = centers[:, 0] - centers[:, 2]
    orange_cluster = int(np.argmax(scores))
    mask = (kmeans.labels_ == orange_cluster).reshape(image_rgb.shape[:2])
    result = image_rgb.copy()
    result[~mask] = 0
    return result


def main() -> None:
    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = orange_mask(hsv)
    labels, contoured, count = segment_watershed(img_rgb, mask)
    kmeans_result = segment_kmeans(img_rgb)

    # Region growing из центров апельсинов (центроиды компонент watershed).
    seeds = []
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.where(labels == label)
        if len(ys) > 500:
            seeds.append((int(ys.mean()), int(xs.mean())))
    rg_mask = segmentation_utils.region_growing(hsv, seeds, threshold=18)
    rg_result = cv2.bitwise_and(img_rgb, img_rgb, mask=rg_mask)

    panel = [
        (img_rgb, "Исходное", None),
        (mask, "Цветовая маска (HSV)", "gray"),
        (labels, f"Watershed: {count} объектов", "nipy_spectral"),
        (contoured, "Контуры апельсинов", None),
        (kmeans_result, "K-средних по цвету", None),
        (rg_result, "Region growing", None),
    ]
    plt.figure(figsize=(16, 10))
    for i, (image, title, cmap) in enumerate(panel, 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()

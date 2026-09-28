from __future__ import annotations

import cv2
import numpy as np


def _estimate_background_color(img: np.ndarray) -> np.ndarray:
    # Estimate colour from corners
    h, w = img.shape[:2]
    corners = np.concatenate([
        img[:5, :5].reshape(-1, 3),
        img[:5, w - 5:].reshape(-1, 3),
        img[h - 5:, :5].reshape(-1, 3),
        img[h - 5:, w - 5:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def normalize_to_canvas(
    img: np.ndarray,
    canvas_size: int = 512,
) -> np.ndarray:
    h, w = img.shape[:2]

    scale = canvas_size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    # Fill padding
    bg_color = _estimate_background_color(img).astype(np.uint8)
    canvas = np.full(
        (canvas_size, canvas_size, 3),
        bg_color,
        dtype=np.uint8,
    )

    top = (canvas_size - new_h) // 2
    left = (canvas_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def prep_reference_image(
    img: np.ndarray,
    canvas_size: int = 512,
) -> np.ndarray:
    return normalize_to_canvas(img, canvas_size)
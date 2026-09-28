from __future__ import annotations

import cv2
import numpy as np


def _circular_mask(
    h: int,
    w: int,
    radius_frac: float = 0.48,
) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    radius = radius_frac * min(h, w)

    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def wear_pattern_features(
    img_rgb: np.ndarray,
    radius_frac: float = 0.48,
) -> np.ndarray:
    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    mask = _circular_mask(*gray.shape, radius_frac)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    features = []
    for channel in (gx, gy, magnitude):
        values = channel[mask]
        features.extend([
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(np.median(values)),
        ])

    return np.array(features, dtype=np.float32)


def color_profile_features(
    img_rgb: np.ndarray,
    radius_frac: float = 0.48,
    n_clusters: int = 5,
) -> np.ndarray:
    hsv = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2HSV,
    ).astype(np.float32)
    mask = _circular_mask(*hsv.shape[:2], radius_frac)
    pixels = np.ascontiguousarray(hsv[mask], dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        n_clusters,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()
    proportions = np.array(
        [(labels == i).mean() for i in range(n_clusters)],
        dtype=np.float32,
    )

    order = np.argsort(-proportions)
    centers = centers[order]
    proportions = proportions[order]

    return np.concatenate([
        centers.flatten(),
        proportions,
    ]).astype(np.float32)


def luminance_features(
    img_rgb: np.ndarray,
    radius_frac: float = 0.48,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    values = hsv[..., 2].astype(np.float32)
    mask = _circular_mask(*values.shape, radius_frac)
    values = values[mask]

    return np.array([
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(np.median(values)),
        float(values.std()),
    ], dtype=np.float32)


def extract_handcrafted_features(
    img_rgb: np.ndarray,
    n_color_clusters: int = 5,
    radius_frac: float = 0.48,
) -> np.ndarray:
    wear = wear_pattern_features(img_rgb, radius_frac)
    color = color_profile_features(
        img_rgb,
        radius_frac,
        n_color_clusters,
    )
    luminance = luminance_features(img_rgb, radius_frac)

    return np.concatenate([
        wear,
        color,
        luminance,
    ]).astype(np.float32)


def feature_vector_size(n_color_clusters: int = 5) -> int:
    return 12 + n_color_clusters * 4 + 5
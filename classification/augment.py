from __future__ import annotations

import random
from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class SwapStudioBackground(A.ImageOnlyTransform):
    def __init__(self, tolerance: int = 18, p: float = 0.9):
        super().__init__(p=p)
        self.tolerance = tolerance

    def _make_synthetic_background(self, h: int, w: int) -> np.ndarray:
        if random.random() < 0.5:
            c1 = np.random.randint(20, 235, size=3)
            c2 = np.random.randint(20, 235, size=3)

            if random.random() < 0.5:
                ramp = np.linspace(0, 1, w)[:, None]
                grad = c1 * (1 - ramp) + c2 * ramp
                background = np.tile(grad[None, :, :], (h, 1, 1))
            else:
                ramp = np.linspace(0, 1, h)[:, None]
                grad = c1 * (1 - ramp) + c2 * ramp
                background = np.tile(grad[:, None, :], (1, w, 1))

            return background.astype(np.uint8)

        color = np.random.randint(15, 220, size=3, dtype=np.uint8)
        background = np.tile(color, (h, w, 1))
        noise = np.random.normal(0, 6, (h, w, 3))

        return np.clip(
            background.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]

        # Estimate background colour
        corners = np.concatenate([
            img[:5, :5].reshape(-1, 3),
            img[:5, w - 5:].reshape(-1, 3),
            img[h - 5:, :5].reshape(-1, 3),
            img[h - 5:, w - 5:].reshape(-1, 3),
        ])

        bg_color = np.median(corners, axis=0)
        diff = np.linalg.norm(
            img.astype(np.int16) - bg_color,
            axis=-1,
        )
        bg_mask = (diff < self.tolerance).astype(np.uint8) * 255

        # Keep edge connected background
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        working = bg_mask.copy()

        for y, x in [
            (0, 0),
            (0, w - 1),
            (h - 1, 0),
            (h - 1, w - 1),
        ]:
            if working[y, x] == 255:
                cv2.floodFill(working, flood_mask, (x, y), 128)

        connected_bg = (working == 128).astype(np.uint8) * 255

        if connected_bg.mean() < 5:
            return img

        alpha = cv2.GaussianBlur(
            connected_bg,
            (9, 9),
            0,
        ).astype(np.float32)[..., None] / 255.0

        background = self._make_synthetic_background(h, w).astype(np.float32)
        img = img.astype(np.float32)

        return (img * (1 - alpha) + background * alpha).astype(np.uint8)


def build_train_transform(
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    dropout_px = max(1, int(image_size * 0.05))

    return A.Compose([
        # Background and geometry
        SwapStudioBackground(p=0.8),
        A.Rotate(
            limit=180,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            p=1.0,
        ),
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.05, 0.05),
            p=0.7,
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.5),

        # Lighting
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.8,
        ),
        A.RandomGamma(gamma_limit=(80, 125), p=0.5),
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.8, 1.2),
            p=0.4,
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            src_radius=int(image_size * 0.2),
            p=0.3,
        ),

        # Image quality
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.4),
        A.GaussNoise(std_range=(0.04, 0.15), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(1, dropout_px),
            hole_width_range=(1, dropout_px),
            fill=0,
            p=0.2,
        ),

        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_eval_transform(
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
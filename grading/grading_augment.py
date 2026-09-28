from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_grading_train_transform(
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    to_tensor: bool = True,
) -> A.Compose:
    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(225, 225, 225),
        ),
        A.Rotate(
            limit=180,
            border_mode=0,
            fill=(225, 225, 225),
            p=0.8,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=(-0.08, 0.05),
            rotate_limit=0,
            border_mode=0,
            fill=(225, 225, 225),
            p=0.5,
        ),
        A.Perspective(
            scale=(0.01, 0.04),
            keep_size=True,
            p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.5,
        ),
        A.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.03,
            p=0.4,
        ),
        A.GaussNoise(
            std_range=(0.01, 0.03),
            p=0.25,
        ),
        A.ImageCompression(
            quality_range=(80, 100),
            p=0.3,
        ),
    ]

    if to_tensor:
        transforms.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return A.Compose(transforms)


# Resize and pad for evaluation
def build_grading_eval_transform(
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    to_tensor: bool = True,
) -> A.Compose:
    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(225, 225, 225),
        ),
    ]

    if to_tensor:
        transforms.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return A.Compose(transforms)
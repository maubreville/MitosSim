"""Augmentation pipeline: standard geometric/photometric augmentation plus
albumentations' native H&E stain augmentation (`A.HEStain`, added in
albumentations 2.0 -- needs that version or newer, the installed 1.3.1
predates it).

Train-time only; eval uses just the resize/normalize tail.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 128) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.HEStain(method="random_preset", intensity_scale_range=(0.7, 1.3), intensity_shift_range=(-0.2, 0.2), p=0.7),
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transform(image_size: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

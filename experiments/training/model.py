"""Basic timm EfficientNet classifier for the 4 mitosis-phase classes."""
from __future__ import annotations

import timm
import torch.nn as nn

from dataset import CLASSES


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES))


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

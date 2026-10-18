"""

This file is part of MitosSim.

This file defines spindle poles (which we use in every phase generator)

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Spindle:
    poles: np.ndarray  # (n_poles, 3), world frame


def bipolar_spindle(half_length: float) -> Spindle:
    return Spindle(poles=np.array([[0.0, 0.0, -half_length], [0.0, 0.0, half_length]]))

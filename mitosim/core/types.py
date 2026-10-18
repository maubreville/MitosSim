"""

This file is part of MitosSim.


Core data types shared across the whole pipeline.

CellState is the single object passed from phase generators to rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class Phase(Enum):
    PROPHASE = auto()
    PROMETAPHASE = auto()
    METAPHASE = auto()
    ANAPHASE = auto()
    TELOPHASE = auto()


@dataclass(frozen=True)
class CellState:
    chromatids: list[Any]   # geometry in world coordinates
    poles: np.ndarray       # (n_poles, 3)
    phase: Phase
    cell_radius: float       # rounded-up mitotic cell boundary
    provenance: dict = field(default_factory=dict)  # seed, config hash, parameter draws

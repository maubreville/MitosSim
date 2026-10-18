"""

This file is part of MitosSim.

Here, we define the solid-mass geometric primitive for the sectioning core.

The world frame is represented in micrometers. We always rotate the geometry into place,
never the voxel grid, to simplify sampling.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray  # (3,) world-frame centre, micrometres
    radius: float

    def indicator(self, points: np.ndarray) -> np.ndarray:
        """points: (..., 3) world coords -> boolean occupancy."""
        d2 = np.sum((points - self.center) ** 2, axis=-1)
        return d2 <= self.radius**2

    def analytic_volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.radius**3

    def slab_projected_area(self, s: float, t: float) -> float:
        zc = self.center[2]
        r = self.radius
        lo = max(zc - r, s - t / 2)
        hi = min(zc + r, s + t / 2)
        if hi <= lo:
            return 0.0
        a, b = lo - zc, hi - zc
        z0 = min(max(0.0, a), b)
        return float(np.pi * max(r**2 - z0**2, 0.0))

"""

This file is part of MitosSim.


Chromosome/chromatid geometry: a condensed chromatid is a swept tube along a cubic spline centreline, with a
chevron bend at the centromere. According to Booth et al. 2016, the tube's radius is constant along each arm 
with a pinch (smaller radius) at the centromere.

The shape (control points and radii) is generated once per chromosome and then placed into the world frame (place_chromatid).
Sister chromatids are separated by a lateral space (according to the phase of the mitosis).

"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from mitosim.geometry.karyotype import ChromosomeSpec

#: Centromere radius as a fraction of the (otherwise constant) arm radius,
#: and how localized that constriction is in normalized arc-length t (the
#: centromere always sits at t=0.5 -- see place_chromatid). Applied as a
#: Gaussian notch (see place_chromatid) rather than folded into the same
#: control-point spline the centreline uses: fitting [1,1,pinch,1,1]
#: through that shared cubic-spline basis overshoots above 1.0 just either
#: side of the dip (verified numerically), which would read as the arm
#: bulging right next to the constriction -- not the flat-with-a-pinch
#: profile Booth et al. 2016 supports.
_CENTROMERE_PINCH = 0.55
_CENTROMERE_PINCH_WIDTH = 0.05


def chromosome_shape(
    rng: np.random.Generator,
    spec: ChromosomeSpec,
    bend_deg_range: tuple[float, float] = (100.0, 160.0),
) -> tuple[np.ndarray, float]:
    """ Defines the control points and arm radius for one chromatid.

    In the local frame, the centromere sits at the origin; the two arms open around local +x with a random bend angle. 
    Control points get small random lateral jitter for a natural, non-ruler-straight look. 
    
    This function returns (control_points (5,3), arm_radius).
    """
    frac_p = spec.centromere_index / 100.0
    frac_q = 1.0 - frac_p
    bend = np.deg2rad(rng.uniform(*bend_deg_range))
    dir_p = np.array([np.cos(bend / 2), np.sin(bend / 2), 0.0])
    dir_q = np.array([np.cos(bend / 2), -np.sin(bend / 2), 0.0])
    len_p, len_q = frac_p * spec.length, frac_q * spec.length

    def jitter(scale: float) -> np.ndarray:
        return rng.normal(scale=scale, size=3) * np.array([0.3, 1.0, 1.0])

    tel_p = dir_p * len_p
    mid_p = dir_p * len_p * 0.5 + jitter(0.06 * spec.length)
    centromere = np.zeros(3)
    mid_q = dir_q * len_q * 0.5 + jitter(0.06 * spec.length)
    tel_q = dir_q * len_q
    control = np.stack([tel_p, mid_p, centromere, mid_q, tel_q])

    return control, spec.radius


def _rotation_aligning_x_to(direction: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """ Builds a 3×3 rotation matrix that maps the x-axis onto a given direction, with an optional roll.
    """
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ey = np.cross(direction, helper)
    ey /= np.linalg.norm(ey)
    ez = np.cross(direction, ey)
    ey_r = np.cos(roll) * ey + np.sin(roll) * ez
    ez_r = -np.sin(roll) * ey + np.cos(roll) * ez
    return np.stack([direction, ey_r, ez_r], axis=1)


def rotation_aligning_x_to(direction: np.ndarray, roll: float = 0.0) -> np.ndarray:
    return _rotation_aligning_x_to(direction, roll)


# Chromatid is our central class for chromatids. 
@dataclass(frozen=True)
class Chromatid:
    points: np.ndarray        # (n_samples, 3) fine centreline samples in world frame
    radii: np.ndarray         # (n_samples,) tube radius at each sample
    chromosome_id: int        # Which chromosome does this represent?
    sister: int               # 0 or 1 -- which of the two sister chromatids?
    centromere: np.ndarray    # (3,) exact centromere point in world frame

    def __post_init__(self):
        object.__setattr__(self, "_tree", None)

    def indicator(self, query_points: np.ndarray) -> np.ndarray:
        # This is a membership test: for each query point, is it inside the tube?
        # We use a CKDTree to check for distance, which we build lazily upon first request.
        if self._tree is None:
            object.__setattr__(self, "_tree", cKDTree(self.points))
        out_shape = query_points.shape[:-1]
        flat = query_points.reshape(-1, 3)
        dist, idx = self._tree.query(flat)
        inside = dist <= self.radii[idx]
        return inside.reshape(out_shape)

    def bounding_half_width(self) -> float:
        centre = self.points.mean(axis=0)
        return float(np.max(np.linalg.norm(self.points - centre, axis=-1)) + self.radii.max())


def enclosing_radius(chromatids: list["Chromatid"], margin: float = 0.5) -> float:
    """ This function computes the radius of a sphere centered at the world origin that contains every chromatid, 
     plus a safety margin (`margin`). It's meant to define the cell boundary (CellState.cell_radius) 
     from the geometry that was actually built.
    """
    if not chromatids:
        return margin
    furthest = 0.0
    for c in chromatids:
        reach = np.linalg.norm(c.points, axis=-1) + c.radii
        furthest = max(furthest, float(reach.max()))
    return furthest + margin


def _spline_basis(n_control: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """ Precomputes a matrix that turns a small set of spline control values into a densely sampled curve. 
    Fitting the cubic spline per chromatid is the expensive part, so this does it once and reduces every 
    later evaluation to a matrix multiply, quickly evaluated on modern hardware.
    """
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    t = np.linspace(0.0, 1.0, n_samples)
    basis = np.empty((n_samples, n_control))
    centromere_basis = np.empty(n_control)
    for i in range(n_control):
        unit = np.zeros(n_control)
        unit[i] = 1.0
        spline = CubicSpline(t_ctrl, unit)
        basis[:, i] = spline(t)
        centromere_basis[i] = spline(0.5)
    return basis, centromere_basis


def place_chromatid(
    control: np.ndarray,
    radius: float,
    chromosome_id: int,
    sister: int,
    position: np.ndarray,
    rotation: np.ndarray,
    lateral_offset: float = 0.0,
    n_samples: int = 60,
) -> Chromatid:
    """Sample the spline with a fine grid, push sisters apart along local z direction (perpendicular to the V-plane) before 
    placing, then rotate+translate into the world frame again."""
    basis, centromere_basis = _spline_basis(len(control), n_samples)
    local_points = basis @ control + np.array([0.0, 0.0, lateral_offset])

    t = np.linspace(0.0, 1.0, n_samples)
    notch = np.exp(-0.5 * ((t - 0.5) / _CENTROMERE_PINCH_WIDTH) ** 2)
    local_radii = radius * (1.0 - (1.0 - _CENTROMERE_PINCH) * notch)
    world_points = local_points @ rotation.T + position

    local_centromere = centromere_basis @ control + np.array([0.0, 0.0, lateral_offset])
    world_centromere = local_centromere @ rotation.T + position
    return Chromatid(
        points=world_points,
        radii=local_radii,
        chromosome_id=chromosome_id,
        sister=sister,
        centromere=world_centromere,
    )

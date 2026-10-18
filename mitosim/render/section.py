"""Sectioning: spindle-orientation sampling, slab-offset sampling, and the numeric (voxel-based) projected-area measurement 
used to cross-check the closed-form formulas in geometry/shapes.py.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_WORLD_Z = np.array([0.0, 0.0, 1.0])


def sample_orientation(rng: np.random.Generator, n: int) -> np.ndarray:
    """n unit vectors uniform on the sphere -- the spindle-axis orientation.

    """
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tilt_from_normal(orientation: np.ndarray, normal: np.ndarray = _WORLD_Z) -> np.ndarray:
    """Tilt (radians, in [0, pi/2]): angle between the spindle axis and the section PLANE (not its normal). For a 
    uniformly-oriented spindle, sin(tilt) is uniform on [0, 1] -- half of all spindles are tilted more than 30 degrees.
    """
    cos_to_normal = np.clip(np.abs(orientation @ normal), -1.0, 1.0)
    return np.arcsin(cos_to_normal)


def orientation_from_tilt_azimuth(tilt: float, azimuth: float) -> np.ndarray:
    """Inverse of tilt_from_normal: build the unit orientation vector for a given tilt (radians, angle from the 
    section plane) and azimuth (radians).
    """
    return np.array([np.cos(tilt) * np.cos(azimuth), np.cos(tilt) * np.sin(azimuth), np.sin(tilt)])


def sample_offset(rng: np.random.Generator, extent: float, n: int) -> np.ndarray:
    """Slab-centre offset s, uniform over [-extent, extent]. 
    """
    return rng.uniform(-extent, extent, size=n)


def fast_slab_mask(chromatids, s: float, t: float, spacing: float = 0.2) -> tuple[np.ndarray, float]:
    """Fast production path for population experiments: the 2D projected boolean mask of chromatin within the 
    slab z in [s-t/2, s+t/2], plus the pixel area.
    """
    z0, z1 = s - t / 2, s + t / 2
    xs_list, ys_list, r_list = [], [], []
    for c in chromatids:
        z = c.points[:, 2]
        r = c.radii
        lo = np.maximum(z - r, z0)
        hi = np.minimum(z + r, z1)
        touches = hi > lo
        if not touches.any():
            continue
        z_lo = lo[touches] - z[touches]
        z_hi = hi[touches] - z[touches]
        z0_local = np.clip(0.0, z_lo, z_hi)
        r_eff = np.sqrt(np.maximum(r[touches] ** 2 - z0_local**2, 0.0))
        xs_list.append(c.points[touches, 0])
        ys_list.append(c.points[touches, 1])
        r_list.append(r_eff)

    if not xs_list:
        return np.zeros((1, 1), dtype=bool), spacing**2

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    rs = np.concatenate(r_list)
    pad = float(rs.max()) + spacing
    xmin, xmax = xs.min() - pad, xs.max() + pad
    ymin, ymax = ys.min() - pad, ys.max() + pad
    nx = max(int(np.ceil((xmax - xmin) / spacing)) + 1, 2)
    ny = max(int(np.ceil((ymax - ymin) / spacing)) + 1, 2)
    gx = xmin + spacing * np.arange(nx)
    gy = ymin + spacing * np.arange(ny)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    tree = cKDTree(np.stack([xs, ys], axis=-1))
    dist, idx = tree.query(grid_pts)
    mask = (dist <= rs[idx]).reshape(nx, ny)
    return mask, spacing**2


def slab_projected_area_numeric(density: np.ndarray, grid, s: float, t: float) -> float:
    """Numeric projected area: union, over z-slices within the slab, of the occupied pixels, times pixel area. 
    """
    z = grid.z_coords()
    in_slab = (z >= s - t / 2) & (z <= s + t / 2)
    if not np.any(in_slab):
        return 0.0
    projected = density[:, :, in_slab].any(axis=-1)
    return float(projected.sum()) * grid.pixel_area()

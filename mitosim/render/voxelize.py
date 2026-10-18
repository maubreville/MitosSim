"""Voxelisation of world-frame shapes onto a regular grid. 
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Grid:
    origin: np.ndarray          # (3,) world coords of voxel (0,0,0) centre
    spacing: float               # isotropic voxel size, micrometres
    shape: tuple[int, int, int]  # (nx, ny, nz)

    def centers(self) -> np.ndarray:
        nx, ny, nz = self.shape
        xs = self.origin[0] + self.spacing * np.arange(nx)
        ys = self.origin[1] + self.spacing * np.arange(ny)
        zs = self.origin[2] + self.spacing * np.arange(nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([X, Y, Z], axis=-1)

    def z_coords(self) -> np.ndarray:
        return self.origin[2] + self.spacing * np.arange(self.shape[2])

    def voxel_volume(self) -> float:
        return self.spacing**3

    def pixel_area(self) -> float:
        return self.spacing**2


def grid_for_bounds(lo: np.ndarray, hi: np.ndarray, spacing: float) -> Grid:
    """Smallest grid whose voxel centres cover [lo, hi] on every axis."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) + 1 for i in range(3))
    return Grid(origin=lo, spacing=spacing, shape=shape)


def voxelize(shape, grid: Grid) -> np.ndarray:
    """Boolean occupancy field for `shape` on `grid`."""
    return shape.indicator(grid.centers())


def voxel_volume_of(density: np.ndarray, grid: Grid) -> float:
    return float(density.sum()) * grid.voxel_volume()


def voxelize_chromatids(chromatids: Sequence, grid: Grid) -> np.ndarray:
    """ Boolean occupancy field for many Chromatid objects at once.
    """
    if not chromatids:
        return np.zeros(grid.shape, dtype=bool)
    all_points = np.concatenate([c.points for c in chromatids], axis=0)
    all_radii = np.concatenate([c.radii for c in chromatids], axis=0)
    tree = cKDTree(all_points)
    flat = grid.centers().reshape(-1, 3)
    # workers=-1: cKDTree parallelises a single large query cleanly across cores (measured ~6x on an 11-core machine), 
    # and this query (hundreds of thousands to millions of grid points against a few thousand centreline points) is the dominant cost of this function.
    dist, idx = tree.query(flat, workers=-1)
    inside = dist <= all_radii[idx]
    return inside.reshape(grid.shape)

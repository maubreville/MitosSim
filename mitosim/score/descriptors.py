"""Mask hollowness descriptor.

Ground-truth leakage trap: this module must only ever receive a 2D mask or
density array -- never a CellState.
"""
from __future__ import annotations

import numpy as np


def hollowness(mask: np.ndarray, pixel_area: float) -> float:
    """~0 for a solid mask, approaching 1 for a ring (empty centre). Compares
    an inner disc to an outer annulus after whitening by the mask's own
    principal axes, so an elongated (not hollow) mask isn't misread as
    hollow just because an isotropic annulus around its centroid falls
    outside a thin bar along the short axis. Returns nan for an empty mask.
    """
    # `r[mask]` needs boolean indexing -- an integer {0, 1} mask (e.g.
    # render/optics.py's uint8 convention) would instead be read as *fancy*
    # indexing, silently producing a wrong r_max. Cast once here.
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")

    ys, xs = np.nonzero(mask)
    coords = np.stack([xs, ys], axis=-1).astype(float)
    centroid = coords.mean(axis=0)
    coords -= centroid
    cov = np.cov(coords.T)
    evals, evecs = np.linalg.eigh(cov)  # ascending order

    yy, xx = np.indices(mask.shape)
    grid_coords = np.stack([xx, yy], axis=-1).astype(float) - centroid
    whitened = grid_coords @ evecs / np.sqrt(np.maximum(evals, 1e-9))
    r = np.linalg.norm(whitened, axis=-1)
    r_max = r[mask].max()
    inner_ring = r < 0.3 * r_max
    outer_ring = (r >= 0.3 * r_max) & (r <= r_max)
    inner_density = mask[inner_ring].mean() if inner_ring.any() else 0.0
    outer_density = mask[outer_ring].mean() if outer_ring.any() else 1e-9
    return float(1.0 - inner_density / max(outer_density, 1e-9))

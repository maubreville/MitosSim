"""

  This file is part of MitosSim.

  This file represents the prophase, where chromosomes are condensing, scattered through a roughly spherical nuclear volume, 
  envelope still present. There is no spindle, and thus no alignment.
"""

from __future__ import annotations

import numpy as np

from mitosim.core.types import CellState, Phase
from mitosim.geometry.chromosome import chromosome_shape, enclosing_radius, place_chromatid, rotation_aligning_x_to
from mitosim.geometry.karyotype import sample_specs
from mitosim.phases.base import PhaseParams


def configure(rng: np.random.Generator, params: PhaseParams) -> CellState:
    specs = sample_specs(rng)
    chromatids = []
    for chromosome_id, spec in enumerate(specs):
        control, radius = chromosome_shape(rng, spec)

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radial_frac = rng.uniform(0.0, 1.0) ** (1.0 / 3.0)  # uniform within the ball
        max_r = max(params.nuclear_radius - spec.length / 2, 0.5)
        position = direction * radial_frac * max_r

        axis_dir = rng.normal(size=3)
        axis_dir /= np.linalg.norm(axis_dir)
        rotation = rotation_aligning_x_to(axis_dir, roll=rng.uniform(0, 2 * np.pi))

        gap = spec.radius
        chromatids.append(place_chromatid(control, radius, chromosome_id, 0, position, rotation, gap / 2, params.n_samples))
        chromatids.append(place_chromatid(control, radius, chromosome_id, 1, position, rotation, -gap / 2, params.n_samples))

    cell_radius = max(params.nuclear_radius + 1.0, enclosing_radius(chromatids))
    return CellState(
        chromatids=chromatids,
        poles=np.zeros((0, 3)),
        phase=Phase.PROPHASE,
        cell_radius=cell_radius,
        provenance={},
    )

"""

  This file is part of MitosSim.

  This file represents a mitosis in metaphase (i.e., chromosomes aligned at the equatorial plate, a disc of finite thickness)
  There can be a hollow or sparse centre where the plate is thinnest. 

"""
from __future__ import annotations

import numpy as np

from mitosim.core.types import CellState, Phase
from mitosim.geometry.chromosome import chromosome_shape, enclosing_radius, place_chromatid, rotation_aligning_x_to
from mitosim.geometry.karyotype import sample_specs
from mitosim.geometry.spindle import bipolar_spindle
from mitosim.phases.base import PhaseParams


def configure(rng: np.random.Generator, params: PhaseParams) -> CellState:
    specs = sample_specs(rng)
    chromatids = []
    for chromosome_id, spec in enumerate(specs):
        control, radius = chromosome_shape(rng, spec)

        theta = rng.uniform(0.0, 2 * np.pi) # uniform distribution, theta can have any value
        r = rng.uniform(params.hollow_fraction * params.plate_radius, params.plate_radius)
        z = rng.uniform(-params.plate_thickness / 2, params.plate_thickness / 2)
        position = np.array([r * np.cos(theta), r * np.sin(theta), z])

        axis_dir = np.array([np.cos(theta), np.sin(theta), 0.0])  # radial, arms projecting outward
        rotation = rotation_aligning_x_to(axis_dir, roll=rng.uniform(0, 2 * np.pi))

        gap = spec.radius
        chromatids.append(place_chromatid(control, radius, chromosome_id, 0, position, rotation, gap / 2, params.n_samples))
        chromatids.append(place_chromatid(control, radius, chromosome_id, 1, position, rotation, -gap / 2, params.n_samples))

    poles = bipolar_spindle(params.spindle_half_length).poles
    cell_radius = max(params.plate_radius, params.spindle_half_length) + 1.0
    cell_radius = max(cell_radius, enclosing_radius(chromatids))
    return CellState(
        chromatids=chromatids,
        poles=poles,
        phase=Phase.METAPHASE,
        cell_radius=cell_radius,
        provenance={},
    )

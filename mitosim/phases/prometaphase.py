"""

  This file is part of MitosSim.

  This file represents a mitosis in prometaphase, with envelope broken down, chromosomes at varying stages of congression.

"""
from __future__ import annotations

import numpy as np

from mitosim.core.types import CellState, Phase
from mitosim.geometry.chromosome import chromosome_shape, enclosing_radius, place_chromatid, rotation_aligning_x_to
from mitosim.geometry.karyotype import sample_specs
from mitosim.geometry.spindle import bipolar_spindle
from mitosim.phases.base import PhaseParams

#  `PhaseParams.progress` biases how many chromosomes have reached the plate it warps the per-chromosome,
#  `congression` draw's exponent rather than replacing it, so:
#   - progress=0 (early) skews most chromosomes toward peripheral and 
#   - progress=1 (late) skews most toward the plate, but both ends keep some chromosomes at every intermediate stage. 
#   - progress=0.5 (default) yields a flat `rng.uniform(0, 1)` draw.

# A still-peripheral chromosome (congression near 0) needs its own independent random orientation, continuous with 
# prophase.py's fully random axis_dir, it must NOT reuse the same random direction that also places it in space, or every peripheral
# chromosome ends up pointing radially outward from the cell centre (long axis aligned with its own position vector) purely as an artifact of
# sharing one random draw for two unrelated roles, before any real congression has pulled it there.

def configure(rng: np.random.Generator, params: PhaseParams) -> CellState:
    specs = sample_specs(rng)
    chromatids = []
    # x**congression_exponent (x ~ Uniform(0,1)) skews the draw toward 0
    # (exponent > 1) or toward 1 (exponent < 1) while keeping full [0,1]
    # support -- progress=0.5 gives exponent=1 for any base, the original
    # flat draw. Base 10 gives exponent=10 at progress=0 (mean congression
    # 0.09), pulling early prometaphase convincingly toward "still a blob"
    # rather than visibly plate-ward.
    congression_exponent = 10.0 ** (1.0 - 2.0 * params.progress)
    for chromosome_id, spec in enumerate(specs):
        control, radius = chromosome_shape(rng, spec)
        congression = rng.uniform(0.0, 1.0) ** congression_exponent  # 0 = still peripheral, 1 = at the plate

        random_dir = rng.normal(size=3)
        random_dir /= np.linalg.norm(random_dir)
        radial_frac = rng.uniform(0.0, 1.0) ** (1.0 / 3.0)
        start_pos = random_dir * radial_frac * params.nuclear_radius

        random_orientation = rng.normal(size=3)
        random_orientation /= np.linalg.norm(random_orientation)

        theta = rng.uniform(0.0, 2 * np.pi)
        r_plate = rng.uniform(params.hollow_fraction * params.plate_radius, params.plate_radius)
        plate_pos = np.array(
            [r_plate * np.cos(theta), r_plate * np.sin(theta), rng.uniform(-params.plate_thickness / 2, params.plate_thickness / 2)]
        )
        position = (1 - congression) * start_pos + congression * plate_pos

        radial_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        axis_dir = (1 - congression) * random_orientation + congression * radial_dir
        axis_dir /= np.linalg.norm(axis_dir)
        rotation = rotation_aligning_x_to(axis_dir, roll=rng.uniform(0, 2 * np.pi))

        gap = spec.radius
        chromatids.append(place_chromatid(control, radius, chromosome_id, 0, position, rotation, gap / 2, params.n_samples))
        chromatids.append(place_chromatid(control, radius, chromosome_id, 1, position, rotation, -gap / 2, params.n_samples))

    poles = bipolar_spindle(params.spindle_half_length).poles
    cell_radius = max(params.nuclear_radius, params.spindle_half_length) + 1.0
    cell_radius = max(cell_radius, enclosing_radius(chromatids))
    return CellState(
        chromatids=chromatids,
        poles=poles,
        phase=Phase.PROMETAPHASE,
        cell_radius=cell_radius,
        provenance={},
    )

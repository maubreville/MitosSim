"""

  This file is part of MitosSim.

  This file represents the telophase, where two masses are decondensing toward daughter nuclei. The chromatids puff up 
  (radius grows again), they lose their straight-bar silhouette, and their oriented, rod-like arrangement as `PhaseParams.progress` goes 
  from 0 (just past anaphase) to 1 (fully decondensed, near-spherical mass). The two masses also keep migrating apart as telophase proceeds 
  (continued poleward movement into early cytokinesis) rather than freezing at the late-anaphase position, so that late telophase (masses 
  far apart) is actually reachable. Other parts of cytokinesis happening in late phases of telophase (cleavage furrow and midbody) are not modelled.
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
    decond = params.progress

    # The bend degree range goes from near 180 degrees (anaphase) to lower angles (100-160) later.
    bend_deg_range = (100.0 + 68.0 * (1.0 - decond), 160.0 + 18.0 * (1.0 - decond))

    for chromosome_id, spec in enumerate(specs):
        control, radius = chromosome_shape(rng, spec, bend_deg_range=bend_deg_range)
        radius = radius * (1.0 + 1.5 * decond)

        for sister, pole_sign in ((0, 1.0), (1, -1.0)):
            spread = 0.4 + 1.2 * decond
            xy = rng.normal(scale=spread, size=2)
            # (1.0 + decond) lets masses roughly double their distance from the equator by full 
            #  decondensation, instead of being capped at the late-anaphase position (0.75-0.95x) regardless of how
            # far telophase has progressed.
            z = pole_sign * params.spindle_half_length * rng.uniform(0.75, 0.95) * (1.0 + decond)
            position = np.array([xy[0], xy[1], z])

            oriented_dir = np.array([0.0, 0.0, -pole_sign])
            random_dir = rng.normal(size=3)
            random_dir /= np.linalg.norm(random_dir)
            axis_dir = (1 - decond) * oriented_dir + decond * random_dir
            axis_dir /= np.linalg.norm(axis_dir)
            rotation = rotation_aligning_x_to(axis_dir, roll=rng.uniform(0, 2 * np.pi))

            chromatids.append(
                place_chromatid(control, radius, chromosome_id, sister, position, rotation, 0.0, params.n_samples)
            )

    poles = bipolar_spindle(params.spindle_half_length).poles
    cell_radius = max(params.spindle_half_length * (1.0 + decond) + 3.0, enclosing_radius(chromatids))
    return CellState(
        chromatids=chromatids,
        poles=poles,
        phase=Phase.TELOPHASE,
        cell_radius=cell_radius,
        provenance={},
    )

"""

  This file is part of MitosSim.

  Here, we simulate anaphase, where sister chromatids separate into the direction of the poles, with centromeres 
  leading and arms trailing (concave v bent). 

  `PhaseParams.progress` interpolates A (chromatids moving, poles static) toward full separation;
  anaphase B (poles moving apart) is not modelled separately - both read the same spindle_half_length.
  The representative value for anaphase is 0.6, not the PhaseParams default of 0.5 (see
  experiments/data/generate_synthetic.py's ANAPHASE_PARAMS) -- pass it explicitly when configuring this phase.


   Chromatid shape is also driven by `progress`, so the phase boundaries on both sides line up:
    - at progress=0 (onset, chromatids just leaving the plate) it's the same chevron `chromosome_shape()` draws
      everywhere by default, continuous with the preceding metaphase/prometaphase shape; 
    - at progress=1 ("the last step of anaphase") it's pulled to the same near-straight bar silhouette and `telophase.py` 
      starts from at its own progress=0.
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
    progress = params.progress
    bend_deg_range = (100.0 + 68.0 * progress, 160.0 + 18.0 * progress)

    for chromosome_id, spec in enumerate(specs):
        control, radius = chromosome_shape(rng, spec, bend_deg_range=bend_deg_range)
        xy_scatter = rng.normal(scale=0.6, size=2)

        for sister, pole_sign in ((0, 1.0), (1, -1.0)):
            z = pole_sign * progress * params.spindle_half_length * rng.uniform(0.7, 1.0)
            position = np.array([xy_scatter[0], xy_scatter[1], z])
            # centromere (local origin) leads toward the pole; arms trail back toward the equator.
            axis_dir = np.array([0.0, 0.0, -pole_sign])
            rotation = rotation_aligning_x_to(axis_dir, roll=rng.uniform(0, 2 * np.pi))
            chromatids.append(
                place_chromatid(control, radius, chromosome_id, sister, position, rotation, 0.0, params.n_samples)
            )

    poles = bipolar_spindle(params.spindle_half_length).poles
    cell_radius = max(params.spindle_half_length + 2.0, enclosing_radius(chromatids))
    return CellState(
        chromatids=chromatids,
        poles=poles,
        phase=Phase.ANAPHASE,
        cell_radius=cell_radius,
        provenance={},
    )

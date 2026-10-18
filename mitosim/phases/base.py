"""

  This file is part of MitosSim.

  This generates the prototype for phase parameters. 

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseParams:
    # Pole-to-pole distance is 2 * spindle_half_length = 12.3um, matching the metaphase spindle length measured by 
    # Fu et al. ("TPX2 phosphorylation maintains metaphase spindle length by regulating microtubule flux").
    spindle_half_length: float = 6.15
    nuclear_radius: float = 6.0
    # Tuned so an edge-on section renders as a clean compact bar rather than a loose, broken-up mess: a larger/thicker 
    # plate (5.0/2.0) often splits into 2+ components edge-on. This does make the ring orientation's hollow center less
    # pronounced on average, which `RING_PARAMS` in experiments/data/generate_synthetic.py compensates for with a more 
    # pronounced `hollow_fraction`.
    plate_radius: float = 4.0
    plate_thickness: float = 0.5
    hollow_fraction: float = 0.25    # inner plate radius fraction left sparse/empty

    # 0 = early/onset snapshot of whichever phase generator reads this, 1 = late/fully-progressed snapshot
    # (meaning depends on the phase)
    progress: float = 0.5
    n_samples: int = 60              # centreline samples per chromatid, 60 should be fine

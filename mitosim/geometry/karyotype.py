"""

This file is part of MitosSim.

This file defines the human karyotype that encompasses 23 chromosomes (1-22 and then XX/XY). We
define/model these all individually, according to sources from the literature.

- Relative length were taken from Van Dyke et al., 1986. (see Table below)
- Van Dyke's relative length is a percentage of total haploid autosomal length, not an absolute size, 
  so it needs one absolute anchor to convert to micrometres. We scale linearly. For this, we anchor chromosome 1 
  at 8um (the midpoint of Booth et al. 2016's directly-measured ~7-9um for chromosome 1) and then scale linearly.
- The centromere index (the percent of a chromosome's length that is the short (p) arm) was taken from the same Van Dyke paper.
- The chromatid radius is drawn from a flat range independent of chromosome identity, according to findings by Booth et al. 2016.

`sample_specs` then builds one real diploid human set per cell, including both copies of each autosome 1-22, plus a sex-chromosome 
pair representing 46,XX or 46,XY (picked uniformly at random per cell)

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Van Dyke et al., 1986, Table 1, mid-metaphase columns:
# This is a dict of 4-tuples:
# name -> (relative length mean %, relative length SD %, centromere index mean, centromere index SD).
# Relative length is percent of total haploid autosomal length.
_VAN_DYKE_MID_METAPHASE: dict[str, tuple[float, float, float, float]] = {
    "1": (9.00, 0.42, 51.20, 1.55),
    "2": (8.37, 0.17, 39.42, 0.45),
    "3": (6.96, 0.20, 46.36, 0.95),
    "4": (6.44, 0.19, 27.44, 1.38),
    "5": (6.16, 0.14, 26.59, 0.95),
    "6": (5.95, 0.18, 35.69, 3.49),
    "7": (5.38, 0.18, 38.61, 0.61),
    "8": (4.77, 0.16, 34.16, 0.91),
    "9": (4.76, 0.14, 33.81, 1.73),
    "10": (4.69, 0.25, 33.57, 1.06),
    "11": (4.63, 0.07, 37.36, 2.04),
    "12": (4.70, 0.10, 29.05, 0.85),
    "13": (3.41, 0.04, 10.74, 0.91),
    "14": (3.32, 0.07, 11.61, 0.32),
    "15": (3.42, 0.20, 12.25, 1.00),
    "16": (3.27, 0.11, 44.95, 0.93),
    "17": (3.21, 0.09, 32.52, 2.05),
    "18": (2.75, 0.11, 28.26, 0.86),
    "19": (2.63, 0.07, 45.55, 2.18),
    "20": (2.33, 0.09, 44.41, 1.77),
    "21": (1.41, 0.08, 18.33, 3.20),
    "22": (1.90, 0.06, 19.87, 1.85),
    "X": (5.08, 0.11, 40.13, 2.64),
    "Y": (2.02, 0.30, 27.84, 2.07),
}

# Scale of microns per 1% relative length -> anchors chromosome 1 (9.00%) at 8um to then liearly scale.
_LENGTH_SCALE_UM_PER_PCT = 8.0 / _VAN_DYKE_MID_METAPHASE["1"][0]

# Clipping of entromere index to (0, 100), so an unlucky tail draw never produces a zero-length arm in chromosome.py's arm-length split.
_CENTROMERE_INDEX_CLIP = (2.0, 98.0)

# Booth et al. 2016: diameter ~1.0-1.3um, flat across chromosomes -> radius ~0.5-0.65um.
_RADIUS_RANGE = (0.5, 0.6)


@dataclass(frozen=True)
class ChromosomeIdentity:
    name: str                     # "1".."22", "X", "Y"
    length_mean: float             # in microns
    length_sd: float                # in microns
    centromere_index_mean: float   # % of length that is the p (short) arm
    centromere_index_sd: float


def _identity(name: str) -> ChromosomeIdentity:
    rel_mean, rel_sd, ci_mean, ci_sd = _VAN_DYKE_MID_METAPHASE[name]
    return ChromosomeIdentity(
        name=name,
        length_mean=rel_mean * _LENGTH_SCALE_UM_PER_PCT,
        length_sd=rel_sd * _LENGTH_SCALE_UM_PER_PCT,
        centromere_index_mean=ci_mean,
        centromere_index_sd=ci_sd,
    )


_AUTOSOMES = tuple(_identity(str(i)) for i in range(1, 23))
_X = _identity("X")
_Y = _identity("Y")


@dataclass(frozen=True)
class ChromosomeSpec:
    name: str              # chromosome identity, e.g. "1", "21", "X"
    length: float           # condensed length, micrometres
    radius: float            # chromatid radius, micrometres
    centromere_index: float  # % of length that is the p (short) arm


def _sample_one(rng: np.random.Generator, identity: ChromosomeIdentity) -> ChromosomeSpec:
    length = max(0.3, float(rng.normal(identity.length_mean, identity.length_sd)))
    centromere_index = float(
        np.clip(rng.normal(identity.centromere_index_mean, identity.centromere_index_sd), *_CENTROMERE_INDEX_CLIP)
    )
    radius = float(rng.uniform(*_RADIUS_RANGE))
    return ChromosomeSpec(name=identity.name, length=length, radius=radius, centromere_index=centromere_index)


def sample_specs(rng: np.random.Generator) -> list[ChromosomeSpec]:
    specs = [_sample_one(rng, identity) for identity in _AUTOSOMES for _ in range(2)]
    is_xy = rng.random() < 0.5
    sex_identities = (_X, _Y) if is_xy else (_X, _X)
    specs.extend(_sample_one(rng, identity) for identity in sex_identities)
    return specs

"""

  This file is part of MitosSim.

  Sectioning experiments sample a spindle orientation and must rotate the geometry into that orientation before slicing. 

"""
from __future__ import annotations

import numpy as np

from mitosim.core.types import CellState
from mitosim.geometry.chromosome import Chromatid


def rotation_aligning_z_to(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ [0, 0, 1] == direction (a unit vector).

    Transforms a cell built in its original coordinate system (spindle along local +z) to world coordinates.

    """
    # Normalise so callers may pass any non-zero vector.
    direction = direction / np.linalg.norm(direction)

    # We need a vector that is NOT parallel to `direction` to start a cross product. Use world-x unless the direction is nearly 
    # along x (the cross product would then be close to zero and numerically unstable), in which case use world-y.
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    # Build an orthonormal, right-handed frame (e1, e2, direction):
    e1 = np.cross(helper, direction)   # perpendicular to direction
    e1 /= np.linalg.norm(e1)           # unit length
    e2 = np.cross(direction, e1)       # perpendicular to both, already unit length

    # Columns are where the local x, y, z axes land in world coordinates.The third column is `direction`, so 
    # R @ [0, 0, 1] == direction. e1 × e2 == direction, so det(R) == +1 (a proper rotation, no mirroring).
    return np.stack([e1, e2, direction], axis=1)


def rotate_cell_state(state: CellState, rotation: np.ndarray) -> CellState:
    """Return a new CellState with all geometry rotated about the world origin.

    - rotation is a 3x3 matrix acting on column vectors (v' = R @ v).

    """
    # Points are stored as rows, overall shape is (n, 3). 
    # For row vectors, applying R to every point is `points @ R.T`.
    #
    # Chromatid is frozen, so we build new instances rather than mutate.
    new_chromatids = [
        Chromatid(
            points=c.points @ rotation.T,
            radii=c.radii,                     # tube radii are rotation-invariant
            chromosome_id=c.chromosome_id,     # identity metadata is unchanged
            sister=c.sister,
            centromere=c.centromere @ rotation.T,  # a point, so it rotates too
        )
        for c in state.chromatids
    ]

    # Spindle poles are points as well, and we need to rotate them alike. 
    new_poles = state.poles @ rotation.T if state.poles.size else state.poles

    return CellState(
        chromatids=new_chromatids,
        poles=new_poles,
        phase=state.phase,
        cell_radius=state.cell_radius,
        provenance=state.provenance,
    )

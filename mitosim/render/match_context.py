"""Match a synthetic chromatin mask to the context image in
`weights/context_ellipse_db.csv` (built by `build_context_ellipse_db.py`)
whose own real mitotic figure was the closest size and orientation --
rather than a context image picked at random, which can mismatch badly
(e.g. a small round metaphase mask conditioned against a context photo
whose own real figure was a large elongated anaphase bridge).

Orientation match is widened for free, two ways, both exact and lossless
(`PIL.Image.transpose`, no resampling):

- a context image can be **rotated** by 0/90/180/270 degrees
  (`rotate_image_90`), which shifts its ellipse's angle by the same amount,
  mod 180 (an ellipse's orientation is 180-degree periodic, it looks the
  same rotated by half a turn);
- a context image can be **mirrored** (`mirror_image`, a left-right flip),
  which negates its ellipse's angle, mod 180.

Composed (mirror applied before rotation -- `_TRANSFORMS` enumerates all
eight combinations in that fixed order, so a stored `ContextMatch` can be
replayed exactly), these eight transforms place a candidate's angle
anywhere a quarter-circle's worth of its original value can reach: any
angle in [0, 90) degrees has a same-shaped partner in [90, 180) degrees
reachable by mirroring alone, so the database only needs looking at one
quarter-turn's worth of true orientations to cover the whole range.
Semi-axis extents never change under any of the eight. So for each
candidate, only its *transform* is a free choice; matching then reduces
to: over the eight transforms, which one gets closest to the target's own
orientation, combined with a fixed (transform-invariant) size distance.

Two selection strategies build on that scoring:

- `match_context_images`/`pick_matched_context_image` -- continuous
  nearest-neighbour: rank every candidate by combined size+orientation
  distance and pick among the closest few. Good for the CLI's own
  diagnostic ranking (below), but narrow as a *generation* strategy: the
  "closest few" to any one target mask is nearly the same handful of
  filenames every time, so a large generated dataset ends up reusing a
  small slice of the corpus's real background variety.
- `category_for`/`pick_bucketed_context_image` -- coarse category,
  uniform random: bucket the whole corpus into a fixed 3 (angle) x 5
  (size) grid (every cell holds at least ~150 of the 2638 real masks
  after mirror-folding),
  find the target mask's own bucket, and pick *uniformly at random* among
  every real mask in it. This is the default context-selection strategy
  `gan_render_synthetic.py` actually uses for `context="matched"`: every
  same-category candidate is equally likely, not just the handful nearest
  by distance, which is what "limited variance in selected backgrounds"
  (the complaint this entire module exists to fix) actually required --
  being in the right category matters, being the *closest* match within
  it doesn't.

Run directly: `python -m mitosim.render.match_context --help`.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mitosim.render.inpaint import DEFAULT_WEIGHTS_DIR, EllipseFit, fit_ellipse, load_context_image

#: PIL transpose ops for the four exact, lossless 90-degree rotations --
#: keyed by how many degrees they add to a mask's fitted angle (mod 180).
_ROTATION_OPS: dict[int, Image.Transpose | None] = {
    0: None,
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}

#: Every (mirror, rotation_deg) combination `match_context_images` searches
#: over, mirror applied first -- see `transform_image`. Order matches
#: `_predicted_angle`'s own math, so a stored `ContextMatch` always
#: reproduces the angle it was scored against.
_TRANSFORMS: tuple[tuple[bool, int], ...] = tuple((mirror, rotation_deg) for mirror in (False, True) for rotation_deg in (0, 90, 180, 270))


def rotate_image_90(image: Image.Image, rotation_deg: int) -> Image.Image:
    """Rotate `image` by an exact multiple of 90 degrees (0/90/180/270),
    via `Image.transpose` rather than `Image.rotate` -- pixel-exact, no
    interpolation, so a real photo's texture is never resampled/blurred
    just to adjust which way it's "facing"."""
    if rotation_deg % 90 != 0:
        raise ValueError(f"rotation_deg must be a multiple of 90, got {rotation_deg}")
    op = _ROTATION_OPS[rotation_deg % 360]
    return image if op is None else image.transpose(op)


def mirror_image(image: Image.Image) -> Image.Image:
    """Flip `image` left-right -- exact, lossless, no interpolation.
    Negates a mask's fitted angle, mod 180 (a top-bottom flip would do the
    same thing to the angle; left-right is just the convention used
    throughout this module)."""
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def transform_image(image: Image.Image, rotation_deg: int, mirror: bool) -> Image.Image:
    """Apply `mirror_image` (if `mirror`) then `rotate_image_90` -- the
    fixed order `_predicted_angle` assumes and every `ContextMatch` was
    scored against."""
    if mirror:
        image = mirror_image(image)
    return rotate_image_90(image, rotation_deg)


def _predicted_angle(row_angle_deg: float, rotation_deg: int, mirror: bool) -> float:
    """What a candidate's fitted angle becomes after `transform_image` with
    these parameters -- mirror negates it, rotation then adds, matching
    that function's fixed mirror-then-rotate order."""
    angle = -row_angle_deg if mirror else row_angle_deg
    return angle + rotation_deg


def load_ellipse_db(weights_dir: Path | str | None = None, path: Path | str | None = None) -> list[dict]:
    """Read `build_context_ellipse_db.py`'s CSV output into a list of dict
    rows, with the numeric columns already parsed as float."""
    path = Path(path) if path is not None else (Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR) / "context_ellipse_db.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no ellipse database at {path} -- run `python -m mitosim.render.build_context_ellipse_db` first")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "filename": row["filename"],
                    "centroid_x": float(row["centroid_x"]),
                    "centroid_y": float(row["centroid_y"]),
                    "extent_major_px": float(row["extent_major_px"]),
                    "extent_minor_px": float(row["extent_minor_px"]),
                    "angle_deg": float(row["angle_deg"]),
                }
            )
    return rows


@dataclass(frozen=True)
class ContextMatch:
    filename: str
    rotation_deg: int  # apply via transform_image to both the context image and its own mask
    mirror: bool  # likewise -- transform_image applies this first, then rotation_deg
    size_distance: float  # relative L2 distance in (extent_major, extent_minor) space, >= 0
    orientation_distance_deg: float  # angular distance after the transform, in [0, 90]
    score: float  # size_weight*size_distance + orientation_weight*(orientation_distance_deg/90); lower is better


def _angle_distance_mod180(a_deg: float, b_deg: float) -> float:
    """Angular distance between two orientations that are each only
    defined mod 180 (an ellipse looks the same rotated by half a turn),
    folded into [0, 90]."""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def _best_transform_for_row(target_angle_deg: float, row_angle_deg: float) -> tuple[int, bool, float]:
    """Which of the eight (rotation, mirror) transforms (`_TRANSFORMS`) of
    this candidate gets its angle closest to the target's, and the
    resulting (already-minimal) angular distance."""
    best_rotation, best_mirror, best_distance = 0, False, float("inf")
    for mirror, rotation_deg in _TRANSFORMS:
        distance = _angle_distance_mod180(target_angle_deg, _predicted_angle(row_angle_deg, rotation_deg, mirror))
        if distance < best_distance:
            best_rotation, best_mirror, best_distance = rotation_deg, mirror, distance
    return best_rotation, best_mirror, best_distance


def match_context_images(
    mask_or_fit: np.ndarray | EllipseFit,
    db: list[dict] | None = None,
    weights_dir: Path | str | None = None,
    top_k: int = 5,
    size_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> list[ContextMatch]:
    """Rank every row of the ellipse database by combined size+orientation
    distance to `mask_or_fit` (a synthetic chromatin mask, or an
    already-computed `fit_ellipse(mask)`), and return the `top_k` closest.

    Both distances are normalised to comparable scales before weighting:
    `size_distance` is a relative (unitless) L2 difference in extents, and
    `orientation_distance_deg` is divided by 90 (its own max possible
    value) -- so `size_weight=orientation_weight=1.0` (the default) treats
    a candidate that's 100% off in size the same as one that's maximally
    (90 degrees) misoriented. Raises ValueError if `mask_or_fit` is a mask
    with fewer than 2 foreground pixels (see `fit_ellipse`).
    """
    if isinstance(mask_or_fit, EllipseFit):
        target = mask_or_fit
    else:
        target = fit_ellipse(mask_or_fit)
        if target is None:
            raise ValueError("mask has fewer than 2 foreground pixels -- no ellipse to match against")

    if db is None:
        db = load_ellipse_db(weights_dir)

    target_angle_deg = np.degrees(target.angle_rad)
    matches = []
    for row in db:
        size_distance = float(
            np.hypot(
                (target.extent_major - row["extent_major_px"]) / max(target.extent_major, 1e-6),
                (target.extent_minor - row["extent_minor_px"]) / max(target.extent_minor, 1e-6),
            )
        )
        rotation_deg, mirror, orientation_distance_deg = _best_transform_for_row(target_angle_deg, row["angle_deg"])
        score = size_weight * size_distance + orientation_weight * (orientation_distance_deg / 90.0)
        matches.append(
            ContextMatch(
                filename=row["filename"],
                rotation_deg=rotation_deg,
                mirror=mirror,
                size_distance=size_distance,
                orientation_distance_deg=orientation_distance_deg,
                score=score,
            )
        )

    matches.sort(key=lambda m: m.score)
    return matches[:top_k]


def load_matched_context(match: ContextMatch, weights_dir: Path | str | None = None) -> Image.Image:
    """Load `match.filename`'s context image and apply its chosen
    transform (mirror, then rotation -- see `transform_image`) -- the
    actual (image, mask-shape) pairing a caller would feed into the GAN
    context encoder."""
    return transform_image(load_context_image(match.filename, weights_dir), match.rotation_deg, match.mirror)


def pick_matched_context_image(
    mask: np.ndarray,
    rng: np.random.Generator,
    db: list[dict] | None = None,
    weights_dir: Path | str | None = None,
    top_k: int = 5,
    size_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> tuple[ContextMatch, Image.Image]:
    """Rank context images by size+orientation similarity to `mask`
    (`match_context_images`), then pick uniformly at random among the
    `top_k` best via `rng` (a numpy Generator, not Python's own `random` --
    see `pick_context_image`'s reasoning for why: composes with the rest of
    MitoSim's seed-derived reproducibility). Returns the winning
    `ContextMatch` alongside its already-transformed (mirrored/rotated --
    see `load_matched_context`) image, ready to feed into the GAN context
    encoder as the context image.

    `db` lets a caller reuse an already-`load_ellipse_db`-loaded database
    across many calls (the batch CLI does this, rather than re-reading and
    re-parsing a 2638-row CSV once per cell); when omitted, one is loaded
    fresh from `weights_dir`.

    Raises the same errors `match_context_images` does: `FileNotFoundError`
    if `weights/context_ellipse_db.csv` doesn't exist yet, `ValueError` if
    `mask` has fewer than 2 foreground pixels (too degenerate to fit an
    ellipse to at all -- callers that want to fall back to
    `pick_context_image`'s random selection for that rare case should catch
    it themselves, since silently downgrading match quality isn't this
    function's call to make).
    """
    matches = match_context_images(mask, db=db, weights_dir=weights_dir, top_k=top_k, size_weight=size_weight, orientation_weight=orientation_weight)
    if not matches:
        raise FileNotFoundError("context_ellipse_db.csv has no rows -- see README's inpainting section")
    chosen = matches[int(rng.integers(0, len(matches)))]
    return chosen, load_matched_context(chosen, weights_dir)


#: Angle/size category grid used by `pick_bucketed_context_image`. Angle
#: is folded twice --
#: first to [0, 180) (an ellipse's orientation is 180-degree periodic),
#: then to its canonical [0, 90] representative via `fold_angle_to_90`
#: (mirroring reaches the other half for free, see module docstring).
#: Size bins are quintile edges of `extent_major_px` across all 2638 real
#: AMi-Br training masks (`weights/context_ellipse_db.csv`), computed once
#: and hardcoded here rather than re-derived at runtime, so category
#: boundaries stay stable even if that CSV is regenerated with minor
#: numerical differences. Re-derive (and update this constant) after any
#: deliberate corpus change, such as a manual mask-correction pass --
#: these values are current as of the pass completed 2026-10-02.
ANGLE_BIN_EDGES: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)
SIZE_BIN_EDGES: tuple[float, ...] = (11.2, 18.1, 21.2, 24.3, 29.5, 230.3)
SIZE_BIN_LABELS: tuple[str, ...] = ("very small", "small", "medium", "large", "very large")


def fold_angle_to_90(angle_deg: float) -> float:
    """theta and 180-theta are mirror-equivalent -- fold any angle down to
    its canonical [0, 90] representative (first mod 180, since orientation
    is already 180-degree periodic on its own)."""
    folded = angle_deg % 180.0
    return min(folded, 180.0 - folded)


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    """Which of `len(edges) - 1` bins `value` falls into, clamping into
    the last bin if `value` meets or exceeds the final edge (so an
    unusually large synthetic mask still gets a category instead of
    falling through to nothing)."""
    for i in range(len(edges) - 2):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def category_for(fit: EllipseFit) -> tuple[int, int]:
    """(angle_bin, size_bin) for an already-computed ellipse fit -- see
    `ANGLE_BIN_EDGES`/`SIZE_BIN_EDGES`."""
    return _bin_index(fold_angle_to_90(np.degrees(fit.angle_rad)), ANGLE_BIN_EDGES), _bin_index(fit.extent_major, SIZE_BIN_EDGES)


def _category_for_row(row: dict) -> tuple[int, int]:
    """Same as `category_for`, for a plain `load_ellipse_db` row dict
    rather than an `EllipseFit` (the database stores the un-folded angle
    exactly as fitted, same convention `category_for` folds on demand)."""
    return _bin_index(fold_angle_to_90(row["angle_deg"]), ANGLE_BIN_EDGES), _bin_index(row["extent_major_px"], SIZE_BIN_EDGES)


def pick_bucketed_context_image(
    mask_or_fit: np.ndarray | EllipseFit,
    rng: np.random.Generator,
    db: list[dict] | None = None,
    weights_dir: Path | str | None = None,
) -> tuple[ContextMatch, Image.Image]:
    """Pick uniformly at random among every context image sharing the
    target's own (angle, size) category (`category_for`) -- not the
    closest few by continuous distance (`pick_matched_context_image`'s
    approach), so every real mask in the category is equally likely to be
    picked. This is the default context-selection strategy `context=
    "matched"` actually uses (`gan_render_synthetic.py`): maximising
    background variety mattered more than picking the single closest
    match, since every one of the 15 categories holds at least ~150 of
    the 2638 real masks -- comfortably enough to never feel like the
    same handful of backgrounds on repeat.

    The picked candidate still gets `_best_transform_for_row`'s exact
    mirror+rotation alignment against the target's own (un-folded) angle,
    same as `pick_matched_context_image` -- category membership only
    guarantees "same ballpark" (within a 30-degree wedge and a size
    quintile), the transform search is what gets it precisely aligned.

    `db` lets a caller reuse an already-`load_ellipse_db`-loaded database
    across many calls, same as `pick_matched_context_image`.

    Raises `ValueError` if `mask_or_fit` is a mask with fewer than 2
    foreground pixels (see `fit_ellipse`) -- callers that want to fall
    back to `pick_context_image`'s plain random selection for that rare
    case should catch it themselves. Raises `FileNotFoundError` if the
    database is missing, or (should not happen given the corpus's actual
    distribution, but guarded anyway) if the target's category happens to
    be empty.
    """
    if isinstance(mask_or_fit, EllipseFit):
        target = mask_or_fit
    else:
        target = fit_ellipse(mask_or_fit)
        if target is None:
            raise ValueError("mask has fewer than 2 foreground pixels -- no ellipse to match against")

    if db is None:
        db = load_ellipse_db(weights_dir)
    if not db:
        raise FileNotFoundError("context_ellipse_db.csv has no rows -- see README's inpainting section")

    target_bucket = category_for(target)
    candidates = [row for row in db if _category_for_row(row) == target_bucket]
    if not candidates:
        raise FileNotFoundError(f"no context images in angle/size category {target_bucket} -- see README's inpainting section")

    row = candidates[int(rng.integers(0, len(candidates)))]
    target_angle_deg = np.degrees(target.angle_rad)
    size_distance = float(
        np.hypot(
            (target.extent_major - row["extent_major_px"]) / max(target.extent_major, 1e-6),
            (target.extent_minor - row["extent_minor_px"]) / max(target.extent_minor, 1e-6),
        )
    )
    rotation_deg, mirror, orientation_distance_deg = _best_transform_for_row(target_angle_deg, row["angle_deg"])
    match = ContextMatch(
        filename=row["filename"],
        rotation_deg=rotation_deg,
        mirror=mirror,
        size_distance=size_distance,
        orientation_distance_deg=orientation_distance_deg,
        score=size_distance + orientation_distance_deg / 90.0,
    )
    return match, load_matched_context(match, weights_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mask_path", help="path to a binary mask PNG to match against (e.g. one written by generate_synthetic.py)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--size-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--weights-dir", default=None, help="default: MitoSim's weights/ (or $MITOSIM_WEIGHTS_DIR)")
    args = parser.parse_args()

    mask = np.array(Image.open(args.mask_path).convert("L")) > 0
    target = fit_ellipse(mask)
    if target is None:
        raise SystemExit(f"{args.mask_path} has fewer than 2 foreground pixels -- nothing to match")
    print(f"target: major={target.extent_major:.1f}px minor={target.extent_minor:.1f}px angle={np.degrees(target.angle_rad):.1f}deg")

    matches = match_context_images(
        target, weights_dir=args.weights_dir, top_k=args.top_k, size_weight=args.size_weight, orientation_weight=args.orientation_weight
    )
    print(f"\ntop {len(matches)} matches:")
    for m in matches:
        print(
            f"  {m.filename:<24} rotate={m.rotation_deg:>3}deg  mirror={str(m.mirror):<5}  "
            f"size_dist={m.size_distance:.3f}  orient_dist={m.orientation_distance_deg:.1f}deg  score={m.score:.3f}"
        )


if __name__ == "__main__":
    main()

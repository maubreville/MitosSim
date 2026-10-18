"""Phase A of synthetic data generation: masks only (CPU-bound, heavily
parallelized). Phase B (`gan_render_synthetic.py`) turns these into actual
images by running them through the pretrained context-encoder GAN
conditioned on each mask -- the same `gan_from_simulator_mask` approach
`notebooks/build_human_eval_200_notebook.py` uses, not a direct optical
render. A direct MitoSim render looks too clean/simulated for a
classifier trained on it to transfer to real data; the GAN's job is
exactly to bridge that gap.

Target size: `SYNTH_MULTIPLIER` x the real training-set size (5x by
default), split evenly across the 4 classifier classes -- not matched to
the real class proportions, since the whole point is to shore up the
classes real data under-represents (`ring_shape` has only 23 real
training examples). A separate, smaller synthetic validation set is
generated at the same ratio real_val bears to real_train.

Class -> generator mapping: see gan_render_synthetic.py's module
docstring for the full picture; this script only produces the mask side:
    prometaphase        <- prometaphase.configure(), random orientation
    anaphase_telophase  <- anaphase.configure() / telophase.configure(),
                            split evenly, random orientation
    metaphase / ring_shape <- metaphase.configure(), tilt-dependent
                            rejection sampling on hollowness
                            (mitosim.score.descriptors.hollowness) -- a
                            cheap (geometry-only, fast_slab_mask) pre-check
                            first, then an authoritative check against the
                            actual rendered mask
                            (render_chromatin_mask(...)) for draws the
                            pre-check lets through.

Per-sample parameter jitter: every numeric PhaseParams field is
multiplicatively jittered by `JITTER_FRAC` (uniform, independent per
field per sample).

Run: python generate_synthetic.py
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for `import mitosim`

from mitosim.geometry.transform import rotate_cell_state, rotation_aligning_z_to
from mitosim.phases import anaphase, metaphase, prometaphase, telophase
from mitosim.phases.base import PhaseParams
from mitosim.render.optics import OpticsParams, render_chromatin_mask
from mitosim.render.section import (
    fast_slab_mask,
    orientation_from_tilt_azimuth,
    sample_offset,
    sample_orientation,
)
from mitosim.score.descriptors import hollowness

MASKS_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "synth_masks"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "splits"
REAL_TRAIN = SPLITS_DIR / "real_train.csv"
REAL_VAL = SPLITS_DIR / "real_val.csv"

SYNTH_MULTIPLIER = 5
CLASSES = ("prometaphase", "metaphase", "ring_shape", "anaphase_telophase")

JITTER_FRAC = 0.15
JITTERED_FIELDS = (
    "spindle_half_length", "nuclear_radius", "plate_radius", "plate_thickness",
    "hollow_fraction", "progress",
)

THICKNESS_UM = 3.0
MASK_OPTICS = OpticsParams(pixel_size_um=0.25, patch_size_px=128)
MASK_PIXEL_AREA = MASK_OPTICS.pixel_size_um ** 2

BASE_PARAMS = PhaseParams()
#: A second, more pronounced hollow-centre plate for the ring_shape class.
RING_PARAMS = PhaseParams(plate_radius=4.2, plate_thickness=1.8, hollow_fraction=0.3)
#: anaphase's own representative snapshot is later than PhaseParams' shared
#: default (0.6, not 0.5) -- see PhaseParams.progress's own docstring.
ANAPHASE_PARAMS = dataclasses.replace(BASE_PARAMS, progress=0.6)

MIN_AREA_UM2 = 2.0
HOLLOWNESS_RING = 0.15

# Slack the cheap fast-geometry pre-check gives a draw before the
# authoritative check (on the actual rendered mask) gets to see it -- a
# performance knob, never a correctness one.
CHEAP_HOLLOWNESS_MARGIN = 0.15


def jittered(rng: np.random.Generator, params: PhaseParams) -> PhaseParams:
    updates = {f: getattr(params, f) * rng.uniform(1 - JITTER_FRAC, 1 + JITTER_FRAC) for f in JITTERED_FIELDS}
    return dataclasses.replace(params, **updates)


def optics_mask(state, offset: float) -> np.ndarray:
    """The actual chromatin mask the GAN conditions on (Beer-Lambert
    threshold, not raw geometric projection)."""
    return render_chromatin_mask(state, s=offset, t=THICKNESS_UM, params=MASK_OPTICS)


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(out_path)


def hollowness_of(mask: np.ndarray, pixel_area: float) -> float:
    """Hollowness of a 2D mask, independent of connected-component count: a
    near-face-on metaphase plate's thresholded/blurred rendering often
    fragments into 2+ disconnected arcs, which a naive single-component
    check would miss. Returns nan for a mask too small/empty to have a
    meaningful centre."""
    if mask.sum() * pixel_area < MIN_AREA_UM2:
        return float("nan")
    return hollowness(mask, pixel_area)


def sectioned(rng: np.random.Generator, state, tilt_deg: float | None = None):
    """Returns (rotated_state, offset, is_empty). Cheap: geometry-only
    (fast_slab_mask), no optics -- used purely to decide whether to keep
    this draw before paying for the proper mask render."""
    if tilt_deg is None:
        orientation = sample_orientation(rng, 1)[0]
    else:
        azimuth = rng.uniform(0, 2 * np.pi)
        orientation = orientation_from_tilt_azimuth(np.radians(tilt_deg), azimuth)
    offset = float(sample_offset(rng, state.cell_radius + THICKNESS_UM / 2, 1)[0])
    rotation = rotation_aligning_z_to(orientation)
    rotated = rotate_cell_state(state, rotation)
    mask, pixel_area = fast_slab_mask(rotated.chromatids, offset, THICKNESS_UM, spacing=0.2)
    is_empty = mask.sum() * pixel_area < MIN_AREA_UM2
    return rotated, offset, is_empty


def generate_plain(
    rng: np.random.Generator, n: int, configure_fn, out_subdir: Path, prefix: str, tag: str,
    base_params: PhaseParams = BASE_PARAMS,
) -> list[str]:
    """prometaphase / anaphase / telophase: random orientation, discard
    only genuinely empty (degenerate) sections. `base_params` lets a caller
    override the representative `progress` a phase is jittered around (see
    ANAPHASE_PARAMS above) -- defaults to BASE_PARAMS."""
    paths = []
    attempts = 0
    while len(paths) < n:
        attempts += 1
        params = jittered(rng, base_params)
        state = configure_fn(rng, params)
        rotated, offset, is_empty = sectioned(rng, state, tilt_deg=None)
        if is_empty:
            continue
        mask = optics_mask(rotated, offset)
        if mask.sum() < 2:
            continue
        name = f"{prefix}_{tag}_{len(paths):05d}.png"
        save_mask(mask, out_subdir / name)
        paths.append(str((out_subdir / name).relative_to(MASKS_DIR.parent)))
        if attempts > n * 20:
            raise RuntimeError(f"{prefix}: too many rejected draws ({attempts} for {len(paths)}/{n})")
    return paths


def _sectioned_mask(rng: np.random.Generator, state, tilt_deg: float):
    """Fixed-tilt, random-azimuth orientation + offset -> (rotated_state,
    offset, fast geometric mask, its pixel_area). Shared by
    generate_metaphase/generate_ring_shape, which decide accept/reject on
    hollowness directly (see hollowness_of)."""
    azimuth = rng.uniform(0, 2 * np.pi)
    orientation = orientation_from_tilt_azimuth(np.radians(tilt_deg), azimuth)
    offset = float(sample_offset(rng, state.cell_radius + THICKNESS_UM / 2, 1)[0])
    rotated = rotate_cell_state(state, rotation_aligning_z_to(orientation))
    fast_mask, fast_pixel_area = fast_slab_mask(rotated.chromatids, offset, THICKNESS_UM, spacing=0.2)
    return rotated, offset, fast_mask, fast_pixel_area


def generate_metaphase(rng: np.random.Generator, n: int, tag: str) -> list[str]:
    """Normal metaphase, tilt sampled broadly (0-90deg, default params).
    Kept when the actual rendered mask's own hollowness (hollowness_of)
    stays below the ring threshold. The cheap fast-geometry mask only
    pre-screens obviously ring-like draws (with a generous margin) before
    paying for the real render -- a defocus-blurred render can land on
    either side of the threshold relative to the cheap geometric
    projection, so only the final check on the actual rendered mask is
    authoritative."""
    out_subdir = MASKS_DIR / "metaphase"
    out_subdir.mkdir(parents=True, exist_ok=True)
    paths = []
    attempts = 0
    while len(paths) < n:
        attempts += 1
        tilt_deg = rng.uniform(0.0, 90.0)
        params = jittered(rng, BASE_PARAMS)
        state = metaphase.configure(rng, params)
        rotated, offset, fast_mask, fast_pixel_area = _sectioned_mask(rng, state, tilt_deg)

        fast_hollow = hollowness_of(fast_mask, fast_pixel_area)
        if np.isnan(fast_hollow):
            continue  # empty section
        if fast_hollow >= HOLLOWNESS_RING + CHEAP_HOLLOWNESS_MARGIN:
            continue  # unambiguously ring-like already -- skip the expensive render

        mask = optics_mask(rotated, offset)
        final_hollow = hollowness_of(mask, MASK_PIXEL_AREA)
        if np.isnan(final_hollow) or final_hollow >= HOLLOWNESS_RING:
            continue  # the real render says otherwise -- too hollow for plain metaphase

        name = f"metaphase_{tag}_{len(paths):05d}.png"
        save_mask(mask, out_subdir / name)
        paths.append(str((out_subdir / name).relative_to(MASKS_DIR.parent)))
        if attempts > n * 10:
            raise RuntimeError(f"metaphase: too many rejected draws ({attempts} for {len(paths)}/{n})")
    print(f"  metaphase[{tag}]: {attempts} draws -> {len(paths)} kept")
    return paths


def generate_ring_shape(rng: np.random.Generator, n: int, tag: str) -> list[str]:
    """Ring-shape metaphase: tilt concentrated near face-on (70-90deg),
    pronounced-hollow params (`RING_PARAMS` above). Kept when the actual
    rendered mask's own hollowness clears the ring threshold -- see
    generate_metaphase's docstring for why only the final render is
    authoritative."""
    out_subdir = MASKS_DIR / "ring_shape"
    out_subdir.mkdir(parents=True, exist_ok=True)
    paths = []
    attempts = 0
    while len(paths) < n:
        attempts += 1
        tilt_deg = rng.uniform(70.0, 90.0)
        params = jittered(rng, RING_PARAMS)
        state = metaphase.configure(rng, params)
        rotated, offset, fast_mask, fast_pixel_area = _sectioned_mask(rng, state, tilt_deg)

        fast_hollow = hollowness_of(fast_mask, fast_pixel_area)
        if np.isnan(fast_hollow):
            continue  # empty section
        if fast_hollow < HOLLOWNESS_RING - CHEAP_HOLLOWNESS_MARGIN:
            continue  # unambiguously not ring-like yet -- skip the expensive render

        mask = optics_mask(rotated, offset)
        final_hollow = hollowness_of(mask, MASK_PIXEL_AREA)
        if np.isnan(final_hollow) or final_hollow < HOLLOWNESS_RING:
            continue  # the real render says otherwise -- not hollow enough for ring_shape

        name = f"ring_shape_{tag}_{len(paths):05d}.png"
        save_mask(mask, out_subdir / name)
        paths.append(str((out_subdir / name).relative_to(MASKS_DIR.parent)))
        if attempts > n * 400:
            raise RuntimeError(f"ring_shape: too many rejected draws ({attempts} for {len(paths)}/{n})")
    print(f"  ring_shape[{tag}]: {attempts} draws -> {len(paths)} kept")
    return paths


def generate_anaphase_telophase(rng: np.random.Generator, n: int, tag: str) -> list[str]:
    out_subdir = MASKS_DIR / "anaphase_telophase"
    out_subdir.mkdir(parents=True, exist_ok=True)
    n_anaphase = n // 2
    n_telophase = n - n_anaphase
    paths = generate_plain(rng, n_anaphase, anaphase.configure, out_subdir, "anaphase", tag, base_params=ANAPHASE_PARAMS)
    paths += generate_plain(rng, n_telophase, telophase.configure, out_subdir, "telophase", tag)
    return paths


GENERATORS = {
    "prometaphase": lambda rng, n, tag: generate_plain(
        rng, n, prometaphase.configure, MASKS_DIR / "prometaphase", "prometaphase", tag
    ),
    "metaphase": generate_metaphase,
    "ring_shape": generate_ring_shape,
    "anaphase_telophase": generate_anaphase_telophase,
}


def _run_task(args: tuple[str, int, int, int, str]) -> tuple[str, list[str]]:
    cls, n, seed, chunk_id, split_name = args
    (MASKS_DIR / cls).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # split_name in the tag matters: chunk_id resets to 0 for every split,
    # so train and val chunks would otherwise produce identically-named
    # files in the same shared per-class directory and silently overwrite
    # each other (train run first, val second -- val clobbering ~1/6 of
    # train's files was exactly this bug, caught via a train/val manifest
    # path-overlap check after the fact).
    paths = GENERATORS[cls](rng, n, tag=f"{split_name}_c{chunk_id}")
    return cls, paths


def chunk_counts(n_total: int, chunk_size: int) -> list[int]:
    n_chunks = max(1, -(-n_total // chunk_size))
    base, remainder = divmod(n_total, n_chunks)
    return [base + (1 if i < remainder else 0) for i in range(n_chunks)]


def generate_split(n_total: int, split_name: str, base_seed: int, pool, chunk_size: int) -> None:
    n_per_class = n_total // len(CLASSES)
    print(f"=== {split_name}: {n_total} total, {n_per_class} per class (chunk_size={chunk_size}) ===")

    tasks = []
    seed = base_seed
    for cls in CLASSES:
        for chunk_id, n_chunk in enumerate(chunk_counts(n_per_class, chunk_size)):
            tasks.append((cls, n_chunk, seed, chunk_id, split_name))
            seed += 1
    print(f"  {len(tasks)} chunk tasks across {len(CLASSES)} classes")

    results: dict[str, list[str]] = {c: [] for c in CLASSES}
    for cls, paths in pool.imap_unordered(_run_task, tasks):
        results[cls].extend(paths)

    manifest_path = SPLITS_DIR / f"synth_masks_{split_name}.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mask_path", "class"])
        writer.writeheader()
        for cls in CLASSES:
            for p in results[cls]:
                writer.writerow({"mask_path": p, "class": cls})
    print(f"wrote {manifest_path} ({sum(len(v) for v in results.values())} rows)")


def main() -> None:
    import multiprocessing as mp
    import os

    n_workers = int(os.environ.get("SYNTH_WORKERS", min(os.cpu_count() or 4, 8)))
    chunk_size = int(os.environ.get("SYNTH_CHUNK_SIZE", 150))
    print(f"using {n_workers} worker processes, chunk_size={chunk_size}")

    real_train_n = sum(1 for _ in open(REAL_TRAIN)) - 1
    real_val_n = sum(1 for _ in open(REAL_VAL)) - 1
    synth_train_n = real_train_n * SYNTH_MULTIPLIER
    synth_val_n = round(synth_train_n * (real_val_n / real_train_n))

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        generate_split(synth_train_n, "train", base_seed=100, pool=pool, chunk_size=chunk_size)
        generate_split(synth_val_n, "val", base_seed=10000, pool=pool, chunk_size=chunk_size)


if __name__ == "__main__":
    main()

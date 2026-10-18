"""Mask-source ablation: same GAN, same rendering procedure (matched
context-photo swap + erase_mask_arr) as the main synthetic pipeline, but
conditioned on REAL masks instead of MitoSim-simulated ones -- isolates
whether the real+synth benefit comes from the GAN's photorealistic
context-recombination alone, or specifically from MitoSim's ability to
generate mask *shapes* beyond what the small real training set has.

Real masks are only available (in weights/context_masks) for a subset of
the real_train/real_val patches. Per class, masks are drawn *with
replacement* from that class's own split (train masks only from
real_train's own patches, val only from real_val's) up to the same
per-class target as the main pipeline's 5x condition -- computed from the
current real_train.csv/real_val.csv row counts, same as
generate_synthetic.py's own SYNTH_MULTIPLIER computation, so this stays in
sync automatically if the real split ever changes. ring_shape, the
scarcest class, ends up reusing its small handful of real masks many
times over, with a different swapped-in context photo each time; that
heavy reuse is the point, not a flaw. No CPU-bound rejection sampling is
needed here (unlike the MitoSim path), so mask selection and GAN rendering
happen in one pass.

Run: python generate_realmask_ablation.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from gan_render_synthetic import CHECKPOINT_PATH, UNetGenerator, pick_device, render_one
from mitosim.render.match_context import load_ellipse_db

DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
SPLITS_DIR = DATA_CACHE / "splits"
PATCHES_DIR = DATA_CACHE / "synth_patches_realmask"
CONTEXT_MASKS_DIR = Path(__file__).resolve().parent.parent.parent / "weights" / "context_masks"

CLASSES = ("prometaphase", "metaphase", "ring_shape", "anaphase_telophase")
SYNTH_MULTIPLIER = 5  # must match generate_synthetic.py's own constant


def _target_per_class() -> dict[str, int]:
    """Same per-class train/val targets as the main pipeline's 5x
    condition, computed from the current real splits (not hardcoded) so
    this never silently drifts out of sync with them."""
    real_train_n = sum(1 for _ in open(SPLITS_DIR / "real_train.csv")) - 1
    real_val_n = sum(1 for _ in open(SPLITS_DIR / "real_val.csv")) - 1
    synth_train_n = real_train_n * SYNTH_MULTIPLIER
    synth_val_n = round(synth_train_n * (real_val_n / real_train_n))
    return {"train": synth_train_n // len(CLASSES), "val": synth_val_n // len(CLASSES)}


TARGET_PER_CLASS = _target_per_class()


def available_real_masks(split_name: str) -> dict[str, list[str]]:
    """class -> list of dataset_uid stems (e.g. 'MIDOG21_528') that have
    both a real_{split} row and a real mask file."""
    rows = list(csv.DictReader(open(SPLITS_DIR / f"real_{split_name}.csv")))
    mask_files = {p.stem for p in CONTEXT_MASKS_DIR.glob("*.png")}
    by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
    for r in rows:
        stem = f"{r['dataset']}_{r['uid']}"
        if stem in mask_files:
            by_class[r["class"]].append(stem)
    return by_class


def render_split(G, device, ellipse_db, split_name: str, seed_base: int) -> None:
    by_class = available_real_masks(split_name)
    n_target = TARGET_PER_CLASS[split_name]
    print(f"=== {split_name} ===")
    for cls in CLASSES:
        print(f"  {cls}: {len(by_class[cls])} real masks available, sampling {n_target} with replacement")

    out_rows = []
    rng_pick = np.random.default_rng(seed_base)
    for cls in CLASSES:
        pool = by_class[cls]
        out_dir = PATCHES_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_target):
            stem = pool[rng_pick.integers(0, len(pool))]
            mask_img = Image.open(CONTEXT_MASKS_DIR / f"{stem}.png").convert("L")
            if mask_img.size != (128, 128):
                mask_img = mask_img.resize((128, 128), Image.NEAREST)
            mask_arr = np.array(mask_img) > 0

            rng = np.random.default_rng(seed_base + 1 + i if cls == CLASSES[0] else hash((cls, i)) % (2**31))
            img = render_one(G, device, ellipse_db, mask_arr, rng)

            name = f"{cls}_{split_name}_{i:05d}.png"
            img.save(out_dir / name)
            out_rows.append({"path": str((out_dir / name).relative_to(DATA_CACHE)), "class": cls})

            if (i + 1) % 500 == 0:
                print(f"    {cls}: {i + 1}/{n_target}")

    manifest_path = SPLITS_DIR / f"synth_{split_name}_realmask.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {manifest_path} ({len(out_rows)} rows)")


def main() -> None:
    device = pick_device()
    print(f"device={device}")

    import torch
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    G = UNetGenerator().to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch', '?')})")

    ellipse_db = load_ellipse_db()

    render_split(G, device, ellipse_db, "train", seed_base=700000)
    render_split(G, device, ellipse_db, "val", seed_base=800000)


if __name__ == "__main__":
    main()

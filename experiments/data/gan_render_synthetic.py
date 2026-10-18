"""Phase B of synthetic data generation: turns the masks `generate_synthetic.py`
produced into actual images by running them through the pretrained
context-encoder GAN (`weights/checkpoint_best.pt`), conditioned on each
mask -- the exact `gan_from_simulator_mask` approach from
`notebooks/build_human_eval_200_notebook.py`: pick a real context photo
whose own mask is in the same angle/size category ("matched" bucketing),
erase both the new mask's footprint and the context photo's own pre-
existing structure, and let the GAN paint the new shape in. This is
deliberately NOT a direct MitoSim optical render -- a direct render looks
too clean/simulated for a classifier trained on it to transfer to real
data, which is exactly the gap the GAN is there to close.

No border -- that was specific to the human_eval_200 visual QC page's own
stimulus images and never belonged on classifier training data.

Needs a CUDA or MPS device to run in reasonable time (falls back to CPU,
slow). Single process -- this is GPU-bound, not CPU-bound like Phase A,
so it doesn't need (and would just contend for the device with) multiple
workers.

Run: python gan_render_synthetic.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from gan_context_encoder import UNetGenerator, build_sample
from mitosim.render.inpaint import load_context_image, load_context_mask, pick_context_image
from mitosim.render.match_context import load_ellipse_db, pick_bucketed_context_image, transform_image

DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
MASKS_DIR = DATA_CACHE / "synth_masks"
PATCHES_DIR = DATA_CACHE / "synth_patches"
SPLITS_DIR = DATA_CACHE / "splits"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "weights" / "checkpoint_best.pt"

IMAGE_SIZE = 128
CONTEXT = "matched"  # "matched" | "random"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_img(t: torch.Tensor) -> Image.Image:
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def render_one(G, device, ellipse_db, mask_arr: np.ndarray, rng: np.random.Generator) -> Image.Image:
    name, rotation_deg, mirror = CONTEXT, 0, False
    if CONTEXT == "matched":
        try:
            match, context_image = pick_bucketed_context_image(mask_arr, rng, db=ellipse_db)
            name, rotation_deg, mirror = match.filename, match.rotation_deg, match.mirror
        except ValueError:
            name, context_image = pick_context_image(rng)
    else:
        name, context_image = pick_context_image(rng)

    if context_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        context_image = context_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    context_mask_img = load_context_mask(name).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    context_mask_arr = np.array(transform_image(context_mask_img, rotation_deg, mirror)) > 0

    sample = build_sample(context_image, mask_arr, erase_mask_arr=context_mask_arr)
    with torch.no_grad():
        fake = G(sample.context_input.unsqueeze(0).to(device))[0]
    return to_img(fake)


def render_split(G, device, ellipse_db, split_name: str, seed_base: int) -> None:
    manifest_in = SPLITS_DIR / f"synth_masks_{split_name}.csv"
    rows = list(csv.DictReader(open(manifest_in)))
    print(f"=== {split_name}: {len(rows)} masks to render ===")

    out_rows = []
    for i, row in enumerate(rows):
        cls = row["class"]
        mask_path = DATA_CACHE / row["mask_path"]
        mask_arr = np.array(Image.open(mask_path).convert("L")) > 0

        rng = np.random.default_rng(seed_base + i)
        img = render_one(G, device, ellipse_db, mask_arr, rng)

        out_dir = PATCHES_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = mask_path.name  # same stem as the mask, for traceability
        img.save(out_dir / out_name)
        out_rows.append({"path": str((out_dir / out_name).relative_to(DATA_CACHE)), "class": cls})

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)} rendered")

    manifest_out = SPLITS_DIR / f"synth_{split_name}.csv"
    with open(manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {manifest_out} ({len(out_rows)} rows)")


def main() -> None:
    device = pick_device()
    print(f"device={device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    G = UNetGenerator().to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch', '?')})")

    ellipse_db = load_ellipse_db()

    render_split(G, device, ellipse_db, "train", seed_base=500000)
    render_split(G, device, ellipse_db, "val", seed_base=600000)


if __name__ == "__main__":
    main()

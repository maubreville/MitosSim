"""Builds the real-image side of the phase-classifier dataset from the
expert-rated ground truth at `Atypical_MIDOG_TUPAC.csv`.

Filtering (per the experiment spec): keep only rows where both experts
agree on the label AND both say the figure is not atypical. Label mapping
to the 4 classifier classes:

    NMF prometaphase        -> prometaphase
    NMF metaphase           -> metaphase
    NMF ring shape          -> ring_shape
    NMF anaphase-telophase  -> anaphase_telophase

All AMF (atypical) rows and all disagreement rows are dropped. There is no
real "prophase" category in this ground truth at all -- pathologists don't
operationally count prophase nuclei as mitotic figures -- so that class
never appears here; it's synthetic-only in the dataset this feeds into.

Patches are extracted directly from the local whole-slide images at
128x128, level 0, centred on (x, y) -- the exact convention already used
by `AtypicalMF_Dataset/ExtractAtypical.ipynb`, so these line up pixel-for-
pixel with any patches that notebook already produced.

Run: python extract_real_patches.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import openslide

CSV_PATH = Path("/Users/marcaubreville/code/AtypicalMF_Dataset/Atypical_MIDOG_TUPAC.csv")
WSI_ROOT = Path("/Users/marcaubreville/code/AtypicalMF_Dataset")
SLIDE_FOLDERS = {
    "MIDOG21": WSI_ROOT / "images",
    "TUPAC16": WSI_ROOT / "TUPAC16_AlternativeLabels" / "stitched",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "real_patches"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "real_manifest.csv"
PATCH_SIZE = 128

LABEL_TO_CLASS = {
    "NMF prometaphase": "prometaphase",
    "NMF metaphase": "metaphase",
    "NMF ring shape": "ring_shape",
    "NMF anaphase-telophase": "anaphase_telophase",
}
CLASSES = ("prometaphase", "metaphase", "ring_shape", "anaphase_telophase")


def agree_nonatypical(row: dict) -> bool:
    return (
        row["expert1_label"] == row["expert2_label"]
        and row["expert1_atypical"] == "False"
        and row["expert2_atypical"] == "False"
    )


def extract_patch(slide_path: Path, x: float, y: float) -> "PIL.Image.Image":
    slide = openslide.open_slide(str(slide_path))
    try:
        return slide.read_region(
            (int(x) - PATCH_SIZE // 2, int(y) - PATCH_SIZE // 2), level=0, size=(PATCH_SIZE, PATCH_SIZE)
        ).convert("RGB")
    finally:
        slide.close()


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH)))
    qualifying = [r for r in rows if agree_nonatypical(r) and r["expert1_label"] in LABEL_TO_CLASS]
    print(f"{len(rows)} total rows, {len(qualifying)} pass agreement+non-atypical+known-label filter")

    for cls in CLASSES:
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    skipped = []
    for row in qualifying:
        cls = LABEL_TO_CLASS[row["expert1_label"]]
        folder = SLIDE_FOLDERS.get(row["dataset"])
        if folder is None:
            skipped.append((row["dataset"], row["uid"], "unknown dataset"))
            continue
        slide_path = folder / row["slide"]
        if not slide_path.is_file():
            skipped.append((row["dataset"], row["uid"], f"missing slide {slide_path}"))
            continue

        out_name = f"{row['dataset']}_{row['uid']}.png"
        out_path = OUT_DIR / cls / out_name
        if not out_path.is_file():
            patch = extract_patch(slide_path, float(row["x"]), float(row["y"]))
            patch.save(out_path)

        manifest_rows.append({
            "path": str(out_path.relative_to(OUT_DIR.parent)), "class": cls,
            "dataset": row["dataset"], "uid": row["uid"], "slide": row["slide"],
        })

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "dataset", "uid", "slide"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {len(manifest_rows)} patches to {OUT_DIR}")
    if skipped:
        print(f"skipped {len(skipped)} rows (missing slide or unknown dataset), e.g. {skipped[:5]}")
    for cls in CLASSES:
        n = sum(1 for r in manifest_rows if r["class"] == cls)
        print(f"  {cls}: {n}")
    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

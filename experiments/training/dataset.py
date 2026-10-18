"""PyTorch Dataset over the manifests written by `data/build_splits.py`
and `data/generate_synthetic.py`. Both manifests share the same two
columns (`path`, `class`) and the same 4 class names, so a single class
handles real, synthetic, or a concatenation of both.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ("prometaphase", "metaphase", "ring_shape", "anaphase_telophase")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"


def read_manifest(name: str) -> list[dict]:
    """`name` is one of real_train/real_val/real_test/synth_train/synth_val."""
    path = DATA_CACHE / "splits" / f"{name}.csv"
    return list(csv.DictReader(open(path)))


class PhaseDataset(Dataset):
    def __init__(self, manifest_names: list[str], transform=None):
        self.rows = []
        for name in manifest_names:
            self.rows.extend(read_manifest(name))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = np.array(Image.open(DATA_CACHE / row["path"]).convert("RGB"))
        label = CLASS_TO_IDX[row["class"]]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for row in self.rows:
            counts[row["class"]] += 1
        return counts


#: Which manifests feed each experiment condition. real_synth_realmask is
#: the mask-source ablation (see README): val is the full, unscaled
#: synth_val_realmask -- only the train-side mask source varies, so early
#: stopping's signal stays comparable against real_synth.
CONDITION_MANIFESTS = {
    "real": {"train": ["real_train"], "val": ["real_val"]},
    "real_synth": {"train": ["real_train", "synth_train"], "val": ["real_val", "synth_val"]},
    "synth": {"train": ["synth_train"], "val": ["synth_val"]},
    "real_synth_realmask": {"train": ["real_train", "synth_train_realmask"], "val": ["real_val", "synth_val_realmask"]},
}
TEST_MANIFEST = ["real_test"]  # always real, shared by every condition

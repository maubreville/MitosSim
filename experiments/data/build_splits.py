"""Stratified-by-class, grouped-by-slide train/val/test split of the real-patch manifest.

Run: python build_splits.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data_cache" / "real_manifest.csv"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "splits"

N_TEST_FOLDS = 5    # 1 fold held out ~ 20% test
N_VAL_FOLDS = 7     # 1 fold held out ~ 15% of the trainval remainder
SEED = 0


def _group_split(rows: list[dict], n_folds: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Splits `rows` into (kept, held_out) via one fold of a StratifiedGroupKFold over `class` labels grouped by (dataset, slide),
    so no slide straddles the two halves."""
    labels = [r["class"] for r in rows]
    groups = [f"{r['dataset']}/{r['slide']}" for r in rows]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kept_idx, held_out_idx = next(iter(splitter.split(np.zeros(len(rows)), labels, groups)))
    return [rows[i] for i in kept_idx], [rows[i] for i in held_out_idx]


def _group_set(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["dataset"], r["slide"]) for r in rows}


def main() -> None:
    rows = list(csv.DictReader(open(MANIFEST_PATH)))

    trainval_rows, test_rows = _group_split(rows, N_TEST_FOLDS, SEED)
    train_rows, val_rows = _group_split(trainval_rows, N_VAL_FOLDS, SEED)

    assert not (_group_set(train_rows) & _group_set(val_rows))
    assert not (_group_set(train_rows) & _group_set(test_rows))
    assert not (_group_set(val_rows) & _group_set(test_rows))

    counts = defaultdict(lambda: defaultdict(int))
    for name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for r in split_rows:
            counts[r["class"]][name] += 1
    for cls in sorted(counts):
        c = counts[cls]
        total = c["train"] + c["val"] + c["test"]
        print(f"{cls:>20}: {c['train']} train / {c['val']} val / {c['test']} test  (of {total} total)")
    print(f"slides: {len(_group_set(train_rows))} train / {len(_group_set(val_rows))} val / "
          f"{len(_group_set(test_rows))} test (of {len(_group_set(rows))} total)")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class", "dataset", "uid", "slide"]
    for name, split_rows in (("real_train", train_rows), ("real_val", val_rows), ("real_test", test_rows)):
        with open(SPLITS_DIR / f"{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"wrote {len(split_rows)} rows to {SPLITS_DIR / f'{name}.csv'}")


if __name__ == "__main__":
    main()

"""Single training run: one (condition, seed) pair.

Condition picks which manifests feed train/val (see dataset.py's
CONDITION_MANIFESTS); the test set is always real_test, for all three
conditions, so results are comparable. Early stopping on validation loss,
patience configurable (default 5, per the experiment spec). Logs every
epoch to Weights & Biases and writes a self-contained JSON result (final
test metrics + full epoch history) to --out-dir, so aggregation across
repeats doesn't require querying the wandb API.

Run: python train.py --condition real_synth --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, unused directly but kept
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for the sibling modules below

from augmentations import eval_transform, train_transform
from dataset import CLASSES, CONDITION_MANIFESTS, TEST_MANIFEST, PhaseDataset
from model import build_model, pick_device


def class_weights(dataset: PhaseDataset, device: str) -> torch.Tensor:
    counts = dataset.class_counts()
    freq = np.array([counts[c] for c in CLASSES], dtype=np.float64)
    freq[freq == 0] = 1.0  # guard: a class absent from this condition's train split (e.g. real-only has none anyway)
    weights = freq.sum() / (len(CLASSES) * freq)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
    mean_loss = total_loss / len(loader.dataset)
    acc = float((all_preds == all_labels).mean())
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return mean_loss, acc, balanced_acc, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=sorted(CONDITION_MANIFESTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="mitosim-phase-classifier")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"condition={args.condition} seed={args.seed} device={device}")

    manifests = CONDITION_MANIFESTS[args.condition]
    train_ds = PhaseDataset(manifests["train"], transform=train_transform(args.image_size))
    val_ds = PhaseDataset(manifests["val"], transform=eval_transform(args.image_size))
    test_ds = PhaseDataset(TEST_MANIFEST, transform=eval_transform(args.image_size))
    print(f"train={len(train_ds)} ({train_ds.class_counts()})")
    print(f"val={len(val_ds)} ({val_ds.class_counts()})")
    print(f"test={len(test_ds)} ({test_ds.class_counts()})")

    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    import wandb
    run = wandb.init(
        project=args.wandb_project, mode=args.wandb_mode, group=args.condition,
        name=f"{args.condition}_seed{args.seed}",
        config=vars(args) | {"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)},
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []
    t0 = time.time()

    for epoch in range(args.max_epochs):
        train_loss, train_acc, train_bacc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_bacc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_balanced_acc": train_bacc,
            "val_loss": val_loss, "val_acc": val_acc, "val_balanced_acc": val_bacc,
        })
        wandb.log(history[-1])
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_bacc={val_bacc:.3f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch} (patience={args.patience})")
                break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_bacc, test_preds, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(CLASSES))))

    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_balanced_acc": test_bacc})
    wandb.summary["test_acc"] = test_acc
    wandb.summary["test_balanced_acc"] = test_bacc
    wandb.finish()

    result = {
        "condition": args.condition, "seed": args.seed, "arch": args.arch,
        "n_epochs_run": len(history), "wall_time_s": time.time() - t0,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss, "test_acc": test_acc, "test_balanced_acc": test_bacc,
        "per_class_report": report, "confusion_matrix": cm.tolist(),
        "history": history,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"test_acc={test_acc:.4f} test_balanced_acc={test_bacc:.4f}  wrote {out_path}")


if __name__ == "__main__":
    main()

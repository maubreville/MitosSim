#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

IMAGE_SIZE = 128
CROP_SIZE = 64  # the hole-rect crop is resized to this before hitting the discriminator
HOLE_L1_WEIGHT = 0.0  # optional stabilizer (classic Context Encoder uses a small one); 0 = pure adversarial on the hole
HOLE_MARGIN = 1.15 * 1.10
DISCRIMINATOR_RECT_MARGIN = 1.6


def _fit_ellipse(mask: np.ndarray, margin: float):
    """Shared fit: (centroid, evecs, evals, r_max) or None if too few
    foreground pixels to fit at all. evals/evecs ascending by eigenvalue;
    r_max already includes `margin`."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    coords = np.stack([xs, ys], axis=-1).astype(float)
    centroid = coords.mean(axis=0)
    cov = np.cov((coords - centroid).T)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    evals = np.maximum(evals, 1e-9)
    whitened = (coords - centroid) @ evecs / np.sqrt(evals)
    r_max = np.linalg.norm(whitened, axis=-1).max() * margin
    return centroid, evecs, evals, r_max


def ellipse_mask(mask: np.ndarray, margin: float = HOLE_MARGIN) -> np.ndarray:
    """Boolean array, same shape as `mask`: True inside the fitted (and
    `margin`-scaled) ellipse -- the actual hole."""
    fit = _fit_ellipse(mask, margin)
    h, w = mask.shape
    if fit is None:
        return np.ones((h, w), dtype=bool)  # degenerate mask: hole the whole frame rather than guess
    centroid, evecs, evals, r_max = fit
    yy, xx = np.indices((h, w))
    grid = np.stack([xx, yy], axis=-1).astype(float) - centroid
    grid_whitened = grid @ evecs / np.sqrt(evals)
    return np.linalg.norm(grid_whitened, axis=-1) <= r_max


def ellipse_bounding_rect(mask: np.ndarray, margin: float = 1.15) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) -- axis-aligned bounding box of the same fitted
    ellipse `ellipse_mask` rasterizes -- for cropping a rectangular tensor
    out for the discriminator only (see module note above); never use this
    as the hole itself."""
    h, w = mask.shape
    fit = _fit_ellipse(mask, margin)
    if fit is None:
        return 0, 0, w, h
    centroid, evecs, evals, r_max = fit
    semi_minor, semi_major = r_max * np.sqrt(evals)  # evals ascending

    major_dir = evecs[:, -1]
    theta = np.arctan2(major_dir[1], major_dir[0])
    half_w = np.hypot(semi_major * np.cos(theta), semi_minor * np.sin(theta))
    half_h = np.hypot(semi_major * np.sin(theta), semi_minor * np.cos(theta))

    x0 = int(np.clip(np.floor(centroid[0] - half_w), 0, w))
    x1 = int(np.clip(np.ceil(centroid[0] + half_w), 0, w))
    y0 = int(np.clip(np.floor(centroid[1] - half_h), 0, h))
    y1 = int(np.clip(np.ceil(centroid[1] + half_h), 0, h))
    if x1 <= x0:
        x1 = min(x0 + 1, w)
    if y1 <= y0:
        y1 = min(y0 + 1, h)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    context_input: torch.Tensor
    real: torch.Tensor           # (3, H, W): the real, uncropped image, in [-1, 1]
    hole_mask: torch.Tensor      # (1, H, W): 1 inside the true ellipse, 0 outside -- NOT the bounding rect
    rect: tuple[int, int, int, int]  # only for the discriminator's crop (see module note above)


class PairedImageMaskDataset(Dataset):
    """Pairs `images_dir/<name>` with `masks_dir/<name>` by identical
    filename -- same convention as the rest of this project's tooling, just
    reimplemented here with plain pathlib/PIL (no mitosim import)."""

    def __init__(self, images_dir: str | Path, masks_dir: str | Path, image_size: int = IMAGE_SIZE):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.filenames = sorted(p.name for p in self.images_dir.glob("*.png"))
        if not self.filenames:
            raise FileNotFoundError(f"no .png files in {self.images_dir}")
        missing = [f for f in self.filenames if not (self.masks_dir / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) have no same-named mask, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Sample:
        name = self.filenames[idx]
        image = Image.open(self.images_dir / name).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(self.masks_dir / name).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        mask_arr = np.array(mask) > 0
        return build_sample(image, mask_arr)


def build_sample(image: Image.Image, mask_arr: np.ndarray, erase_mask_arr: np.ndarray | None = None) -> Sample:

    rect = ellipse_bounding_rect(mask_arr, margin=DISCRIMINATOR_RECT_MARGIN)  # for the discriminator's crop only
    hole_bool = ellipse_mask(mask_arr, margin=HOLE_MARGIN)  # the actual hole -- true ellipse, not the rect
    if erase_mask_arr is not None:
        hole_bool = hole_bool | ellipse_mask(erase_mask_arr, margin=HOLE_MARGIN)
    hole_mask = hole_bool.astype(np.float32)

    image_arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # -> [-1, 1]
    cropout = image_arr.copy()
    cropout[hole_bool] = 1.0  # white fill, in [-1, 1] terms white == 1.0 -- ellipse-shaped, not rectangular

    real_t = torch.from_numpy(image_arr).permute(2, 0, 1)
    cropout_t = torch.from_numpy(cropout).permute(2, 0, 1)
    hole_t = torch.from_numpy(hole_mask)[None]
    chromatin_t = torch.from_numpy(mask_arr.astype(np.float32))[None]  # the actual conditioning signal

    return Sample(
        context_input=torch.cat([cropout_t, hole_t, chromatin_t], dim=0),
        real=real_t,
        hole_mask=hole_t,
        rect=rect,
    )


def collate(samples: list[Sample]):
    return (
        torch.stack([s.context_input for s in samples]),
        torch.stack([s.real for s in samples]),
        torch.stack([s.hole_mask for s in samples]),
        [s.rect for s in samples],
    )


def crop_and_resize(images: torch.Tensor, rects: list[tuple[int, int, int, int]], size: int = CROP_SIZE) -> torch.Tensor:
    crops = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        crop = images[i : i + 1, :, y0:y1, x0:x1]
        crops.append(F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False))
    return torch.cat(crops, dim=0)


def split_dataset(dataset: "PairedImageMaskDataset", val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
    if n - n_val < 1:
        raise ValueError(f"--val-fraction {val_fraction} leaves no training samples out of {n} total")
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _down(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch, affine=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """128 -> 64 -> 32 -> 16 -> 8 -> 4, mirrored back up with skip connections -- standard pix2pix-style U-Net, 
    sized for 128x128.

    `in_ch=5` by default: white-cropout RGB (3) + erasure/hole-location (1)
    + target chromatin mask (1) -- see `Sample.context_input`'s docstring
    for why both mask-like channels are needed, not just one."""

    def __init__(self, in_ch: int = 5, out_ch: int = 3):
        super().__init__()
        self.e1 = nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)  # 128 -> 64, no norm on the first layer
        self.e2 = _down(64, 128)    # 64 -> 32
        self.e3 = _down(128, 256)   # 32 -> 16
        self.e4 = _down(256, 512)   # 16 -> 8
        self.e5 = _down(512, 512)   # 8 -> 4

        self.d1 = _up(512, 512, dropout=True)       # 4 -> 8
        self.d2 = _up(512 + 512, 256)                # 8 -> 16  (skip from e4)
        self.d3 = _up(256 + 256, 128)                # 16 -> 32 (skip from e3)
        self.d4 = _up(128 + 128, 64)                 # 32 -> 64 (skip from e2)
        self.out = nn.ConvTranspose2d(64 + 64, out_ch, 4, stride=2, padding=1)  # 64 -> 128 (skip from e1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        d1 = self.d1(e5)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        out = self.out(torch.cat([d4, e1], dim=1))
        return torch.tanh(out)


# ---------------------------------------------------------------------------
# Discriminator: PatchGAN, applied only to the cropped hole-rect region
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),            # 64 -> 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),  # 32 -> 16
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),  # 16 -> 8
            nn.Conv2d(256, 1, 4, stride=1, padding=1),   # 8 -> 7: a small patch grid of logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def masked_l1(fake: torch.Tensor, real: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
    diff = (fake - real).abs() * weight_mask
    return diff.sum() / weight_mask.sum().clamp(min=1.0)


def train(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    wandb_run = None
    if args.wandb:
        import wandb  # lazy/optional -- the script has no hard dependency on it otherwise

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
            mode=args.wandb_mode, config=vars(args),
        )

    full_dataset = PairedImageMaskDataset(args.images_dir, args.masks_dir, image_size=args.image_size)
    train_dataset, val_dataset = split_dataset(full_dataset, args.val_fraction, args.val_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    print(f"{len(full_dataset)} paired samples -> {len(train_dataset)} train / {len(val_dataset)} val (seed={args.val_seed})")

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if start_epoch >= args.epochs:
            raise SystemExit(
                f"--resume checkpoint is already at epoch {start_epoch - 1}, which is >= --epochs {args.epochs} "
                f"-- raise --epochs past {start_epoch - 1} to continue training"
            )
        print(f"resumed from {args.resume} (epoch {ckpt.get('epoch', '?')}) -- continuing at epoch {start_epoch}")
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_l1 = float("inf")
    epochs_without_improvement = 0

    step = 0
    for epoch in range(start_epoch, args.epochs):
        for context_input, real, hole_mask, rects in train_loader:
            context_input, real, hole_mask = context_input.to(device), real.to(device), hole_mask.to(device)

            fake = G(context_input)
            real_crop = crop_and_resize(real, rects)
            fake_crop = crop_and_resize(fake, rects)

            # -- discriminator step --
            opt_d.zero_grad()
            d_real = D(real_crop)
            d_fake = D(fake_crop.detach())
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            opt_d.step()

            # -- generator step --
            opt_g.zero_grad()
            d_fake_for_g = D(fake_crop)
            loss_adv = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_l1_context = masked_l1(fake, real, 1.0 - hole_mask)
            loss_g = loss_adv + args.lambda_l1_context * loss_l1_context
            if HOLE_L1_WEIGHT > 0:
                loss_g = loss_g + HOLE_L1_WEIGHT * masked_l1(fake, real, hole_mask)
            loss_g.backward()
            opt_g.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}  D={loss_d.item():.3f}  G_adv={loss_adv.item():.3f}  G_l1_ctx={loss_l1_context.item():.3f}")
                if wandb_run is not None:
                    # D's own real/fake scores (post-sigmoid) are the usual
                    # first thing to check for GAN pathologies -- both
                    # pinned near 0.5 means D can no longer tell real from
                    # fake apart (G "winning", or D too weak); both pinned
                    # near their target (real~1, fake~0) with G_adv stuck
                    # high means the opposite (D "winning", G stalled).
                    wandb_run.log(
                        {
                            "loss/d": loss_d.item(),
                            "loss/g_adv": loss_adv.item(),
                            "loss/g_l1_context": loss_l1_context.item(),
                            "loss/g_total": loss_g.item(),
                            "d/real_score": d_real.sigmoid().mean().item(),
                            "d/fake_score": d_fake.sigmoid().mean().item(),
                            "epoch": epoch,
                        },
                        step=step,
                    )
            step += 1

        torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch}, out_dir / "checkpoint.pt")

        val_l1, val_adv, val_batch = validate(G, D, val_loader, device, bce)
        print(f"epoch {epoch} done -- val_l1_context={val_l1:.4f}  val_g_adv={val_adv:.4f}")
        if wandb_run is not None:
            wandb_run.log({"val/loss_l1_context": val_l1, "val/loss_g_adv": val_adv}, step=step)

        if val_batch is not None:
            sample_grid = save_sample_grid(*val_batch, out_dir / f"sample_epoch{epoch:03d}.png")
            if wandb_run is not None:
                import wandb

                wandb_run.log({"samples": wandb.Image(sample_grid)}, step=step)

        # Early stopping/checkpoint selection on val_l1_context specifically
        if val_l1 < best_val_l1 - args.min_delta:
            best_val_l1 = val_l1
            epochs_without_improvement = 0
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "val_l1_context": val_l1}, out_dir / "checkpoint_best.pt")
            print(f"  new best val_l1_context={val_l1:.4f} -- saved checkpoint_best.pt")
        else:
            epochs_without_improvement += 1
            if args.early_stopping and epochs_without_improvement >= args.patience:
                print(
                    f"early stopping: no val_l1_context improvement for {args.patience} epochs "
                    f"(best={best_val_l1:.4f}) -- stopping at epoch {epoch}"
                )
                break

    if wandb_run is not None:
        wandb_run.finish()


def validate(G: nn.Module, D: nn.Module, val_loader: DataLoader, device: str, bce: nn.Module):
    G.eval()
    total_l1, total_adv, n_batches = 0.0, 0.0, 0
    first_batch = None
    with torch.no_grad():
        for context_input, real, hole_mask, rects in val_loader:
            context_input, real, hole_mask = context_input.to(device), real.to(device), hole_mask.to(device)
            fake = G(context_input)
            loss_l1_context = masked_l1(fake, real, 1.0 - hole_mask)

            real_crop = crop_and_resize(real, rects)
            fake_crop = crop_and_resize(fake, rects)
            d_fake = D(fake_crop)
            loss_adv = bce(d_fake, torch.ones_like(d_fake))

            total_l1 += loss_l1_context.item()
            total_adv += loss_adv.item()
            n_batches += 1
            if first_batch is None:
                first_batch = (context_input, fake, real)
    G.train()
    return total_l1 / max(n_batches, 1), total_adv / max(n_batches, 1), first_batch


def save_sample_grid(context_input: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, path: Path, n: int = 4) -> Image.Image:
    def to_img(t: torch.Tensor) -> np.ndarray:
        arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        return np.transpose(arr, (1, 2, 0))

    n = min(n, context_input.shape[0])
    rows = []
    for i in range(n):
        rows.append(np.concatenate([to_img(context_input[i, :3]), to_img(fake[i]), to_img(real[i])], axis=1))
    grid = np.concatenate(rows, axis=0)
    image = Image.fromarray(grid)
    image.save(path)
    return image


def generate(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    G = UNetGenerator().to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    dataset = PairedImageMaskDataset(args.images_dir, args.masks_dir, image_size=args.image_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            fake = G(sample.context_input.unsqueeze(0).to(device))[0]
            fake_img = ((fake.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
            Image.fromarray(np.transpose(fake_img, (1, 2, 0))).save(out_dir / dataset.filenames[idx])
    print(f"wrote {len(dataset)} generated images to {out_dir}")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--out-dir", default="gan_runs/run1")
    parser.add_argument("--checkpoint", default=None, help="required for --mode generate")
    parser.add_argument("--resume", default=None, help="--mode train: continue training (G, D, and epoch count) from this checkpoint instead of from scratch")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1-context", type=float, default=10.0, help="weight on the context (outside-hole) L1 term")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="fraction of paired samples held out for validation")
    parser.add_argument("--val-seed", type=int, default=0, help="seed for the deterministic train/val split")
    parser.add_argument("--patience", type=int, default=10, help="epochs without val_l1_context improvement before stopping early")
    parser.add_argument("--min-delta", type=float, default=0.0, help="minimum val_l1_context decrease to count as an improvement")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="disable early stopping (still tracks/saves checkpoint_best.pt)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="log losses/D-scores/sample grids to Weights & Biases")
    parser.add_argument("--wandb-project", default="mitosim-gan-context-encoder")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.mode == "generate":
        if not args.checkpoint:
            raise SystemExit("--mode generate requires --checkpoint")
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

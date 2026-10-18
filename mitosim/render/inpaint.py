"""

Util functions for the inpainting.


"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Root directory holding `context_images/`/`context_masks/` and the GAN checkpoint. Overridable via `MITOSIM_WEIGHTS_DIR` 
# so this works for anyone regenerating the weights elsewhere, but defaults to the copy checked into this repo's (gitignored)
#  `weights/` directory.
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("MITOSIM_WEIGHTS_DIR", Path(__file__).resolve().parents[2] / "weights"))


def context_images_dir(weights_dir: Path | str | None = None) -> Path:
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    return weights_dir / "context_images"


def list_context_images(weights_dir: Path | str | None = None) -> list[str]:
    """Sorted filenames of every available context image (real H&E crops the GAN was trained to condition on) """
    return sorted(p.name for p in context_images_dir(weights_dir).glob("*.png"))


def load_context_image(name: str, weights_dir: Path | str | None = None) -> Image.Image:
    path = context_images_dir(weights_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"no such context image {name!r} in {context_images_dir(weights_dir)}")
    return Image.open(path).convert("RGB")


def context_masks_dir(weights_dir: Path | str | None = None) -> Path:
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    return weights_dir / "context_masks"


def load_context_mask(name: str, weights_dir: Path | str | None = None) -> Image.Image:
    """A context image's own real chromatin mask (same filename, sibling directory), needed whenever that image is picked to 
    condition  generation on a different mask's shape, so the picked image's own pre-existing structure can also be erased."""
    path = context_masks_dir(weights_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"no such context mask {name!r} in {context_masks_dir(weights_dir)}")
    return Image.open(path).convert("L")


def pick_context_image(rng: np.random.Generator, weights_dir: Path | str | None = None) -> tuple[str, Image.Image]:
    """Pick one context image at random """
    names = list_context_images(weights_dir)
    if not names:
        raise FileNotFoundError(f"no context images found in {context_images_dir(weights_dir)}")
    name = names[rng.integers(0, len(names))]
    return name, load_context_image(name, weights_dir)


@dataclass(frozen=True)
class EllipseFit:
    """An ellipse fit to a mask's own principal axes."""

    centroid_x: float
    centroid_y: float
    extent_major: float  # semi-major axis length, px
    extent_minor: float  # semi-minor axis length, px
    angle_rad: float  # major-axis angle from +x, in (-pi/2, pi/2]


def fit_ellipse(mask: np.ndarray) -> EllipseFit | None:
    """Centroid + covariance eigendecomposition of `mask`'s foreground pixels -- the same math `score/descriptors.py`'s `hollowness` uses,
    but returning the enclosing ellipse's own parameters (for context-image size/orientation matching, hole shaping, etc.) rather
    than a derived scalar.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None

    coords = np.stack([xs, ys], axis=-1).astype(float)
    centroid = coords.mean(axis=0)
    cov = np.cov((coords - centroid).T)
    evals, evecs = np.linalg.eigh(cov)  # ascending order
    evals = np.maximum(evals, 1e-9)

    whitened = (coords - centroid) @ evecs / np.sqrt(evals)
    r_max = np.linalg.norm(whitened, axis=-1).max()
    extent_minor, extent_major = r_max * np.sqrt(evals)  # evals ascending

    return EllipseFit(
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        extent_major=float(extent_major),
        extent_minor=float(extent_minor),
        angle_rad=float(np.arctan2(evecs[1, -1], evecs[0, -1])),
    )


def rasterize_ellipse(shape: tuple[int, int], fit: EllipseFit, scale: float = 1.0) -> np.ndarray:
    """Boolean (H, W) array: True inside the ellipse described by `fit`, optionally grown by `scale`."""
    yy, xx = np.indices(shape)
    dx = xx - fit.centroid_x
    dy = yy - fit.centroid_y
    c, s = np.cos(fit.angle_rad), np.sin(fit.angle_rad)
    u = (dx * c + dy * s) / (scale * max(fit.extent_major, 1e-6))
    v = (-dx * s + dy * c) / (scale * max(fit.extent_minor, 1e-6))
    return (u**2 + v**2) <= 1.0

"""Build a lookup database of (size, orientation) for every context mask in
`weights/context_masks/`, so a synthetic chromatin mask can later be paired
with a context crop whose *real* mitotic figure was a similar size and
orientation -- rather than a context image picked at random, which can
mismatch badly (a small round metaphase mask conditioned against a context
photo whose own real figure was a large elongated anaphase bridge, etc.).

Each context mask is fit to an enclosing ellipse (`render/inpaint.py`'s
`fit_ellipse` -- centroid + covariance eigendecomposition, scaled to
actually contain the mask's farthest foreground pixel). The database is a
flat CSV, one row per context image, with the ellipse's semi-major/minor
extents (px) and orientation angle (deg): everything a later matching step
needs to compute a size+orientation distance to a target mask, including
against a candidate rotated by 0/90/180/270 degrees (cheap, exact,
distortion-free -- no arbitrary-angle resampling needed to expand
orientation coverage of a fixed image library).

This script only builds the database -- `mitosim/render/match_context.py`
is what actually reads it for the matching/retrieval step.

Run directly: `python -m mitosim.render.build_context_ellipse_db --help`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from mitosim.render.inpaint import DEFAULT_WEIGHTS_DIR, context_images_dir, fit_ellipse


def context_masks_dir(weights_dir: Path | str | None = None) -> Path:
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    return weights_dir / "context_masks"


def build_ellipse_db(weights_dir: Path | str | None = None) -> list[dict]:
    """Fit an ellipse to every PNG in `context_masks_dir(weights_dir)` and
    return one row per file: filename, centroid, extents, and angle (both
    radians and degrees -- degrees for a human skimming the CSV, radians
    kept alongside since that's what every consumer in this codebase
    actually computes with). Masks with fewer than 2 foreground pixels
    (`fit_ellipse` returns None -- see its docstring) are skipped, not
    zero-filled, so they can't silently be matched against as if they were
    a real (if tiny) object.
    """
    masks_dir = context_masks_dir(weights_dir)
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"no context masks directory at {masks_dir} -- see README's inpainting section")

    rows: list[dict] = []
    skipped: list[str] = []
    for path in sorted(masks_dir.glob("*.png")):
        mask = np.array(Image.open(path).convert("L")) > 0
        fit = fit_ellipse(mask)
        if fit is None:
            skipped.append(path.name)
            continue
        rows.append(
            {
                "filename": path.name,
                "centroid_x": round(fit.centroid_x, 2),
                "centroid_y": round(fit.centroid_y, 2),
                "extent_major_px": round(fit.extent_major, 3),
                "extent_minor_px": round(fit.extent_minor, 3),
                "angle_rad": round(fit.angle_rad, 5),
                "angle_deg": round(np.degrees(fit.angle_rad), 2),
            }
        )

    if skipped:
        print(f"  skipped {len(skipped)} mask(s) with fewer than 2 foreground pixels: {skipped[:10]}" + (" ..." if len(skipped) > 10 else ""))
    return rows


def write_ellipse_db(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", default=None, help="default: MitoSim's weights/ (or $MITOSIM_WEIGHTS_DIR)")
    parser.add_argument("--out", default=None, help="default: <weights-dir>/context_ellipse_db.csv")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir) if args.weights_dir else DEFAULT_WEIGHTS_DIR
    out_path = Path(args.out) if args.out else weights_dir / "context_ellipse_db.csv"

    print(f"Fitting ellipses for every mask in {context_masks_dir(weights_dir)} -> {out_path}")
    rows = build_ellipse_db(weights_dir)
    write_ellipse_db(rows, out_path)

    n_context_images = len(list(context_images_dir(weights_dir).glob("*.png")))
    print(f"\nWrote {len(rows)} rows to {out_path} ({n_context_images} context images available)")
    if rows:
        majors = np.array([r["extent_major_px"] for r in rows])
        minors = np.array([r["extent_minor_px"] for r in rows])
        print(f"extent_major_px: min={majors.min():.1f} median={np.median(majors):.1f} max={majors.max():.1f}")
        print(f"extent_minor_px: min={minors.min():.1f} median={np.median(minors):.1f} max={minors.max():.1f}")


if __name__ == "__main__":
    main()

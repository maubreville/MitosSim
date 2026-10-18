"""Forward optical model: 

voxelised chromatin density -> Beer-Lambert optical density -> defocus-dependent blur -> thresholded binary chromatin mask

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from mitosim.core.types import CellState
from mitosim.render.voxelize import Grid, grid_for_bounds, voxelize_chromatids


def abbe_resolution_um(numerical_aperture: float, wavelength_um: float = 0.55) -> float:
    """Diffraction-limited resolvable distance (Abbe criterion)."""
    return 0.61 * wavelength_um / numerical_aperture


def fwhm_to_gaussian_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


#: 40x/0.75NA objective, green-ish effective wavelength.
_DEFAULT_PSF_SIGMA_UM = fwhm_to_gaussian_sigma(abbe_resolution_um(0.75))


@dataclass(frozen=True)
class OpticsParams:
    voxel_size_um: float = 0.1            # rendering voxel size; configurable for a convergence study
    extinction_chromatin: float = 0.6      # per micrometre of chromatin path length (placeholder, pending V4)
    focal_offset_um: float = 0.0           # focal plane position, relative to slab centre s
    depth_of_field_um: float = 1.0         # defocus growth scale away from the focal plane
    psf_sigma_um: float = _DEFAULT_PSF_SIGMA_UM
    pixel_size_um: float = 0.25            # target scanning resolution
    patch_size_px: int = 128
    # Chromatin Beer-Lambert absorption fraction above which a pixel counts as chromatin 
    mask_threshold: float = 0.2


def _slab_grid(state_cell_radius: float, s: float, t: float, voxel_size_um: float, margin_um: float = 3.0) -> Grid:
    half_extent = state_cell_radius + margin_um
    lo = np.array([-half_extent, -half_extent, s - t / 2])
    hi = np.array([half_extent, half_extent, s + t / 2])
    return grid_for_bounds(lo, hi, spacing=voxel_size_um)


def defocus_blurred_path_length(density3d: np.ndarray, grid: Grid, focal_z: float, dof_um: float, psf_sigma_um: float) -> np.ndarray:
    """Integrate density along z into a path-length map (micrometres), blurring each z-slice by a Gaussian 
    whose width grows with distance from the focal plane -- the depth-of-field term. This is an approximation 
    (defocus blur is applied to density before the Beer-Lambert exponential, not to transmitted light).
    """
    z_coords = grid.z_coords()
    voxel = grid.spacing
    sigma_px_base = psf_sigma_um / voxel
    column = np.zeros(density3d.shape[:2], dtype=float)
    for i, z in enumerate(z_coords):
        defocus = np.sqrt(1.0 + ((z - focal_z) / dof_um) ** 2)
        sigma_px = sigma_px_base * defocus
        blurred = gaussian_filter(density3d[:, :, i].astype(float), sigma=sigma_px, mode="constant", cval=0.0)
        column += blurred * voxel
    return column


def beer_lambert_transmittance(path_length_um: np.ndarray, extinction_coefficient: float) -> np.ndarray:
    return np.exp(-extinction_coefficient * np.maximum(path_length_um, 0.0))


def resample_and_crop(
    image: np.ndarray, voxel_size_um: float, pixel_size_um: float, patch_size_px: int, fill_value, order: int = 1
) -> np.ndarray:
    """Resample from the fine voxelisation grid to the scanning resolution then centre-crop/pad to a fixed patch
    size. `order=0` (nearest-neighbour) is for the binary chromatin mask below -- bilinear interpolation would reintroduce
    fractional values at the boundary, which defeats the point of a hard mask."""
    factor = voxel_size_um / pixel_size_um
    zoom_factors = (factor, factor, 1) if image.ndim == 3 else (factor, factor)
    resampled = zoom(image, zoom_factors, order=order)

    out_shape = (patch_size_px, patch_size_px) + image.shape[2:]
    out = np.empty(out_shape, dtype=resampled.dtype)
    out[..., :] = fill_value

    h, w = resampled.shape[:2]

    def _paste_ranges(size_out: int, size_in: int) -> tuple[slice, slice]:
        if size_in >= size_out:
            start_in = (size_in - size_out) // 2
            return slice(0, size_out), slice(start_in, start_in + size_out)
        start_out = (size_out - size_in) // 2
        return slice(start_out, start_out + size_in), slice(0, size_in)

    out_y, in_y = _paste_ranges(patch_size_px, h)
    out_x, in_x = _paste_ranges(patch_size_px, w)
    out[out_y, out_x, ...] = resampled[in_y, in_x, ...]
    return out


def render_chromatin_mask_from_alpha(chromatin_alpha: np.ndarray, params: OpticsParams = OpticsParams()) -> np.ndarray:
    """2D Beer-Lambert absorption fraction -> 2D binary mask (uint8, values in {0, 1}) at the patch's final resolution/crop: 
    wherever light attenuation through the chromatin crosses `params.mask_threshold`.
    """
    projected = (chromatin_alpha >= params.mask_threshold).astype(np.uint8)
    return resample_and_crop(projected, params.voxel_size_um, params.pixel_size_um, params.patch_size_px, fill_value=0, order=0)


def render_chromatin_mask(state: CellState, s: float, t: float, params: OpticsParams = OpticsParams()) -> np.ndarray:
    """CellState + slab -> binary chromatin mask (uint8, {0, 1}) at the patch's final resolution/crop."""
    grid = _slab_grid(state.cell_radius, s, t, params.voxel_size_um)
    focal_z = s + params.focal_offset_um

    chromatin_density = voxelize_chromatids(state.chromatids, grid)
    chromatin_path = defocus_blurred_path_length(chromatin_density, grid, focal_z, params.depth_of_field_um, params.psf_sigma_um)
    chromatin_alpha = 1.0 - beer_lambert_transmittance(chromatin_path, params.extinction_chromatin)

    return render_chromatin_mask_from_alpha(chromatin_alpha, params)

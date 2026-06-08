from __future__ import annotations

import numpy as np


def restore_wavelet_rgb_dtype(
        denoised_rgb: np.ndarray,
        original_rgb: np.ndarray,
) -> np.ndarray:
    """Restore skimage wavelet RGB output to the source RGB dtype and range."""
    normalized_rgb = np.asarray(denoised_rgb, dtype=np.float64)
    if not np.isfinite(normalized_rgb).all():
        normalized_original = _normalize_rgb_to_unit_float(original_rgb)
        normalized_rgb = np.where(
                np.isfinite(normalized_rgb),
                normalized_rgb,
                normalized_original,
        )

    original_dtype = original_rgb.dtype
    if np.issubdtype(original_dtype, np.integer):
        dtype_info = np.iinfo(original_dtype)
        scaled = np.rint(np.clip(normalized_rgb, 0.0, 1.0) * dtype_info.max)
        return scaled.astype(original_dtype)

    if np.issubdtype(original_dtype, np.floating):
        return np.clip(normalized_rgb, 0.0, 1.0).astype(original_dtype, copy=False)

    return np.asarray(denoised_rgb).astype(original_dtype, copy=False)


def _normalize_rgb_to_unit_float(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB data to the normalized range returned by denoise_wavelet."""
    rgb_float = rgb.astype(np.float64, copy=False)
    if np.issubdtype(rgb.dtype, np.integer):
        return rgb_float / np.iinfo(rgb.dtype).max
    return np.clip(rgb_float, 0.0, 1.0)

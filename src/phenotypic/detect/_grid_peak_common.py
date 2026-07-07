"""Shared helpers for grid-peak detector preprocessing."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from skimage import morphology

from phenotypic.detect._thresholding_registry import ThresholdingRegistry


def _round_odd(n: int) -> int:
    """Round to the next odd integer with a minimum of 3."""
    n = max(n, 3)
    return n if n % 2 == 1 else n + 1


def grid_peak_background_kernel(
    matrix_shape: tuple[int, int],
    *,
    footprint_width: int,
    nrows: int | None,
    ncols: int | None,
    round_odd: Callable[[int], int] = _round_odd,
) -> np.ndarray:
    """Return the background-subtraction footprint for grid-peak detectors."""
    if nrows is not None:
        bg_h = round_odd(round((matrix_shape[0] / nrows) * 1.5))
        bg_w = round_odd(round((matrix_shape[1] / (ncols or nrows)) * 1.5))
        return morphology.footprint_rectangle((bg_h, bg_w))

    dim = round_odd(footprint_width * 2)
    return morphology.footprint_rectangle((dim, dim))


def grid_peak_threshold_mask(
    matrix: np.ndarray,
    *,
    thresh_method: str,
    subtract_background: bool,
    footprint_width: int,
    nrows: int | None,
    ncols: int | None,
    round_odd: Callable[[int], int] = _round_odd,
) -> np.ndarray:
    """Threshold a grid-peak detection matrix without mutating the input."""
    kernel = grid_peak_background_kernel(
        matrix.shape,
        footprint_width=footprint_width,
        nrows=nrows,
        ncols=ncols,
        round_odd=round_odd,
    )
    enh_matrix = matrix.copy()
    if subtract_background:
        enh_matrix = morphology.white_tophat(enh_matrix, kernel)

    return ThresholdingRegistry.threshold_mask(
        enh_matrix,
        method=thresh_method,
        local_block_size=max(footprint_width * 2 + 1, 3),
        allowed_methods=ThresholdingRegistry.GRID_METHODS,
    )

"""Unit tests for shared grid-peak detector helpers."""
from __future__ import annotations

import numpy as np
from skimage import filters

from phenotypic.detect._grid_peak_common import (
    grid_peak_background_kernel,
    grid_peak_threshold_mask,
)


def test_background_kernel_plain_image_uses_footprint_width() -> None:
    """Plain images use the odd fallback size derived from footprint width."""
    kernel = grid_peak_background_kernel(
        (20, 30),
        footprint_width=6,
        nrows=None,
        ncols=None,
    )

    assert kernel.shape == (13, 13)


def test_background_kernel_grid_image_uses_adaptive_spacing() -> None:
    """Grid images size the kernel to 1.5x row/column spacing."""
    kernel = grid_peak_background_kernel(
        (100, 200),
        footprint_width=6,
        nrows=10,
        ncols=20,
    )

    assert kernel.shape == (15, 15)


def test_background_kernel_uses_nrows_when_ncols_missing() -> None:
    """The historical grid fallback uses nrows for both axes when ncols is None."""
    kernel = grid_peak_background_kernel(
        (100, 200),
        footprint_width=6,
        nrows=10,
        ncols=None,
    )

    assert kernel.shape == (15, 31)


def test_background_kernel_accepts_custom_round_odd_hook() -> None:
    """Callers can preserve detector-level odd-rounding overrides."""
    calls: list[int] = []

    def custom_round_odd(value: int) -> int:
        calls.append(value)
        return 9

    kernel = grid_peak_background_kernel(
        (20, 30),
        footprint_width=2,
        nrows=None,
        ncols=None,
        round_odd=custom_round_odd,
    )

    assert calls == [4]
    assert kernel.shape == (9, 9)


def test_threshold_mask_without_background_matches_registry_threshold() -> None:
    """Without background subtraction, mean thresholding preserves old semantics."""
    matrix = np.array([[0.0, 0.5], [1.0, 2.0]])
    expected = matrix >= filters.threshold_mean(matrix)

    observed = grid_peak_threshold_mask(
        matrix,
        thresh_method="mean",
        subtract_background=False,
        footprint_width=6,
        nrows=None,
        ncols=None,
    )

    np.testing.assert_array_equal(observed, expected)


def test_local_threshold_uses_footprint_derived_block_size() -> None:
    """Local thresholding uses ``footprint_width * 2 + 1`` above the minimum."""
    matrix = np.arange(225, dtype=float).reshape(15, 15)
    expected = matrix >= filters.threshold_local(matrix, block_size=13)

    observed = grid_peak_threshold_mask(
        matrix,
        thresh_method="local",
        subtract_background=False,
        footprint_width=6,
        nrows=None,
        ncols=None,
    )

    np.testing.assert_array_equal(observed, expected)


def test_threshold_mask_does_not_modify_input() -> None:
    """The helper works on a copy, matching the detector-local implementation."""
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    before = matrix.copy()

    grid_peak_threshold_mask(
        matrix,
        thresh_method="local",
        subtract_background=True,
        footprint_width=1,
        nrows=None,
        ncols=None,
    )

    np.testing.assert_array_equal(matrix, before)

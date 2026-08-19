"""Pyramid geometry, checked against the committed logic-validation script.

The script under docs/superpowers/logic_validation_scripts/ is the reference
implementation for level counts and level shapes; it depends only on numpy and
has already refuted a floor-based formula. These tests assert the shipped
helpers agree with it, so the two can never drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phenotypic.sdk_ import ngff_

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "superpowers"
    / "logic_validation_scripts"
    / "2026-08-18-ome-zarr-image-store"
    / "ngff_store_geometry.py"
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("ngff_store_geometry", _SCRIPT)
    assert spec is not None and spec.loader is not None, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference()

PLATES = [(2048, 2048), (4000, 3000), (6000, 4000), (512, 512), (300, 200), (513, 100)]


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_count_matches_reference(height: int, width: int) -> None:
    assert ngff_.pyramid_level_count(height, width) == REFERENCE.level_count(
        height, width
    )


def test_level_count_uses_ceil_not_floor() -> None:
    """floor(log2(4000/512)) + 1 == 3, which stops one level early at 1000x750."""
    assert ngff_.pyramid_level_count(4000, 3000) == 4


def test_single_level_at_or_below_stop_px() -> None:
    assert ngff_.pyramid_level_count(512, 512) == 1
    assert ngff_.pyramid_level_count(100, 100) == 1


@pytest.mark.parametrize(("height", "width"), PLATES)
def test_level_shapes_match_reference(height: int, width: int) -> None:
    levels = ngff_.pyramid_level_count(height, width)
    shapes = ngff_.pyramid_level_shapes((height, width), levels)
    assert [tuple(s) for s in shapes] == [
        tuple(s) for s in REFERENCE.level_shapes(height, width)
    ]


def test_level_shapes_ceil_halve_odd_extents() -> None:
    assert ngff_.pyramid_level_shapes((1025, 7), 3) == ((1025, 7), (513, 4), (257, 2))


def test_level_shapes_leave_channel_axis_alone() -> None:
    assert ngff_.pyramid_level_shapes((3, 1025, 7), 2) == ((3, 1025, 7), (3, 513, 4))


def test_scale_vector_comes_from_actual_shapes_not_powers_of_two() -> None:
    """1025 -> 513 is a ratio of 1025/513, which is NOT 2.0."""
    scale = ngff_.level_scale_vector((1025, 7), (513, 4))
    assert scale == pytest.approx([1025 / 513, 7 / 4])
    assert scale[0] != pytest.approx(2.0)


def test_scale_vector_pins_channel_axis_to_one() -> None:
    assert ngff_.level_scale_vector((3, 1024, 1024), (3, 512, 512)) == pytest.approx(
        [1.0, 2.0, 2.0]
    )


def test_label_downsample_invents_no_new_values() -> None:
    rng = np.random.default_rng(20260818)
    labels = rng.choice(np.array([0, 3, 7, 11, 40], dtype=np.uint16), size=(64, 64))
    small = ngff_.downsample_label(labels)
    assert set(np.unique(small)).issubset(set(np.unique(labels)))
    assert small.shape == (32, 32)
    assert small.dtype == labels.dtype


def test_mean_downsample_would_invent_values() -> None:
    """Guards C5: proves the rejected method really is wrong, not merely unchosen."""
    labels = np.array([[0, 40], [40, 40]], dtype=np.uint16)
    meaned = ngff_.downsample_image(labels)
    assert set(np.unique(meaned)) - set(np.unique(labels))


def test_image_downsample_odd_extent_uses_edge_pad_not_zero_pad() -> None:
    array = np.full((3, 3), 100, dtype=np.uint8)
    small = ngff_.downsample_image(array)
    assert small.shape == (2, 2)
    assert (small == 100).all(), "a zero pad would darken the trailing row/column"


def test_image_downsample_preserves_dtype() -> None:
    array = np.arange(16, dtype=np.uint16).reshape(4, 4)
    assert ngff_.downsample_image(array).dtype == np.uint16
    assert ngff_.downsample_image(array.astype(np.float64)).dtype == np.float64


def test_build_pyramid_shapes_and_count() -> None:
    array = np.zeros((1025, 7), dtype=np.uint16)
    levels = ngff_.build_pyramid(array, 3, kind="label")
    assert [lvl.shape for lvl in levels] == [(1025, 7), (513, 4), (257, 2)]


def test_build_pyramid_channel_first_rgb() -> None:
    array = np.zeros((3, 1024, 1024), dtype=np.uint8)
    levels = ngff_.build_pyramid(array, 2, kind="image")
    assert [lvl.shape for lvl in levels] == [(3, 1024, 1024), (3, 512, 512)]


def test_axes_for_series() -> None:
    assert ngff_.axes_for("rgb") == ("c", "y", "x")
    assert ngff_.axes_for("gray") == ("y", "x")
    assert ngff_.axes_for("detect_mat") == ("y", "x")
    assert ngff_.axes_for("objmap") == ("y", "x")

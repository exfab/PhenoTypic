"""Unit tests for the shared per-object geometry helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic.measure._object_geometry import convex_hull_area, object_edt


def _pixel_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(float)


def test_convex_hull_area_is_the_hull_volume_not_its_perimeter():
    """A filled 10x20 pixel block has hull vertices at pixel centres, so the hull is a
    9 x 19 rectangle: area 171, perimeter 56. Qhull on integer coordinates is exact to
    a few ulp, so 1e-9 is ~1e5 ulp of headroom and still 115 below the perimeter.

    Mutation: return `hull.area` instead of `hull.volume` -> 56.0, and this fails.
    """
    hull, area = convex_hull_area(_pixel_coords(np.ones((10, 20), dtype=bool)))
    assert hull is not None
    assert area == pytest.approx(171.0, abs=1e-9)


@pytest.mark.parametrize(
    "mask",
    [np.ones((1, 1), dtype=bool), np.ones((1, 5), dtype=bool)],
    ids=["single-pixel", "collinear-line"],
)
def test_convex_hull_area_is_nan_when_qhull_cannot_build_a_hull(mask):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hull, area = convex_hull_area(_pixel_coords(mask))
    assert hull is None
    assert np.isnan(area)


def test_object_edt_sees_background_on_every_side_of_the_crop():
    """A 3x3 crop fully set: without padding every pixel would be at infinite/undefined
    distance; with one pixel of padding the centre is 2 px from background.
    Integer geometry, exact.

    Mutation: drop the `np.pad` -> the centre is no longer 2.0, and this fails.
    """
    edt = object_edt(np.ones((3, 3), dtype=bool))
    assert edt.shape == (3, 3)
    assert edt[1, 1] == 2.0
    assert edt[0, 0] == 1.0


def test_object_edt_of_a_41x20_rectangle_has_inscribed_radius_10():
    edt = object_edt(np.ones((41, 20), dtype=bool))
    assert edt.max() == 10.0

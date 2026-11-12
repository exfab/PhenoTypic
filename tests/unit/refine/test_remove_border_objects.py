"""``RemoveBorderObjects`` removes exactly the objects inside the edge band.

The band of width ``e`` covers rows ``0 .. e-1`` and ``H-e .. H-1`` and the same
columns; an object is removed when any of its pixels lies in it. Each case
below places one 2 x 2 object at a known distance ``d`` from each edge (``d`` is
the index of its nearest row or column, counted from that edge), so an object
must go exactly when ``d < e``.

Three bugs this file pins, all present before 0.21.0:

- The top and left slices were ``[: e - 1]``, one row short, so at the default
  ``border_size=1`` an object touching the top or left edge was kept.
- ``border_size=None`` (documented as 1% of the smaller dimension) raised
  ``TypeError`` from ``image.shape[[1, 2]]``.
- A fractional ``border_size`` took ``np.min(image.shape)``, which is the
  channel count (3) on an RGB image; 0.05 gave a band of 0, and the bottom and
  right slices ``[-0:]`` then covered the whole image, removing every object.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.refine import RemoveBorderObjects

#: The four edges, and where an object at distance ``d`` from each one sits.
EDGES = ("top", "bottom", "left", "right")


def _square_at(edge: str, distance: int, height: int, width: int, offset: int):
    """Row and column slices of a 2 x 2 object *distance* px from *edge*.

    *offset* moves the object along the edge so that objects never overlap
    and never come within the band of a perpendicular edge.
    """
    if edge == "top":
        return slice(distance, distance + 2), slice(offset, offset + 2)
    if edge == "bottom":
        nearest = height - 1 - distance
        return slice(nearest - 1, nearest + 1), slice(offset, offset + 2)
    if edge == "left":
        return slice(offset, offset + 2), slice(distance, distance + 2)
    nearest = width - 1 - distance
    return slice(offset, offset + 2), slice(nearest - 1, nearest + 1)


def _plate(height: int, width: int, distances, rgb: bool):
    """An image whose object map holds one object per (edge, distance).

    Returns:
        The image and ``{(edge, distance): label}``.
    """
    objmap = np.zeros((height, width), dtype=np.uint16)
    labels = {}
    label = 1
    for edge in EDGES:
        for index, distance in enumerate(distances):
            rows, cols = _square_at(edge, distance, height, width, 40 + 10 * index)
            assert not objmap[rows, cols].any(), "objects must not overlap"
            objmap[rows, cols] = label
            labels[(edge, distance)] = label
            label += 1
    pixels = (
        np.full((height, width, 3), 30, dtype=np.uint8)
        if rgb
        else np.full((height, width), 0.1, dtype=np.float32)
    )
    image = Image(pixels)
    image.objmap[:] = objmap
    return image, labels


def _surviving(image: Image, labels) -> set:
    present = set(np.unique(image.objmap[:]).tolist()) - {0}
    return {key for key, label in labels.items() if label in present}


@pytest.mark.parametrize("border_size", [1, 2, 3])
@pytest.mark.parametrize("rgb", [False, True], ids=["gray", "rgb"])
def test_an_integer_band_removes_objects_closer_than_it(border_size: int, rgb: bool) -> None:
    distances = (0, 1, 2, 3)
    image, labels = _plate(120, 160, distances, rgb)

    RemoveBorderObjects(border_size=border_size).apply(image, inplace=True)

    expected = {(edge, d) for edge in EDGES for d in distances if d >= border_size}
    assert _surviving(image, labels) == expected


def test_the_default_band_removes_objects_touching_every_edge() -> None:
    image, labels = _plate(120, 160, (0, 1), rgb=True)

    RemoveBorderObjects().apply(image, inplace=True)

    assert _surviving(image, labels) == {(edge, 1) for edge in EDGES}


def test_none_is_one_percent_of_the_smaller_dimension() -> None:
    # min(300, 400) * 0.01 = 3 px.
    distances = (2, 3)
    image, labels = _plate(300, 400, distances, rgb=True)

    RemoveBorderObjects(border_size=None).apply(image, inplace=True)

    assert _surviving(image, labels) == {(edge, 3) for edge in EDGES}


def test_a_fraction_is_of_the_smaller_spatial_dimension_not_the_channels() -> None:
    # min(120, 160) * 0.05 = 6 px, on an RGB image whose shape includes 3.
    distances = (5, 6)
    image, labels = _plate(120, 160, distances, rgb=True)

    RemoveBorderObjects(border_size=0.05).apply(image, inplace=True)

    assert _surviving(image, labels) == {(edge, 6) for edge in EDGES}


def test_a_fraction_too_small_for_one_pixel_still_removes_edge_objects() -> None:
    # min(120, 160) * 0.001 = 0.12 px: the band is at least 1 px wide.
    image, labels = _plate(120, 160, (0, 1), rgb=True)

    RemoveBorderObjects(border_size=0.001).apply(image, inplace=True)

    assert _surviving(image, labels) == {(edge, 1) for edge in EDGES}


def test_a_whole_float_is_a_pixel_count() -> None:
    image, labels = _plate(120, 160, (1, 2), rgb=False)

    RemoveBorderObjects(border_size=2.0).apply(image, inplace=True)

    assert _surviving(image, labels) == {(edge, 2) for edge in EDGES}


def test_a_zero_band_removes_nothing() -> None:
    image, labels = _plate(120, 160, (0, 1), rgb=True)

    RemoveBorderObjects(border_size=0).apply(image, inplace=True)

    assert _surviving(image, labels) == set(labels)


@pytest.mark.parametrize("border_size", [-1, -0.5, float("nan"), float("inf")])
def test_an_unusable_band_is_refused_at_construction(border_size) -> None:
    """Not at apply time, as ``int(nan)`` / ``int(inf)`` would raise then."""
    with pytest.raises(ValueError, match="border_size"):
        RemoveBorderObjects(border_size=border_size)

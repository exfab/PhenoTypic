"""Measured columns scale with image resolution exactly as their units say.

Each scene is drawn at radius 60 and again at radius 120 (k = 2): the image is
10 radii square and every disk centre sits at the same fraction of it, so
positions, sizes and gaps all scale by k. One centred disk is measured for
size, shape, intensity and bounding-box positions; a pair of disks in a
two-cell grid, side by side and stacked, is measured for neighbour distances.

A column whose unit has length power p must grow by k**p: lengths, positions
and distances double, areas and integrated intensities quadruple, and
scale-free values (intensities, ratios, angles) hold. A failure names a member
whose unit tag is wrong; fix the tag, never the tolerance (plan 02, Task 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phenotypic as pht
from phenotypic.detect import OtsuDetector
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import (
    MeasureBounds,
    MeasureIntensity,
    MeasureNeighborDist,
    MeasureShape,
    MeasureSize,
)
from phenotypic.schema import BBOX, INTENSITY, NEIGHBOR_DIST, OBJECT, SHAPE, SIZE

K = 2
SMALL_RADIUS = 60

# A scene: disk centres in radii from the top-left corner, and the grid it is
# cut into as (nrows, ncols), or None for a plain Image.
_ONE_DISK = (((5.0, 5.0),), None)
_PAIR_ACROSS = (((5.0, 2.5), (5.0, 7.5)), (1, 2))
_PAIR_DOWN = (((2.5, 5.0), (7.5, 5.0)), (2, 1))


def _scene_image(scene, radius: int) -> pht.Image:
    centres, grid = scene
    size = 10 * radius
    rows, cols = np.ogrid[:size, :size]
    disks = np.zeros((size, size), dtype=bool)
    for row, col in centres:
        # Pixel centres sit at i + 0.5, so this centre scales by exactly k.
        r0, c0 = row * radius - 0.5, col * radius - 0.5
        disks |= (rows - r0) ** 2 + (cols - c0) ** 2 <= radius**2
    arr = np.where(disks, 0.9, 0.1).astype(np.float64)
    if grid is None:
        return pht.Image(arr)
    nrows, ncols = grid
    finder = ManualGridFinder(
        row_edges=np.linspace(0, size, nrows + 1).astype(int),
        col_edges=np.linspace(0, size, ncols + 1).astype(int),
    )
    return pht.GridImage(arr, grid_finder=finder, nrows=nrows, ncols=ncols)


def _measure_scene(measurer: type, scene, radius: int) -> pd.DataFrame:
    image = OtsuDetector().apply(_scene_image(scene, radius))
    expected = len(scene[0])
    assert image.num_objects == expected, f"expected {expected} disks, found {image.num_objects}"
    return measurer().measure(image).sort_values(OBJECT.LABEL).reset_index(drop=True)


@pytest.mark.parametrize(
    "measurer, info, scene",
    [
        pytest.param(MeasureSize, SIZE, _ONE_DISK, id="SIZE"),
        pytest.param(MeasureShape, SHAPE, _ONE_DISK, id="SHAPE"),
        pytest.param(MeasureIntensity, INTENSITY, _ONE_DISK, id="INTENSITY"),
        pytest.param(MeasureBounds, BBOX, _ONE_DISK, id="BBOX"),
        pytest.param(MeasureNeighborDist, NEIGHBOR_DIST, _PAIR_ACROSS, id="NEIGHBOR_DIST-across"),
        pytest.param(MeasureNeighborDist, NEIGHBOR_DIST, _PAIR_DOWN, id="NEIGHBOR_DIST-down"),
    ],
)
def test_columns_scale_as_their_unit_says(measurer, info, scene):
    small = _measure_scene(measurer, scene, radius=SMALL_RADIUS)
    large = _measure_scene(measurer, scene, radius=K * SMALL_RADIUS)
    checked = []
    for member in info:
        p = member.unit.length_power
        column = str(member)
        if p is None or column not in small:
            continue
        a = small[column].to_numpy(dtype=float)
        b = large[column].to_numpy(dtype=float)
        # A neighbour beyond the plate edge is NaN at both scales; nothing to compare.
        assert np.array_equal(np.isnan(a), np.isnan(b)), (column, a, b)
        present = ~np.isnan(a)
        if not present.any():
            continue
        a, b = a[present], b[present]
        if p == 0:
            assert np.all(np.abs(b - a) <= 0.05 * np.maximum(1.0, np.abs(a))), (column, a, b)
        else:
            assert b / a == pytest.approx(np.full(a.shape, float(K**p)), rel=0.05), (column, a, b)
        checked.append(column)
    # A loop that skipped every member would pass while checking nothing.
    assert checked, f"no {info.__name__} column was checked"

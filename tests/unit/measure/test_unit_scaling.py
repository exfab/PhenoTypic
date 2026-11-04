"""Measured columns scale with image resolution exactly as their units say.

Each scene is drawn at radius 60 and again at radius 120 (k = 2): the image is
10 radii square and every disk centre and radius is a fixed multiple of the
scene radius, so positions, sizes and gaps all scale by k.

- One centred disk: size, shape, intensity and bounding-box positions.
- Two disks in a two-cell grid, side by side and stacked: neighbour distances.
- Two disks in one grid cell: grid spread.
- Nine small disks jittered about a 3x3 grid: grid regression statistics.
- One disk with a stepped radial profile: symmetric zones.

A column whose unit has length power p must grow by k**p: lengths, positions
and distances double, areas and integrated intensities quadruple, and
scale-free values (intensities, ratios, angles, counts) hold. A failure names
a member whose unit tag is wrong; fix the tag, never the tolerance (plan 02,
Task 6).

Not measured here, because no simple synthetic scene makes them well-defined,
and pinned only by ``tests/unit/schema/test_unit_tags_snapshot.py``:

- ``ORIENTATION_ZONE_*``: they need a textured colony with a resolvable
  branch-orientation field, and ``MeasureOrientationZones`` analyses it with
  pixel-sized parameters (``sigma_d``, ``radial_ring_width``,
  ``long_range_lag``), so its output is not scale-invariant by design.
- The ``GRID`` interval columns: no operation writes them.

Scaling cannot tell two units of the same length power apart (``RADIAN`` from
``DEGREE``); the snapshot pins those, and ``Shape_Orientation`` is also checked
against a known angle below.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import pytest

import phenotypic as pht
from phenotypic.detect import OtsuDetector
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import (
    MeasureBounds,
    MeasureGridLinRegStats,
    MeasureGridSpread,
    MeasureIntensity,
    MeasureNeighborDist,
    MeasureShape,
    MeasureSize,
    MeasureSymZones,
)
from phenotypic.schema import (
    BBOX,
    GRID_LINREG_STATS,
    GRID_SPREAD,
    INTENSITY,
    NEIGHBOR_DIST,
    OBJECT,
    SHAPE,
    SIZE,
    SYMMETRIC_ZONES,
    UNITS,
)

K = 2
SMALL_RADIUS = 60

#: Every member the test skips because it is not a quantity. Asserted equal to
#: the skipped set, so re-tagging a measured length NOT_A_QUANTITY fails here
#: instead of silently leaving the test.
_NOT_A_QUANTITY = frozenset(
    {
        NEIGHBOR_DIST.LEFT_NEIGHBOR_OBJ_LABEL,
        NEIGHBOR_DIST.RIGHT_NEIGHBOR_OBJ_LABEL,
        NEIGHBOR_DIST.ABOVE_NEIGHBOR_OBJ_LABEL,
        NEIGHBOR_DIST.UNDER_NEIGHBOR_OBJ_LABEL,
        NEIGHBOR_DIST.NEAREST_OBJ_LABEL,
        NEIGHBOR_DIST.NEAREST_RELATION,
    }
)

#: Members that are exactly 0 at both scales in their scene, so their ratio
#: says nothing. Asserted to be 0, so the exclusion cannot hide a real value.
#: CoreRadius: PELT finds no change point on the solid mask's density profile.
_ZERO_IN_ITS_SCENE = frozenset({SYMMETRIC_ZONES.CORE_RADIUS})


class _Scene(NamedTuple):
    """A scene drawn at any radius, and how many objects it holds."""

    draw: Callable[[int], pht.Image]
    objects: int


def _canvas(radius: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Row and column index grids of a 10-radius-square image."""
    size = 10 * radius
    rows, cols = np.ogrid[:size, :size]
    return rows, cols, size


def _distance(rows, cols, radius: int, row: float, col: float) -> np.ndarray:
    """Distance of every pixel from a centre given in radii from the corner.

    Pixel centres sit at i + 0.5, so a centre at ``row * radius - 0.5`` scales
    by exactly k.
    """
    return np.sqrt((rows - (row * radius - 0.5)) ** 2 + (cols - (col * radius - 0.5)) ** 2)


def _disks(
    disks: list[tuple[float, float, float]], grid: tuple[int, int] | None = None
) -> Callable[[int], pht.Image]:
    """Bright disks ``(row, col, disk radius)``, all in radii, on a dark field."""

    def draw(radius: int) -> pht.Image:
        rows, cols, size = _canvas(radius)
        inside = np.zeros((size, size), dtype=bool)
        for row, col, disk_radius in disks:
            inside |= _distance(rows, cols, radius, row, col) <= disk_radius * radius
        arr = np.where(inside, 0.9, 0.1).astype(np.float64)
        if grid is None:
            return pht.Image(arr)
        nrows, ncols = grid
        finder = ManualGridFinder(
            row_edges=np.linspace(0, size, nrows + 1).astype(int),
            col_edges=np.linspace(0, size, ncols + 1).astype(int),
        )
        return pht.GridImage(arr, grid_finder=finder, nrows=nrows, ncols=ncols)

    return draw


def _zoned_colony(radius: int) -> pht.Image:
    """One disk whose intensity steps down at 0.35 and 0.7 radii.

    Smooth steps, so the zone boundaries sit at fixed fractions of the radius
    rather than wherever rasterisation puts a hard edge.
    """
    rows, cols, _ = _canvas(radius)
    d = _distance(rows, cols, radius, 5.0, 5.0) / radius
    profile = 0.35 + 0.15 * (1 - np.tanh((d - 0.35) / 0.06)) + 0.15 * (1 - np.tanh((d - 0.7) / 0.06))
    return pht.Image(np.where(d <= 1.0, profile, 0.05).astype(np.float64))


# Offsets from the 3x3 cell centres, in radii: off-line enough that every
# regression residual is tens of pixels, so the rounded intercepts don't
# dominate it.
_JITTER = (
    ((0.1, -0.3), (-0.3, -0.3), (1.0, 0.3)),
    ((0.3, -0.3), (0.4, -0.8), (-0.9, 0.7)),
    ((-1.0, 1.0), (0.7, 0.6), (-0.9, -0.6)),
)
_CELL_CENTRES = (10 / 6, 5.0, 50 / 6)

_ONE_DISK = _Scene(_disks([(5.0, 5.0, 1.0)]), 1)
_PAIR_ACROSS = _Scene(_disks([(5.0, 2.5, 1.0), (5.0, 7.5, 1.0)], grid=(1, 2)), 2)
_PAIR_DOWN = _Scene(_disks([(2.5, 5.0, 1.0), (7.5, 5.0, 1.0)], grid=(2, 1)), 2)
_PAIR_IN_ONE_CELL = _Scene(_disks([(5.0, 2.5, 1.0), (5.0, 7.5, 1.0)], grid=(1, 1)), 2)
_JITTERED_GRID = _Scene(
    _disks(
        [
            (_CELL_CENTRES[i] + _JITTER[i][j][0], _CELL_CENTRES[j] + _JITTER[i][j][1], 0.5)
            for i in range(3)
            for j in range(3)
        ],
        grid=(3, 3),
    ),
    9,
)
_ZONED_COLONY = _Scene(_zoned_colony, 1)


def _measure_scene(measurer, scene: _Scene, radius: int) -> pd.DataFrame:
    image = OtsuDetector().apply(scene.draw(radius))
    assert image.num_objects == scene.objects, (
        f"expected {scene.objects} objects, found {image.num_objects}"
    )
    return measurer.measure(image).sort_values(OBJECT.LABEL).reset_index(drop=True)


@pytest.mark.parametrize(
    "measurer, info, scene",
    [
        pytest.param(MeasureSize(), SIZE, _ONE_DISK, id="SIZE"),
        pytest.param(MeasureShape(), SHAPE, _ONE_DISK, id="SHAPE"),
        pytest.param(MeasureIntensity(), INTENSITY, _ONE_DISK, id="INTENSITY"),
        pytest.param(MeasureBounds(), BBOX, _ONE_DISK, id="BBOX"),
        pytest.param(MeasureNeighborDist(), NEIGHBOR_DIST, _PAIR_ACROSS, id="NEIGHBOR_DIST-across"),
        pytest.param(MeasureNeighborDist(), NEIGHBOR_DIST, _PAIR_DOWN, id="NEIGHBOR_DIST-down"),
        pytest.param(MeasureGridSpread(), GRID_SPREAD, _PAIR_IN_ONE_CELL, id="GRID_SPREAD"),
        pytest.param(MeasureGridLinRegStats(), GRID_LINREG_STATS, _JITTERED_GRID, id="GRID_LINREG_STATS"),
        # Legacy mode partitions the colony into equal-area annuli, a fixed
        # fraction of its radius. Canonical Method B uses pixel-sized rings and
        # filters, so it is not scale-invariant by design.
        pytest.param(
            MeasureSymZones(legacy_mode=True), SYMMETRIC_ZONES, _ZONED_COLONY, id="SYMMETRIC_ZONES"
        ),
    ],
)
def test_columns_scale_as_their_unit_says(measurer, info, scene):
    small = _measure_scene(measurer, scene, radius=SMALL_RADIUS)
    large = _measure_scene(measurer, scene, radius=K * SMALL_RADIUS)
    checked, not_a_quantity = [], set()
    for member in info:
        p = member.unit.length_power
        column = str(member)
        if p is None:
            not_a_quantity.add(member)
            continue
        if column not in small:
            continue
        a = small[column].to_numpy(dtype=float)
        b = large[column].to_numpy(dtype=float)
        # A neighbour beyond the plate edge is NaN at both scales; nothing to compare.
        assert np.array_equal(np.isnan(a), np.isnan(b)), (column, a, b)
        present = ~np.isnan(a)
        if not present.any():
            continue
        a, b = a[present], b[present]
        if member in _ZERO_IN_ITS_SCENE:
            assert np.all(a == 0) and np.all(b == 0), (column, a, b)
            continue
        if p == 0:
            assert np.all(np.abs(b - a) <= 0.05 * np.maximum(1.0, np.abs(a))), (column, a, b)
        else:
            assert b / a == pytest.approx(np.full(a.shape, float(K**p)), rel=0.05), (column, a, b)
        checked.append(column)
    assert not_a_quantity == {m for m in info if m in _NOT_A_QUANTITY}, (
        "members skipped as NOT_A_QUANTITY differ from the allowlist"
    )
    # A loop that skipped every member would pass while checking nothing.
    assert checked, f"no {info.__name__} column was checked"


def test_orientation_is_reported_in_its_tagged_unit():
    # Scaling cannot separate RADIAN from DEGREE (both length power 0), so pin
    # the value: an ellipse whose major axis is 30 degrees off the row axis.
    rows, cols, _ = _canvas(SMALL_RADIUS)
    centre = 5 * SMALL_RADIUS - 0.5
    theta = math.radians(30)
    y, x = rows - centre, cols - centre
    along = y * math.cos(theta) + x * math.sin(theta)
    across = -y * math.sin(theta) + x * math.cos(theta)
    inside = (along / (2 * SMALL_RADIUS)) ** 2 + (across / SMALL_RADIUS) ** 2 <= 1
    image = OtsuDetector().apply(pht.Image(np.where(inside, 0.9, 0.1).astype(np.float64)))
    value = float(MeasureShape().measure(image)[str(SHAPE.ORIENTATION)].iloc[0])
    in_unit = {UNITS.RADIAN: theta, UNITS.DEGREE: 30.0, UNITS.TURN: 30 / 360}
    assert value == pytest.approx(in_unit[SHAPE.ORIENTATION.unit], rel=0.02), value

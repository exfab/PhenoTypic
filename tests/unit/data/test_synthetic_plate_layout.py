"""Generated synthetic plates put colonies on one pitch, centred in the plate.

``make_synthetic_plate`` and ``make_synthetic_filamentous_plate`` draw each
colony (for a filamentous colony, its core) with ``_radial_colony_mask``, row
by row. The tests wrap that function to label each colony's mask in an object
map, in placement order, then read the colony centroids from that map.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import regionprops

from phenotypic import GridImage
from phenotypic.data import (
    _synthetic_data,
    make_synthetic_filamentous_plate,
    make_synthetic_plate,
)
from phenotypic.grid import CenteredAutoGridFinder
from phenotypic.schema import GRID, OBJECT

_GENERATORS = pytest.mark.parametrize(
    "generator",
    [make_synthetic_plate, make_synthetic_filamentous_plate],
    ids=["yeast", "filamentous"],
)


def _render_with_objmap(monkeypatch, generator, **layout):
    """Generate a plate, labelling each colony's mask (``> 0.5``) as it is drawn.

    Returns the plate and an object map whose label ``k`` is the ``k``-th
    colony placed, so ``divmod(k - 1, ncols)`` is its ``(row, column)``.
    """
    draw_colony = _synthetic_data._radial_colony_mask
    objmap = np.zeros((layout["plate_h"], layout["plate_w"]), dtype=np.uint16)
    drawn = 0

    def labelling_draw(*args, **kwargs):
        nonlocal drawn
        mask = draw_colony(*args, **kwargs)
        drawn += 1
        objmap[(mask > 0.5) & (objmap == 0)] = drawn
        return mask

    monkeypatch.setattr(_synthetic_data, "_radial_colony_mask", labelling_draw)
    return generator(**layout), objmap


def _fit_axis(indices, coords):
    """Least-squares ``coord = first + index * pitch``; returns ``(pitch, first)``."""
    pitch, first = np.polyfit(indices, coords, 1)
    return pitch, first


@_GENERATORS
@pytest.mark.parametrize(
    ("nrows", "ncols", "plate_h", "plate_w"),
    [
        (8, 12, 300, 400),  # the bundled plate's 3:4 shape: rows were 8% wider apart
        (8, 12, 400, 300),  # portrait: the column count limits the pitch
        (4, 6, 240, 360),
        (16, 24, 360, 500),  # rows were 6% wider apart
    ],
)
def test_rows_and_columns_share_one_pitch_and_the_array_is_centred(
    monkeypatch, generator, nrows, ncols, plate_h, plate_w
):
    _, objmap = _render_with_objmap(
        monkeypatch, generator, nrows=nrows, ncols=ncols, plate_h=plate_h, plate_w=plate_w
    )
    props = regionprops(objmap)
    assert [p.label for p in props] == list(range(1, nrows * ncols + 1))
    rows, cols = np.divmod(np.array([p.label for p in props]) - 1, ncols)
    centroid_y, centroid_x = np.array([p.centroid for p in props]).T

    row_pitch, first_row_y = _fit_axis(rows, centroid_y)
    col_pitch, first_col_x = _fit_axis(cols, centroid_x)
    expected_pitch = min(plate_h / (nrows + 1), plate_w / (ncols + 1))

    # Centroids carry +-5% placement jitter and an 8% radial edge noise; fitted
    # over every colony they agree with the layout to well under 1% of a pitch.
    assert row_pitch == pytest.approx(expected_pitch, rel=0.01)
    assert col_pitch == pytest.approx(expected_pitch, rel=0.01)
    assert abs(row_pitch - col_pitch) <= 0.01 * expected_pitch
    # Centred: the middle of the fitted array is the middle of the plate.
    middle_y = first_row_y + (nrows - 1) / 2 * row_pitch
    middle_x = first_col_x + (ncols - 1) / 2 * col_pitch
    assert middle_y == pytest.approx(plate_h / 2, abs=0.05 * expected_pitch)
    assert middle_x == pytest.approx(plate_w / 2, abs=0.05 * expected_pitch)


@_GENERATORS
def test_default_grid_finder_puts_each_colony_in_its_own_well(monkeypatch, generator):
    """The default finder fits one pitch, so it now places all 96 colonies.

    On the old layout (rows ``plate_h / 9``, columns ``plate_w / 13`` apart) the
    two spacings differ by 8% at this 3:4 shape, and the finder doubled some
    wells up, as it does on the bundled plate (88 wells for 96 colonies).
    """
    arr, objmap = _render_with_objmap(
        monkeypatch, generator, nrows=8, ncols=12, plate_h=300, plate_w=400
    )
    image = GridImage(arr)
    image.objmap[:] = objmap

    grid = CenteredAutoGridFinder(nrows=8, ncols=12).measure(image)

    wells = grid.set_index(str(OBJECT.LABEL))[str(GRID.ROW_MAJOR_IDX)].astype(int)
    assert len(wells) == 96
    assert wells.nunique() == 96
    # Each colony is in the well it was placed in (row-major placement order).
    assert (wells.sort_index().to_numpy() == np.arange(96)).all()

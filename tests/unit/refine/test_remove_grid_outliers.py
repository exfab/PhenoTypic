"""RemoveGridOutliers groups by categorical grid lines without a pandas warning.

The grid row and column numbers are categorical (``pd.cut`` in the grid
finder), so a ``groupby`` on them that leaves ``observed`` implicit raises a
pandas ``FutureWarning``. Naming ``observed=False`` (the current default) must
remove the warning and change nothing else. The pinned label sets below were
recorded on ``main`` (251e364fa), before the change; the output objmap must be
the input objmap with exactly those labels zeroed, byte for byte. Each plate
includes a case where one grid line has no colonies left, so its category is
unobserved.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic.data import load_plate_72hr, load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.refine import RemoveGridOutliers
from phenotypic.schema import GRID, OBJECT

# Objects removed with ``max_coeff_variance=0`` (every grid line counts as
# noisy, so the pruning path runs on every line), recorded on main 251e364fa.
_REMOVED_ON_MAIN = {
    ("synth", None): [
        12, 80, 87, 98, 137, 151, 188, 193, 209, 212, 215,
        220, 221, 229, 279, 310, 321, 381, 390, 450, 486,
    ],
    ("synth", (GRID.ROW_NUM, 3)): [
        25, 76, 80, 98, 151, 193, 309, 310, 321, 387, 390, 431, 450, 471, 486, 551,
    ],
    ("synth", (GRID.COL_NUM, 5)): [
        12, 80, 98, 151, 188, 193, 215, 229, 321, 381, 390, 486,
    ],
    ("72h", None): [
        16, 39, 60, 67, 77, 91, 92, 95, 97, 111, 151, 216, 230, 233, 237, 243, 255,
        257, 258,
    ],
    ("72h", (GRID.COL_NUM, 5)): [
        3, 60, 68, 77, 91, 95, 97, 111, 203, 204, 216, 230, 233, 237, 255, 257, 258,
    ],
}


@pytest.fixture(scope="module")
def detected_plates():
    """The synthetic and 72 h plates, Otsu-detected, keyed by short name."""
    return {
        "synth": OtsuDetector().apply(load_synth_yeast_plate()),
        "72h": OtsuDetector().apply(load_plate_72hr(mode="GridImage")),
    }


def _plate_with_empty_line(plate, empty_line):
    """Return a copy of ``plate`` with one grid line's objects zeroed out.

    Args:
        plate: A detected ``GridImage``.
        empty_line: ``None`` for an unchanged copy, or ``(grid_column, number)``
            naming the grid row or column whose objects are removed.
    """
    image = plate.copy()
    if empty_line is None:
        return image
    column, number = empty_line
    info = image.grid.info(include_metadata=False)
    labels = info.loc[info[str(column)] == number, str(OBJECT.LABEL)]
    image.objmap[np.isin(image.objmap[:], labels.to_numpy())] = 0
    return image


@pytest.mark.parametrize("plate_name", ["synth", "72h"])
def test_apply_emits_no_future_warning(detected_plates, plate_name):
    """No FutureWarning (pandas' implicit ``observed``) escapes ``apply``.

    Recorded rather than raised: under ``-W error`` the warning surfaces as a
    ``RuntimeError`` from ``apply``, which would hide which warning it was.
    """
    image = detected_plates[plate_name].copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RemoveGridOutliers().apply(image)
    future = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert future == []


@pytest.mark.parametrize(
    ("plate_name", "empty_line"),
    list(_REMOVED_ON_MAIN),
    ids=[f"{name}-{line[0].name}{line[1]}" if line else f"{name}-full"
         for name, line in _REMOVED_ON_MAIN],
)
def test_pruned_objmap_matches_main(detected_plates, plate_name, empty_line):
    """The objmap equals main's: the input with main's outliers zeroed."""
    image = _plate_with_empty_line(detected_plates[plate_name], empty_line)
    if empty_line is not None:
        column, number = empty_line
        counts = image.grid.info(include_metadata=False)[str(column)].value_counts()
        assert counts[number] == 0, "the emptied line must be an unobserved category"
    expected = image.objmap[:].copy()
    expected[np.isin(expected, _REMOVED_ON_MAIN[(plate_name, empty_line)])] = 0

    result = RemoveGridOutliers(max_coeff_variance=0).apply(image).objmap[:]

    assert result.dtype == expected.dtype
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("empty_line", [None, (GRID.ROW_NUM, 3), (GRID.COL_NUM, 5)])
def test_default_threshold_leaves_the_synthetic_plate_unchanged(
    detected_plates, empty_line
):
    """At the default ``max_coeff_variance`` no synthetic grid line is noisy."""
    image = _plate_with_empty_line(detected_plates["synth"], empty_line)
    expected = image.objmap[:].copy()

    result = RemoveGridOutliers().apply(image).objmap[:]

    assert result.tobytes() == expected.tobytes()

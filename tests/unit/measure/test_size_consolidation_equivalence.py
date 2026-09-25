"""Moved and retained columns keep main's values (baseline: Task 1 of the plan).

Tolerance: every compared value is either a regionprops/Qhull scalar or a
reduction over at most ~1e4 pixels. Cross-platform float differences are
bounded by ~1e4 x 1 ulp ~ 2e-12 relative; rtol=1e-10 leaves 50x headroom
and is still >= 7 orders of magnitude below any real behaviour change (the
convex-area fix alone moved values by > 1e-3).

On the synth plate no colony touches another or the image border (checked
when the plan was written), so the per-object EDT must reproduce main's
whole-image EDT exactly there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic.schema import OBJECT, SIZE

_BASELINE = Path(__file__).parent / "_golden" / "size_consolidation_baseline.parquet"
RTOL = 1e-10


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    assert _BASELINE.is_file(), f"missing baseline {_BASELINE}; see plan Task 1"
    return pd.read_parquet(_BASELINE)


@pytest.fixture(scope="module")
def plate():
    return load_synth_yeast_plate()


def _aligned(new: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = new.merge(baseline, on=str(OBJECT.LABEL), validate="one_to_one")
    assert len(merged) == len(baseline) == 96
    return merged


@pytest.mark.parametrize(
    ("size_header", "old_header"),
    [
        (str(SIZE.AREA), "Shape_Area"),
        (str(SIZE.PERIMETER), "Shape_Perimeter"),
        (str(SIZE.CONVEX_AREA), "Shape_ConvexArea"),
        (str(SIZE.BBOX_AREA), "Shape_BboxArea"),
        (str(SIZE.MAJOR_AXIS_LENGTH), "Shape_MajorAxisLength"),
        (str(SIZE.MINOR_AXIS_LENGTH), "Shape_MinorAxisLength"),
        (str(SIZE.INSCRIBED_RADIUS), "Shape_MaxRadius"),
    ],
)
def test_moved_size_columns_keep_mains_values(plate, baseline, size_header, old_header):
    merged = _aligned(MeasureSize().measure(plate), baseline)
    np.testing.assert_allclose(merged[size_header], merged[old_header], rtol=RTOL)

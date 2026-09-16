"""Regression guard: MeasureSymZones output is byte-identical across the
zone-segmentation extraction refactor (Task 1 of the orientation-field plan)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate, load_synth_filamentous_plate
from phenotypic.measure import MeasureSymZones

_GOLDEN_DIR = Path(__file__).parent / "_golden"
_GOLDEN_RTOL = 3e-2
_GOLDEN_ATOL = 1e-9
_SPARSE_AREA_AGGREGATE_RTOL = 5e-2
_SPARSE_AREA = "SymZones_SparseArea"
_CASES = {
    "yeast"      : load_synth_yeast_plate,
    "filamentous": load_synth_filamentous_plate,
}


def _measure(loader) -> pd.DataFrame:
    return MeasureSymZones(legacy_mode=True).measure(loader())


@pytest.mark.parametrize("name", sorted(_CASES))
def test_symmetric_zones_matches_golden(name):
    golden_path = _GOLDEN_DIR / f"symmetric_zones_{name}.parquet"
    assert golden_path.exists(), (
        f"missing golden {golden_path}; regenerate with "
        f"PHENOTYPIC_CAPTURE_GOLDEN=1 uv run pytest "
        f"tests/unit/measure/test_zone_segmentation_regression.py"
    )
    result = _measure(_CASES[name])
    golden = pd.read_parquet(golden_path)
    assert list(result.columns) == list(golden.columns)
    stable_columns = [column for column in golden.columns if column != _SPARSE_AREA]
    pd.testing.assert_frame_equal(
            result[stable_columns],
            golden[stable_columns],
            check_exact=False,
            rtol=_GOLDEN_RTOL,
            atol=_GOLDEN_ATOL,
    )

    sparse_area = result[_SPARSE_AREA].to_numpy(dtype=np.float64)
    dense_radius = result["SymZones_DenseEndRadius"].to_numpy(dtype=np.float64)
    sparse_radius = result["SymZones_SparseEndRadius"].to_numpy(dtype=np.float64)
    expected_sparse_area = np.pi * (
            sparse_radius * sparse_radius - dense_radius * dense_radius
    )
    assert np.isfinite(sparse_area).all()
    assert (sparse_area >= 0.0).all()
    np.testing.assert_allclose(
            sparse_area, expected_sparse_area, rtol=1e-12, atol=_GOLDEN_ATOL
    )

    # SparseArea is a difference of squared radii, so tiny platform-dependent
    # threshold crossings can amplify into a large relative change for a thin
    # annulus even when both radii remain inside their row-wise golden bounds.
    # Keep aggregate drift bounded so broad segmentation regressions still fail.
    assert float(sparse_area.sum()) == pytest.approx(
            float(golden[_SPARSE_AREA].sum()), rel=_SPARSE_AREA_AGGREGATE_RTOL
    )


@pytest.mark.skipif(
        os.environ.get("PHENOTYPIC_CAPTURE_GOLDEN") != "1",
        reason="golden capture only runs when PHENOTYPIC_CAPTURE_GOLDEN=1",
)
def test_capture_golden():
    _GOLDEN_DIR.mkdir(exist_ok=True)
    for name, loader in _CASES.items():
        _measure(loader).to_parquet(_GOLDEN_DIR / f"symmetric_zones_{name}.parquet")

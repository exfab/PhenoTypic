"""Regression guard: MeasureSymmetricZones output is byte-identical across the
zone-segmentation extraction refactor (Task 1 of the orientation-field plan)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate, load_synth_filamentous_plate
from phenotypic.measure import MeasureSymmetricZones

_GOLDEN_DIR = Path(__file__).parent / "_golden"
_GOLDEN_RTOL = 3e-2
_GOLDEN_ATOL = 1e-9
_CASES = {
    "yeast": load_synth_yeast_plate,
    "filamentous": load_synth_filamentous_plate,
}


def _measure(loader) -> pd.DataFrame:
    return MeasureSymmetricZones().measure(loader())


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
    pd.testing.assert_frame_equal(
        result,
        golden,
        check_exact=False,
        rtol=_GOLDEN_RTOL,
        atol=_GOLDEN_ATOL,
    )


@pytest.mark.skipif(
    os.environ.get("PHENOTYPIC_CAPTURE_GOLDEN") != "1",
    reason="golden capture only runs when PHENOTYPIC_CAPTURE_GOLDEN=1",
)
def test_capture_golden():
    _GOLDEN_DIR.mkdir(exist_ok=True)
    for name, loader in _CASES.items():
        _measure(loader).to_parquet(_GOLDEN_DIR / f"symmetric_zones_{name}.parquet")

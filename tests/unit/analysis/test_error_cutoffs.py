"""Tests for the error-cutoff finder (good-vs-category measurement screen)."""

import pandas as pd
import pytest

from phenotypic.analysis import ErrorCutoffFinder


def _frame(values: dict[str, list[float]], n: int) -> pd.DataFrame:
    """Build a frame with the given measurement columns + filler metadata."""
    base = {
        "Metadata_ImageFile": ["p.tif"] * n,
        "Object_Label": list(range(1, n + 1)),
    }
    base.update(values)
    return pd.DataFrame(base)


def test_measurement_columns_detects_only_numeric_measurements():
    finder = ErrorCutoffFinder()
    df = _frame(
        {
            "Size_Area": [1.0, 2.0, 3.0],
            "Shape_Circularity": [0.1, 0.2, 0.3],
            "Intensity_MeanIntensity": [10.0, 11.0, 12.0],
            "Grid_RowNum": [1, 1, 2],  # grid context, not a measurement
        },
        n=3,
    )
    cols = finder.measurement_columns(df)
    assert "Size_Area" in cols
    assert "Shape_Circularity" in cols
    assert "Intensity_MeanIntensity" in cols
    # Metadata / object-id / grid-context columns are excluded.
    assert "Metadata_ImageFile" not in cols
    assert "Object_Label" not in cols
    assert "Grid_RowNum" not in cols


def test_default_min_n_fields():
    finder = ErrorCutoffFinder()
    assert finder.min_error_n == 8
    assert finder.min_good_n == 8
    # keyword-only pydantic construction; bad kwargs raise.
    with pytest.raises(Exception):
        ErrorCutoffFinder(min_error_n=5, bogus=1)


def test_enough_data_predicate():
    finder = ErrorCutoffFinder(min_error_n=3, min_good_n=3)
    good = _frame({"Size_Area": [1.0] * 5}, n=5)
    error = _frame({"Size_Area": [9.0] * 2}, n=2)  # below min_error_n
    assert finder.enough_data(good, error) is False
    error2 = _frame({"Size_Area": [9.0] * 4}, n=4)
    assert finder.enough_data(good, error2) is True

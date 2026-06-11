"""Tests for public measurement-output utilities."""

from __future__ import annotations

import pandas as pd
import polars as pl

import phenotypic
from phenotypic.schema import (
    ColorHSV,
    ColorLab,
    LINEAR_SOFTPLUS_MODEL,
    MODEL_METRICS,
    OBJECT,
    SHAPE,
    SIZE,
)
from phenotypic.util import generate_output_key, split_measurements


def _base_measurements() -> pd.DataFrame:
    """Build a small mixed measurement table for split tests."""
    return pd.DataFrame(
        {
            "Metadata_Dataset": ["ds1", "ds1"],
            str(OBJECT.LABEL): [1, 2],
            "Custom_Note": ["a", "b"],
            str(SIZE.AREA): [10.0, 20.0],
            str(SIZE.INTEGRATED_INTENSITY): [100.0, 200.0],
            str(SHAPE.AREA): [11.0, 21.0],
            str(SHAPE.PERIMETER): [12.0, 22.0],
        }
    )


def test_util_subpackage_is_public() -> None:
    """``phenotypic.util`` exposes the measurement-output helpers."""
    assert hasattr(phenotypic, "util")
    assert callable(split_measurements)
    assert callable(generate_output_key)


def test_split_measurements_groups_pandas_columns_without_pipeline() -> None:
    """Feature columns split by discovered ``MeasureFeatures`` classes."""
    frame = _base_measurements()

    splits = split_measurements(frame)

    assert set(splits) == {"MeasureSize", "MeasureShape"}
    size_df = splits["MeasureSize"]
    shape_df = splits["MeasureShape"]
    assert isinstance(size_df, pd.DataFrame)
    assert list(size_df.columns) == [
        "Metadata_Dataset",
        str(OBJECT.LABEL),
        "Custom_Note",
        str(SIZE.AREA),
        str(SIZE.INTEGRATED_INTENSITY),
    ]
    assert list(shape_df.columns) == [
        "Metadata_Dataset",
        str(OBJECT.LABEL),
        "Custom_Note",
        str(SHAPE.AREA),
        str(SHAPE.PERIMETER),
    ]


def test_split_measurements_preserves_polars_input_type() -> None:
    """Polars input returns polars split frames."""
    frame = pl.from_pandas(_base_measurements())

    splits = split_measurements(frame)

    assert set(splits) == {"MeasureSize", "MeasureShape"}
    assert isinstance(splits["MeasureSize"], pl.DataFrame)
    assert splits["MeasureSize"].columns == [
        "Metadata_Dataset",
        str(OBJECT.LABEL),
        "Custom_Note",
        str(SIZE.AREA),
        str(SIZE.INTEGRATED_INTENSITY),
    ]


def test_split_measurements_groups_plural_measure_color_infoclasses() -> None:
    """``MeasureColor`` owns columns from every color ``MeasurementInfo`` enum."""
    frame = pd.DataFrame(
        {
            "Metadata_Dataset": ["ds1"],
            str(OBJECT.LABEL): [1],
            str(ColorLab.L_STAR_GEOMEDIAN): [55.0],
            str(ColorHSV.HUE_ROBUST_MEAN): [0.2],
        }
    )

    splits = split_measurements(frame)

    assert set(splits) == {"MeasureColor"}
    assert list(splits["MeasureColor"].columns) == [
        "Metadata_Dataset",
        str(OBJECT.LABEL),
        str(ColorLab.L_STAR_GEOMEDIAN),
        str(ColorHSV.HUE_ROBUST_MEAN),
    ]


def test_split_measurements_groups_model_metrics_with_linear_softplus() -> None:
    """Shared model metrics are included with a present model-specific split."""
    frame = pd.DataFrame(
        {
            "Metadata_Strain": ["WT", "KO"],
            str(LINEAR_SOFTPLUS_MODEL.v): [1.1, 1.2],
            str(LINEAR_SOFTPLUS_MODEL.s0): [0.1, 0.2],
            str(MODEL_METRICS.RMSE): [0.01, 0.02],
            str(MODEL_METRICS.R2): [0.99, 0.98],
        }
    )

    splits = split_measurements(frame)

    assert set(splits) == {"LinearSoftplus"}
    assert list(splits["LinearSoftplus"].columns) == list(frame.columns)


def test_generate_output_key_returns_known_measurement_descriptions() -> None:
    """Only ``MeasurementInfo`` columns appear in input-column order."""
    frame = pd.DataFrame(
        {
            "Custom_Note": ["a"],
            str(OBJECT.LABEL): [1],
            str(SIZE.AREA): [10.0],
            str(LINEAR_SOFTPLUS_MODEL.v): [1.1],
            str(MODEL_METRICS.RMSE): [0.01],
        }
    )

    key = generate_output_key(frame)

    assert list(key.columns) == ["column_header", "description"]
    assert key["column_header"].tolist() == [
        str(OBJECT.LABEL),
        str(SIZE.AREA),
        str(LINEAR_SOFTPLUS_MODEL.v),
        str(MODEL_METRICS.RMSE),
    ]
    descriptions = dict(zip(key["column_header"], key["description"]))
    assert descriptions[str(OBJECT.LABEL)] == OBJECT.LABEL.desc
    assert descriptions[str(SIZE.AREA)] == SIZE.AREA.desc
    assert descriptions[str(LINEAR_SOFTPLUS_MODEL.v)] == (
        LINEAR_SOFTPLUS_MODEL.v.desc
    )
    assert descriptions[str(MODEL_METRICS.RMSE)] == MODEL_METRICS.RMSE.desc

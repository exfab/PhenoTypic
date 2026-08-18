"""Tests for public measurement-output utilities."""

from __future__ import annotations

import pandas as pd
import polars as pl

import phenotypic
from phenotypic.schema import (
    ColorHSV,
    ColorLab,
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    OBJECT,
    SHAPE,
    SIZE,
    TEXTURE,
    qualified_header,
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
    v = qualified_header(LINEAR_LAG_MODEL.v, "Area")
    s0 = qualified_header(LINEAR_LAG_MODEL.s0, "Area")
    rmse = qualified_header(MODEL_METRICS.RMSE, "Area")
    r2 = qualified_header(MODEL_METRICS.R2, "Area")
    frame = pd.DataFrame(
        {
            "Metadata_Strain": ["WT", "KO"],
            v: [1.1, 1.2],
            s0: [0.1, 0.2],
            rmse: [0.01, 0.02],
            r2: [0.99, 0.98],
        }
    )

    splits = split_measurements(frame)

    assert set(splits) == {"LinearLagModel"}
    assert list(splits["LinearLagModel"].columns) == list(frame.columns)


def test_split_measurements_recognizes_texture_dynamic_headers() -> None:
    """MeasureTexture's runtime -deg/-scale headers are now recognized."""
    headers = TEXTURE.get_headers(scale=5, matrix_name="Gray")[:3]
    frame = pd.DataFrame(
        {str(OBJECT.LABEL): [1], **{h: [0.0] for h in headers}}
    )

    splits = split_measurements(frame)

    assert "MeasureTexture" in splits
    assert all(h in splits["MeasureTexture"].columns for h in headers)


def test_generate_output_key_returns_known_measurement_descriptions() -> None:
    """Static, metric-qualified, and texture columns all resolve to a description."""
    v = qualified_header(LINEAR_LAG_MODEL.v, "Area")
    rmse = qualified_header(MODEL_METRICS.RMSE, "Area")
    texture = TEXTURE.get_headers(scale=5, matrix_name="Gray")[0]
    frame = pd.DataFrame(
        {
            "Custom_Note": ["a"],
            str(OBJECT.LABEL): [1],
            str(SIZE.AREA): [10.0],
            v: [1.1],
            rmse: [0.01],
            texture: [0.5],
        }
    )

    key = generate_output_key(frame)

    assert list(key.columns) == ["column_header", "description"]
    assert key["column_header"].tolist() == [
        str(OBJECT.LABEL),
        str(SIZE.AREA),
        v,
        rmse,
        texture,
    ]
    descriptions = dict(zip(key["column_header"], key["description"]))
    assert descriptions[str(OBJECT.LABEL)] == OBJECT.LABEL.desc
    assert descriptions[str(SIZE.AREA)] == SIZE.AREA.desc
    assert descriptions[v] == LINEAR_LAG_MODEL.v.desc
    assert descriptions[rmse] == MODEL_METRICS.RMSE.desc
    assert descriptions[texture] == TEXTURE.ANGULAR_SECOND_MOMENT.desc

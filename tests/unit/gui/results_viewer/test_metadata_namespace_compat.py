"""Results Viewer compatibility coverage for the flat metadata namespace."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from phenotypic.gui.results_viewer._metadata import (
    normalize_metadata_reference,
    normalize_viewer_frame,
)
from phenotypic.schema import CULTURE, IMAGE


def test_mixed_polars_frame_normalizes_metadata_and_preserves_other_columns() -> None:
    legacy = "MetadataCulture_Time"
    canonical = str(CULTURE.TIME)
    source = pl.DataFrame(
        {
            legacy: [1, None],
            "Metadata_Time": [None, 2],
            "Size_Area": [10.0, 11.0],
            "custom_result": ["a", "b"],
        }
    )
    original = source.clone()

    result = normalize_viewer_frame(source)

    assert result.columns == [canonical, "Size_Area", "custom_result"]
    assert result.get_column(canonical).to_list() == [1, 2]
    assert result.get_column("Size_Area").to_list() == [10.0, 11.0]
    assert result.get_column("custom_result").to_list() == ["a", "b"]
    assert source.equals(original)


def test_mixed_pandas_frame_normalization_is_copy_only() -> None:
    source = pd.DataFrame(
        {
            "MetadataImage_ImageName": ["a.png", "b.png"],
            "Object_Label": [1, 2],
            "Shape_Area": [4.0, 9.0],
        }
    )
    original = source.copy(deep=True)

    result = normalize_viewer_frame(source)

    assert list(result.columns) == [str(IMAGE.IMAGE_NAME), "Object_Label", "Shape_Area"]
    assert result is not source
    pd.testing.assert_frame_equal(source, original)


def test_conflicting_legacy_and_flat_columns_raise_without_mutation() -> None:
    source = pl.DataFrame(
        {
            "MetadataCulture_Time": [1, 2],
            "Metadata_Time": [1, 3],
            "Size_Area": [5.0, 6.0],
        }
    )
    original = source.clone()

    with pytest.raises(ValueError, match="conflicting non-null values"):
        normalize_viewer_frame(source)

    assert source.equals(original)


@pytest.mark.parametrize(
    "column",
    ["Time", "MetadataCulture_Time", "Metadata_Time"],
)
def test_metadata_reference_forms_resolve_to_canonical_member(column: str) -> None:
    assert normalize_metadata_reference(column) == str(CULTURE.TIME)


def test_nonmetadata_reference_is_unchanged() -> None:
    assert normalize_metadata_reference("Size_Area") == "Size_Area"

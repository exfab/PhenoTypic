"""Compatibility coverage for post operations consuming flat metadata headers."""

from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.post import AppendString, ExpandMetadata, MergeMetadata, PrependString
from phenotypic.schema import GENETIC, IMAGE, SAMPLE


LEGACY_STRAIN = "MetadataGenetic_Strain"
LEGACY_SAMPLE_ID = "MetadataSample_SampleID"


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (AppendString(column="Strain", value="-edited"), ["WT-edited", "mut-edited"]),
        (PrependString(column="Strain", value="edited-"), ["edited-WT", "edited-mut"]),
    ],
)
def test_string_post_operations_accept_flat_input_without_renaming_source(
    operation: AppendString | PrependString, expected: list[str]
) -> None:
    """Known bare configuration resolves a flat source without changing its header."""
    source = pd.DataFrame({"Metadata_Strain": ["WT", "mut"]})

    result = operation.apply(source)

    assert list(result.columns) == [str(GENETIC.STRAIN)]
    assert result[str(GENETIC.STRAIN)].tolist() == expected
    assert source["Metadata_Strain"].tolist() == ["WT", "mut"]


def test_merge_metadata_accepts_flat_and_current_sources_without_mutation() -> None:
    """A compatible post ingress can mix future-flat and live source headers."""
    source = pd.DataFrame(
        {
            "Metadata_Strain": ["WT", "mut"],
            str(IMAGE.IMAGE_NAME): ["plate_a", "plate_b"],
        }
    )

    result = MergeMetadata(columns=["Strain", "ImageName"], label="SampleID").apply(
        source
    )

    assert list(result.columns) == [
        "Metadata_Strain",
        str(IMAGE.IMAGE_NAME),
        str(SAMPLE.SAMPLE_ID),
    ]
    assert result[str(SAMPLE.SAMPLE_ID)].tolist() == ["WT_plate_a", "mut_plate_b"]
    assert list(source.columns) == ["Metadata_Strain", str(IMAGE.IMAGE_NAME)]


def test_expand_metadata_accepts_flat_framework_source_and_emits_live_headers() -> None:
    """Flat source headers stay intact while newly emitted fields remain current in C3."""
    source = pd.DataFrame({"Metadata_ImageName": ["WT_24h", "mut_48h"]})

    result = ExpandMetadata(
        column="Metadata_ImageName", labels=["Strain", "Time"]
    ).apply(source)

    assert list(result.columns) == [
        "Metadata_ImageName",
        str(GENETIC.STRAIN),
        "Metadata_Time",
    ]
    assert result[str(GENETIC.STRAIN)].tolist() == ["WT", "mut"]
    assert list(source.columns) == ["Metadata_ImageName"]


def test_append_coalesces_equal_metadata_aliases_without_mutating_source() -> None:
    """Equal aliases become one working column before a string operation writes."""
    source = pd.DataFrame(
        {
            LEGACY_STRAIN: ["WT", "mut"],
            str(GENETIC.STRAIN): ["WT", "mut"],
            "Shape_Area": [1.0, 2.0],
        }
    )

    result = AppendString(column="Strain", value="-edited").apply(source)

    assert result.columns.tolist() == [str(GENETIC.STRAIN), "Shape_Area"]
    assert result[str(GENETIC.STRAIN)].tolist() == ["WT-edited", "mut-edited"]
    assert source.columns.tolist() == [
        LEGACY_STRAIN,
        str(GENETIC.STRAIN),
        "Shape_Area",
    ]


@pytest.mark.parametrize(
    ("columns", "expected_columns"),
    [
        (
            [LEGACY_STRAIN, "Shape_Area", str(GENETIC.STRAIN), "Object_Label"],
            ["Shape_Area", str(GENETIC.STRAIN), "Object_Label"],
        ),
        (
            [str(GENETIC.STRAIN), "Shape_Area", LEGACY_STRAIN, "Object_Label"],
            [str(GENETIC.STRAIN), "Shape_Area", "Object_Label"],
        ),
    ],
)
def test_alias_coalescing_keeps_canonical_position_and_nonmetadata_order(
    columns: list[str], expected_columns: list[str]
) -> None:
    """Removing aliases never moves unrelated columns across the current header."""
    values = {
        LEGACY_STRAIN: ["WT"],
        str(GENETIC.STRAIN): ["WT"],
        "Shape_Area": [1.0],
        "Object_Label": [1],
    }
    source = pd.DataFrame({column: values[column] for column in columns})

    result = AppendString(column="Strain", value="-edited").apply(source)

    assert result.columns.tolist() == expected_columns
    assert result[str(GENETIC.STRAIN)].tolist() == ["WT-edited"]
    assert source.columns.tolist() == columns


def test_prepend_coalesces_complementary_metadata_aliases() -> None:
    """Complementary-null aliases merge losslessly on the post-operation copy."""
    source = pd.DataFrame(
        {
            LEGACY_STRAIN: ["WT", None],
            str(GENETIC.STRAIN): [None, "mut"],
        }
    )

    result = PrependString(column="Strain", value="edited-").apply(source)

    assert result.columns.tolist() == [str(GENETIC.STRAIN)]
    assert result[str(GENETIC.STRAIN)].tolist() == ["edited-WT", "edited-mut"]
    assert source.iloc[:, 1].tolist() == [None, "mut"]


def test_expand_reuses_existing_flat_target_alias() -> None:
    """Expand updates an equivalent target instead of adding a duplicate header."""
    source = pd.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["WT_24h", "mut_48h"],
            LEGACY_STRAIN: ["old", "old"],
        }
    )

    result = ExpandMetadata(column="ImageName", labels=["Strain", "Time"]).apply(
        source
    )

    assert result.columns.tolist().count(str(GENETIC.STRAIN)) == 1
    assert LEGACY_STRAIN not in result.columns
    assert result[str(GENETIC.STRAIN)].tolist() == ["WT", "mut"]
    assert source[LEGACY_STRAIN].tolist() == ["old", "old"]


def test_merge_reuses_existing_flat_target_alias() -> None:
    """Merge updates an equivalent target and leaves the caller frame unchanged."""
    source = pd.DataFrame(
        {
            LEGACY_STRAIN: ["WT", "mut"],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            LEGACY_SAMPLE_ID: ["old", "old"],
        }
    )

    result = MergeMetadata(columns=["Strain", "ImageName"], label="SampleID").apply(
        source
    )

    assert result.columns.tolist().count(str(SAMPLE.SAMPLE_ID)) == 1
    assert LEGACY_SAMPLE_ID not in result.columns
    assert result[str(SAMPLE.SAMPLE_ID)].tolist() == ["WT_a", "mut_b"]
    assert source[LEGACY_SAMPLE_ID].tolist() == ["old", "old"]


@pytest.mark.parametrize(
    "operation",
    [
        AppendString(column="Strain", value="-edited"),
        PrependString(column="Strain", value="edited-"),
        ExpandMetadata(column="ImageName", labels=["Strain", "Time"]),
        MergeMetadata(columns=["Strain", "ImageName"], label="SampleID"),
    ],
)
def test_post_operations_reject_conflicting_metadata_aliases_without_mutation(
    operation: AppendString | PrependString | ExpandMetadata | MergeMetadata,
) -> None:
    """Every operation rejects conflicting aliases before it can alter caller data."""
    source = pd.DataFrame(
        {
            LEGACY_STRAIN: ["WT"],
            str(GENETIC.STRAIN): ["mut"],
            str(IMAGE.IMAGE_NAME): ["plate_24h"],
        }
    )
    before = source.copy(deep=True)

    with pytest.raises(ValueError, match="conflicting non-null"):
        operation.apply(source)

    pd.testing.assert_frame_equal(source, before)


def test_append_rejects_unsafe_signed_unsigned_alias_coalescing() -> None:
    """Wide unsigned values never coerce into a signed metadata alias column."""
    source = pd.DataFrame(
        {
            LEGACY_STRAIN: pd.Series(
                [2**64 - 1], dtype="UInt64"
            ),
            str(GENETIC.STRAIN): pd.Series([1], dtype="Int64"),
        }
    )
    before = source.copy(deep=True)

    with pytest.raises(ValueError):
        AppendString(column="Strain", value="-edited").apply(source)

    pd.testing.assert_frame_equal(source, before)

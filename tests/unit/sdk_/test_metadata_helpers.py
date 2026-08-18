"""Tests for centralized metadata ownership and normalization."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from phenotypic.schema import (
    Entry,
    GENETIC,
    IMAGE,
    MetadataInfo,
)
from phenotypic.sdk_ import (
    ensure_metadata_prefix,
    is_metadata_header,
    metadata_category_for_label,
    metadata_category_prefixes,
    metadata_member_for_header,
    metadata_member_for_label,
    metadata_owner_for_header,
    metadata_owner_for_label,
    normalize_metadata_columns,
)


def test_deprecated_prefix_helper_returns_only_canonical_namespace():
    with pytest.warns(DeprecationWarning, match="is_metadata_header"):
        assert metadata_category_prefixes() == ("Metadata_",)


def test_is_metadata_header_true_for_live_enum_value():
    assert is_metadata_header(str(GENETIC.STRAIN))
    assert is_metadata_header(str(IMAGE.IMAGE_NAME))
    assert is_metadata_header("Metadata_UnknownTag")


def test_is_metadata_header_false_for_measurement_columns():
    assert not is_metadata_header("Shape_Area")
    assert not is_metadata_header("Object_Label")
    assert not is_metadata_header("Intensity_MeanIntensity")


def test_category_for_known_label_matches_owning_enum():
    with pytest.warns(DeprecationWarning, match="metadata_owner_for_label"):
        assert metadata_category_for_label("Strain") == "Metadata"


def test_category_for_unknown_label_is_none():
    with pytest.warns(DeprecationWarning, match="metadata_owner_for_label"):
        assert metadata_category_for_label("NotARealTag") is None


def test_known_names_normalize_to_canonical_emitted_header():
    current = str(GENETIC.STRAIN)
    assert ensure_metadata_prefix("Strain") == current
    assert ensure_metadata_prefix(current) == current
    assert ensure_metadata_prefix("Metadata_Strain") == current
    assert ensure_metadata_prefix("Metadata_UnknownTag") == "Metadata_UnknownTag"


def test_reverse_lookup_accepts_bare_canonical_and_legacy_names():
    member = GENETIC.STRAIN
    for name in ("Strain", "Metadata_Strain", "MetadataGenetic_Strain"):
        assert metadata_member_for_header(name) is member
        assert metadata_member_for_label(name) is member
        assert metadata_owner_for_header(name) is GENETIC
        assert metadata_owner_for_label(name) is GENETIC
    assert GENETIC.header_set() == frozenset(str(item) for item in GENETIC)


def test_unknown_generic_metadata_is_valid_but_ownerless():
    unknown = "Metadata_UnknownTag"
    assert is_metadata_header(unknown)
    assert metadata_member_for_header(unknown) is None
    assert metadata_owner_for_header(unknown) is None
    assert metadata_member_for_label("UnknownTag") is None
    assert metadata_owner_for_label("UnknownTag") is None


def test_fake_legacy_prefixes_are_not_metadata_or_owned():
    for unknown in ("MetadataFoo_Strain", "MetadataGenetic_NotARealTag"):
        assert not is_metadata_header(unknown)
        assert metadata_member_for_header(unknown) is None
        assert metadata_member_for_label(unknown) is None
        assert metadata_owner_for_header(unknown) is None
        assert metadata_owner_for_label(unknown) is None


def test_all_74_legacy_headers_resolve_to_the_exact_canonical_member():
    from phenotypic.sdk_._metadata_compatibility import (
        LEGACY_HEADER_TO_CANONICAL,
        LEGACY_HEADER_TO_MEMBER,
    )

    assert len(LEGACY_HEADER_TO_MEMBER) == 74
    assert set(LEGACY_HEADER_TO_MEMBER) == set(LEGACY_HEADER_TO_CANONICAL)
    for legacy_header, member in LEGACY_HEADER_TO_MEMBER.items():
        canonical_header = LEGACY_HEADER_TO_CANONICAL[legacy_header]
        assert canonical_header == member.value
        assert canonical_header.startswith("Metadata_")
        assert canonical_header in type(member).header_set()
        assert legacy_header not in type(member).header_set()
        assert ensure_metadata_prefix(legacy_header) == canonical_header
        assert is_metadata_header(legacy_header)
        for lookup in (metadata_member_for_header, metadata_member_for_label):
            assert lookup(legacy_header) is member
            assert lookup(canonical_header) is member
            assert lookup(member.label) is member
        for lookup in (metadata_owner_for_header, metadata_owner_for_label):
            assert lookup(legacy_header) is type(member)
            assert lookup(canonical_header) is type(member)
            assert lookup(member.label) is type(member)


def test_cluster_order_covers_every_metadata_enum():
    """Every metadata-namespace enum must be placed in the cluster order.

    A new enum added without a cluster slot fails here (coverage gate).
    """
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_OWNER_ORDER,
        _metadata_enums,
    )

    assert set(_METADATA_OWNER_ORDER) == set(_metadata_enums())
    # No duplicate placements.
    assert len(_METADATA_OWNER_ORDER) == len(set(_METADATA_OWNER_ORDER))


def test_cluster_ordered_enums_follow_constant():
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_OWNER_ORDER,
        _cluster_ordered_enums,
    )

    assert _cluster_ordered_enums() == _METADATA_OWNER_ORDER


def test_canonical_order_clusters_then_definition_order():
    from phenotypic.schema import SAMPLE
    from phenotypic.sdk_ import canonical_metadata_order

    order = canonical_metadata_order()

    # Identity (Sample) ranks before Strain (Genetic): whole clusters ordered.
    assert order[SAMPLE.SAMPLE_ID.value] < order[GENETIC.ORGANISM.value]
    # Within Genetic: definition order (Organism declared before Strain).
    assert order[GENETIC.ORGANISM.value] < order[GENETIC.STRAIN.value]
    assert order[IMAGE.IMAGE_NAME.value] > order[GENETIC.STRAIN.value]


def test_canonical_order_unknown_header_absent():
    from phenotypic.sdk_ import canonical_metadata_order

    assert "Metadata_TotallyUnknownTag" not in canonical_metadata_order()


def test_canonical_headers_use_owner_and_member_order():
    from phenotypic.schema import SAMPLE
    from phenotypic.sdk_ import canonical_metadata_order, order_measurement_columns

    rank = canonical_metadata_order()
    assert rank["Metadata_SampleID"] == rank[str(SAMPLE.SAMPLE_ID)]
    assert rank["Metadata_Strain"] == rank[str(GENETIC.STRAIN)]

    columns = [
        "Shape_Area",
        "Metadata_Strain",
        "Metadata_UnknownTag",
        "Metadata_ImageName",
        "Metadata_SampleID",
    ]
    assert order_measurement_columns(columns) == [
        "Metadata_SampleID",
        "Metadata_Strain",
        "Metadata_UnknownTag",
        "Shape_Area",
        "Metadata_ImageName",
    ]


def test_owner_discovery_uses_metadata_base_not_category_prefix():
    from phenotypic.sdk_._metadata_helpers import _metadata_enums

    assert all(issubclass(owner, MetadataInfo) for owner in _metadata_enums())
    assert GENETIC in _metadata_enums()
    assert IMAGE in _metadata_enums()


def test_registry_fails_fast_on_duplicate_labels_and_flat_headers():
    from phenotypic.sdk_._metadata_helpers import _build_metadata_registry

    class FIRST_OWNER(MetadataInfo):
        @classmethod
        def category(cls):
            return "MetadataFirst"

        DUPLICATE = Entry("Duplicate")

    class SECOND_OWNER(MetadataInfo):
        @classmethod
        def category(cls):
            return "MetadataSecond"

        DUPLICATE = Entry("Duplicate")

    with pytest.raises(ValueError, match="Duplicate metadata label"):
        _build_metadata_registry((FIRST_OWNER, SECOND_OWNER))


def test_registry_rejects_duplicate_declarations_hidden_as_enum_aliases():
    from phenotypic.sdk_._metadata_helpers import _build_metadata_registry

    class ALIASED_OWNER(MetadataInfo):
        @classmethod
        def category(cls):
            return "MetadataAliased"

        FIRST = Entry("Duplicate")
        SECOND = Entry("Duplicate")

    assert len(ALIASED_OWNER.__members__) == 2
    assert len(ALIASED_OWNER) == 1
    with pytest.raises(ValueError, match="Enum aliases would hide"):
        _build_metadata_registry((ALIASED_OWNER,))


def test_pandas_normalization_returns_new_frame_and_coalesces():
    frame = pd.DataFrame(
        {
            "MetadataGenetic_Strain": ["BY4741", None],
            "Media": ["YPD", "SC"],
            str(GENETIC.STRAIN): [None, "BY4742"],
        }
    )
    original = frame.copy(deep=True)

    normalized = normalize_metadata_columns(frame)

    assert normalized is not frame
    pd.testing.assert_frame_equal(frame, original)
    assert normalized.columns.tolist() == [
        "Metadata_Media",
        str(GENETIC.STRAIN),
    ]
    assert normalized[str(GENETIC.STRAIN)].tolist() == ["BY4741", "BY4742"]


def test_polars_normalization_returns_new_frame_and_coalesces():
    frame = pl.DataFrame(
        {
            "MetadataGenetic_Strain": ["BY4741", None],
            "Media": ["YPD", "SC"],
            str(GENETIC.STRAIN): [None, "BY4742"],
        }
    )
    original = frame.clone()

    normalized = normalize_metadata_columns(frame)

    assert normalized is not frame
    assert frame.equals(original)
    assert normalized.columns == ["Metadata_Media", str(GENETIC.STRAIN)]
    assert normalized[str(GENETIC.STRAIN)].to_list() == ["BY4741", "BY4742"]


def test_pandas_object_and_nullable_string_coalesce_to_string_at_canonical_position():
    current = str(GENETIC.STRAIN)
    frame = pd.DataFrame(
        {
            "MetadataGenetic_Strain": pd.Series(["BY4741", None], dtype=object),
            "Media": ["YPD", "SC"],
            current: pd.Series(["BY4741", "BY4742"], dtype="string"),
        }
    )
    original = frame.copy(deep=True)

    normalized = normalize_metadata_columns(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert normalized.columns.tolist() == ["Metadata_Media", current]
    assert normalized[current].tolist() == ["BY4741", "BY4742"]
    assert normalized[current].dtype == pd.StringDtype(storage="python")


def test_pandas_nullable_string_and_object_complementary_values_are_lossless():
    current = str(GENETIC.STRAIN)
    frame = pd.DataFrame(
        {
            "MetadataGenetic_Strain": pd.Series(
                ["BY4741", pd.NA], dtype="string"
            ),
            current: pd.Series([None, "BY4742"], dtype=object),
        }
    )
    original = frame.copy(deep=True)

    normalized = normalize_metadata_columns(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert normalized[current].tolist() == ["BY4741", "BY4742"]
    assert normalized[current].dtype == pd.StringDtype(storage="python")


def test_polars_categorical_and_string_coalesce_losslessly_to_string():
    current = str(GENETIC.STRAIN)
    frame = pl.DataFrame(
        [
            pl.Series(
                "MetadataGenetic_Strain",
                ["BY4741", None, "BY4743"],
                dtype=pl.Categorical,
            ),
            pl.Series(
                current,
                ["BY4741", "BY4742", None],
                dtype=pl.String,
            ),
        ]
    )
    original = frame.clone()

    normalized = normalize_metadata_columns(frame)

    assert frame.equals(original)
    assert normalized[current].to_list() == ["BY4741", "BY4742", "BY4743"]
    assert normalized[current].dtype == pl.String


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("left_values", "right_values"),
    [
        ([1, None], [1, 2]),
        ([1, None], [None, 2]),
    ],
    ids=["equal-overlap", "complementary"],
)
def test_int32_int64_coalesce_to_int64_losslessly(
    frame_kind,
    left_values,
    right_values,
):
    current = str(GENETIC.STRAIN)
    if frame_kind == "pandas":
        frame = pd.DataFrame(
            {
                "MetadataGenetic_Strain": pd.Series(left_values, dtype="Int32"),
                current: pd.Series(right_values, dtype="Int64"),
            }
        )
        original = frame.copy(deep=True)
    else:
        frame = pl.DataFrame(
            [
                pl.Series("MetadataGenetic_Strain", left_values, dtype=pl.Int32),
                pl.Series(current, right_values, dtype=pl.Int64),
            ]
        )
        original = frame.clone()

    normalized = normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
        assert normalized[current].tolist() == [1, 2]
        assert normalized[current].dtype == pd.Int64Dtype()
    else:
        assert frame.equals(original)
        assert normalized[current].to_list() == [1, 2]
        assert normalized[current].dtype == pl.Int64


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_float32_float64_coalesce_to_float64_when_roundtrip_is_exact(frame_kind):
    current = str(GENETIC.STRAIN)
    if frame_kind == "pandas":
        frame = pd.DataFrame(
            {
                "MetadataGenetic_Strain": pd.Series(
                    [1.5, None], dtype="Float32"
                ),
                current: pd.Series([1.5, 2.25], dtype="Float64"),
            }
        )
        original = frame.copy(deep=True)
    else:
        frame = pl.DataFrame(
            [
                pl.Series(
                    "MetadataGenetic_Strain", [1.5, None], dtype=pl.Float32
                ),
                pl.Series(current, [1.5, 2.25], dtype=pl.Float64),
            ]
        )
        original = frame.clone()

    normalized = normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
        assert normalized[current].tolist() == [1.5, 2.25]
        assert normalized[current].dtype == pd.Float64Dtype()
    else:
        assert frame.equals(original)
        assert normalized[current].to_list() == [1.5, 2.25]
        assert normalized[current].dtype == pl.Float64


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_coalesces_equal_overlapping_values(frame_kind):
    data = {
        "MetadataGenetic_Strain": ["BY4741", None],
        str(GENETIC.STRAIN): ["BY4741", "BY4742"],
    }
    frame = pd.DataFrame(data) if frame_kind == "pandas" else pl.DataFrame(data)
    original = frame.copy(deep=True) if frame_kind == "pandas" else frame.clone()

    normalized = normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        assert normalized[str(GENETIC.STRAIN)].tolist() == ["BY4741", "BY4742"]
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert normalized[str(GENETIC.STRAIN)].to_list() == ["BY4741", "BY4742"]
        assert frame.equals(original)


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_rejects_conflicts_without_mutating(frame_kind):
    data = {
        "MetadataGenetic_Strain": ["BY4741"],
        str(GENETIC.STRAIN): ["BY4742"],
    }
    frame = pd.DataFrame(data) if frame_kind == "pandas" else pl.DataFrame(data)
    original = frame.copy(deep=True) if frame_kind == "pandas" else frame.clone()

    with pytest.raises(ValueError, match="conflicting non-null values"):
        normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_rejects_incompatible_dtypes(frame_kind):
    data = {
        "MetadataGenetic_Strain": [None, "BY4741"],
        str(GENETIC.STRAIN): [1, None],
    }
    frame = pd.DataFrame(data) if frame_kind == "pandas" else pl.DataFrame(data)

    with pytest.raises(ValueError, match="incompatible dtypes"):
        normalize_metadata_columns(frame)


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_rejects_uint64_int64_without_losing_max(frame_kind):
    maximum = 2**64 - 1
    if frame_kind == "pandas":
        frame = pd.DataFrame(
            {
                "MetadataGenetic_Strain": pd.Series(
                    [maximum, None], dtype="UInt64"
                ),
                str(GENETIC.STRAIN): pd.Series([None, 1], dtype="Int64"),
            }
        )
        original = frame.copy(deep=True)
    else:
        frame = pl.DataFrame(
            [
                pl.Series(
                    "MetadataGenetic_Strain", [maximum, None], dtype=pl.UInt64
                ),
                pl.Series(str(GENETIC.STRAIN), [None, 1], dtype=pl.Int64),
            ]
        )
        original = frame.clone()

    with pytest.raises(ValueError, match="incompatible dtypes"):
        normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
        assert frame["MetadataGenetic_Strain"].iloc[0] == maximum
    else:
        assert frame.equals(original)
        assert frame["MetadataGenetic_Strain"][0] == maximum


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_rejects_boolean_integer_without_mutating(frame_kind):
    if frame_kind == "pandas":
        frame = pd.DataFrame(
            {
                "MetadataGenetic_Strain": pd.Series(
                    [True, None], dtype="boolean"
                ),
                str(GENETIC.STRAIN): pd.Series([None, 1], dtype="Int64"),
            }
        )
        original = frame.copy(deep=True)
    else:
        frame = pl.DataFrame(
            [
                pl.Series(
                    "MetadataGenetic_Strain", [True, None], dtype=pl.Boolean
                ),
                pl.Series(str(GENETIC.STRAIN), [None, 1], dtype=pl.Int64),
            ]
        )
        original = frame.clone()

    with pytest.raises(ValueError, match="incompatible dtypes"):
        normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_normalization_rejects_uint64_float_precision_risk(frame_kind):
    maximum = 2**64 - 1
    if frame_kind == "pandas":
        frame = pd.DataFrame(
            {
                "MetadataGenetic_Strain": pd.Series(
                    [maximum, None], dtype="UInt64"
                ),
                str(GENETIC.STRAIN): pd.Series([None, float(2**53)], dtype="Float64"),
            }
        )
        original = frame.copy(deep=True)
    else:
        frame = pl.DataFrame(
            [
                pl.Series(
                    "MetadataGenetic_Strain", [maximum, None], dtype=pl.UInt64
                ),
                pl.Series(
                    str(GENETIC.STRAIN),
                    [None, float(2**53)],
                    dtype=pl.Float64,
                ),
            ]
        )
        original = frame.clone()

    with pytest.raises(ValueError, match="incompatible dtypes"):
        normalize_metadata_columns(frame)

    if frame_kind == "pandas":
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)

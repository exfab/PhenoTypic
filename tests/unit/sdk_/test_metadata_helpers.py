"""Tests for the centralized metadata-namespace helpers (Task B1).

Decouple-then-flip discipline: every assertion derives from the **live** enum
``category()``/``value``, never a hardcoded post-rename string. The categories
still return ``"Metadata"`` during the decouple phase and flip to per-enum
``Metadata<Topic>`` later (Task B2); deriving from the enum keeps these tests
green in both phases.
"""

from __future__ import annotations

from phenotypic.schema import (
    GENETIC_METADATA,
    METADATA,
)
from phenotypic.sdk_ import (
    is_metadata_header,
    metadata_category_for_label,
    metadata_category_prefixes,
)


def _expected_metadata_prefixes() -> tuple[str, ...]:
    """Expected prefixes in bio-semantic cluster order (independent of the helper)."""
    return (
        "MetadataSample_",
        "MetadataPlate_",
        "MetadataGenetic_",
        "MetadataCondition_",
        "MetadataCulture_",
        "MetadataExperiment_",
        "MetadataStudy_",
        "MetadataAcquisition_",
        "MetadataImage_",
    )


def test_prefixes_match_cluster_order():
    assert metadata_category_prefixes() == _expected_metadata_prefixes()


def test_prefixes_are_well_formed_and_unique():
    prefixes = metadata_category_prefixes()
    assert prefixes  # non-empty
    assert all(p.startswith("Metadata") and p.endswith("_") for p in prefixes)
    assert len(set(prefixes)) == len(prefixes)  # no duplicates


def test_prefixes_cover_genetic_and_framework_enums():
    prefixes = metadata_category_prefixes()
    # The owning enum's own category prefix must be present, both phases.
    assert f"{GENETIC_METADATA.category()}_" in prefixes
    assert f"{METADATA.category()}_" in prefixes  # IMAGE_DATA framework enum


def test_prefix_order_follows_cluster_rank():
    """Strain (MetadataGenetic) sorts ahead of the trailing framework Image block."""
    prefixes = metadata_category_prefixes()
    assert prefixes.index("MetadataGenetic_") < prefixes.index("MetadataImage_")
    # Identity leads: Sample is the first prefix.
    assert prefixes[0] == "MetadataSample_"


def test_is_metadata_header_true_for_live_enum_value():
    # "Metadata_Strain" now, "MetadataGenetic_Strain" after the flip.
    assert is_metadata_header(str(GENETIC_METADATA.STRAIN))
    assert is_metadata_header(str(METADATA.IMAGE_NAME))


def test_is_metadata_header_false_for_measurement_columns():
    assert not is_metadata_header("Shape_Area")
    assert not is_metadata_header("Object_Label")
    assert not is_metadata_header("Intensity_MeanIntensity")


def test_category_for_known_label_matches_owning_enum():
    assert metadata_category_for_label("Strain") == GENETIC_METADATA.category()


def test_category_for_unknown_label_is_none():
    assert metadata_category_for_label("NotARealTag") is None


def test_cluster_order_covers_every_metadata_enum():
    """Every metadata-namespace enum must be placed in the cluster order.

    A new enum added without a cluster slot fails here (coverage gate).
    """
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_CLUSTER_ORDER,
        _metadata_enums,
    )

    assert set(_METADATA_CLUSTER_ORDER) == {e.category() for e in _metadata_enums()}
    # No duplicate placements.
    assert len(_METADATA_CLUSTER_ORDER) == len(set(_METADATA_CLUSTER_ORDER))


def test_cluster_ordered_enums_follow_constant():
    from phenotypic.sdk_._metadata_helpers import (
        _METADATA_CLUSTER_ORDER,
        _cluster_ordered_enums,
    )

    cats = [e.category() for e in _cluster_ordered_enums()]
    assert cats == list(_METADATA_CLUSTER_ORDER)


def test_canonical_order_clusters_then_definition_order():
    from phenotypic.schema import GENETIC_METADATA, SAMPLE_METADATA
    from phenotypic.sdk_ import canonical_metadata_order

    order = canonical_metadata_order()

    # Identity (Sample) ranks before Strain (Genetic): whole clusters ordered.
    assert order[SAMPLE_METADATA.SAMPLE_ID.value] < order[GENETIC_METADATA.ORGANISM.value]
    # Within Genetic: definition order (Organism declared before Strain).
    assert order[GENETIC_METADATA.ORGANISM.value] < order[GENETIC_METADATA.STRAIN.value]
    # MetadataImage_ ranks last among categories.
    from phenotypic.schema import METADATA
    assert order[METADATA.IMAGE_NAME.value] > order[GENETIC_METADATA.STRAIN.value]


def test_canonical_order_unknown_header_absent():
    from phenotypic.sdk_ import canonical_metadata_order

    assert "Metadata_TotallyUnknownTag" not in canonical_metadata_order()

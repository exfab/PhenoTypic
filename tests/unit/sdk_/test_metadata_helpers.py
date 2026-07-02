"""Tests for the centralized metadata-namespace helpers (Task B1).

Decouple-then-flip discipline: every assertion derives from the **live** enum
``category()``/``value``, never a hardcoded post-rename string. The categories
still return ``"Metadata"`` during the decouple phase and flip to per-enum
``Metadata<Topic>`` later (Task B2); deriving from the enum keeps these tests
green in both phases.
"""

from __future__ import annotations

import phenotypic.schema as schema
from phenotypic.schema import (
    GENETIC_METADATA,
    METADATA,
    MeasurementInfo,
    REMBI_MODULE,
)
from phenotypic.sdk_ import (
    is_metadata_header,
    metadata_category_for_label,
    metadata_category_prefixes,
)


def _expected_metadata_prefixes() -> tuple[str, ...]:
    """Independently derive the expected prefixes from the public schema.

    Mirrors the helper's contract without reusing its private internals: walk
    the exported metadata-namespace enums (``category()`` starts with
    ``"Metadata"``), order by REMBI module then category, and dedupe.
    """
    order = {m: i for i, m in enumerate(REMBI_MODULE)}
    enums = []
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
            and obj.category().startswith("Metadata")
        ):
            enums.append(obj)
    enums.sort(
        key=lambda e: (order.get(next(iter(e)).resolved_rembi_module, 99), e.category())
    )
    return tuple(dict.fromkeys(f"{e.category()}_" for e in enums))


def test_prefixes_match_schema_derivation():
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


def test_prefix_order_follows_rembi_module_rank():
    """Prefixes group by REMBI-module rank: a lower-ranked module's prefix
    sorts before a higher-ranked one.

    Algorithm-independent: derived from the *public* ``REMBI_MODULE`` definition
    order plus each owning enum's live ``category()`` -- not the helper's private
    walk (unlike ``test_prefixes_match_schema_derivation``, which re-derives it).
    Flip-stable: in the decouple phase the two categories coincide and the guard
    makes the ordering check vacuous; post-flip it pins Biosample
    (``GENETIC_METADATA``, rank 1) ahead of the ImageData framework enum
    (``METADATA``, rank 4).
    """
    order = {m: i for i, m in enumerate(REMBI_MODULE)}
    # Anchor on the public taxonomy the helper must honour.
    assert order[REMBI_MODULE.BIOSAMPLE] < order[REMBI_MODULE.IMAGE_DATA]

    prefixes = metadata_category_prefixes()
    biosample = f"{GENETIC_METADATA.category()}_"  # REMBI Biosample
    image_data = f"{METADATA.category()}_"         # REMBI ImageData
    if biosample != image_data:  # distinct only after the category flip
        assert prefixes.index(biosample) < prefixes.index(image_data)


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

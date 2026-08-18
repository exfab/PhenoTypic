"""Ownership and uniqueness gates for the metadata vocabulary."""
from collections import defaultdict
import warnings

import phenotypic.schema as schema
import pytest
from phenotypic.schema import MetadataInfo


def _metadata_enums():
    seen = set()
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MetadataInfo)
            and obj is not MetadataInfo
            and list(obj)
            and obj not in seen
        ):
            seen.add(obj)
            yield obj


def test_metadata_labels_are_globally_unique():
    owners: dict[str, list[str]] = defaultdict(list)
    for enum in _metadata_enums():
        for member in enum:
            owners[member.label].append(f"{enum.__name__}.{member.name}")
    collisions = {label: sites for label, sites in owners.items() if len(sites) > 1}
    assert not collisions, (
        "metadata labels must be unique across the Metadata-namespace enums so "
        "metadata_category_for_label cannot mis-route:\n"
        + "\n".join(f"  {label!r}: {sites}" for label, sites in collisions.items())
    )


def test_metadata_vocabulary_has_74_labels_with_exactly_one_owner():
    members = [member for owner in _metadata_enums() for member in owner]
    assert len(members) == 74
    assert len({member.label for member in members}) == 74
    for member in members:
        owners = [
            owner
            for owner in _metadata_enums()
            if any(candidate is member for candidate in owner)
        ]
        assert owners == [type(member)]


def test_all_nine_metadata_enums_use_flat_namespace_and_canonical_class_names():
    owners = tuple(_metadata_enums())
    expected_names = {
        "IMAGE",
        "GENETIC",
        "SAMPLE",
        "PLATE",
        "CONDITION",
        "CULTURE",
        "EXPERIMENT",
        "STUDY",
        "ACQUISITION",
    }
    assert len(owners) == 9
    assert MetadataInfo.category() == "Metadata"
    assert all(issubclass(owner, MetadataInfo) for owner in owners)
    assert {owner.__name__ for owner in owners} == expected_names
    assert all(owner.category() == "Metadata" for owner in owners)
    assert all(
        member.value == f"Metadata_{member.label}"
        for owner in owners
        for member in owner
    )
    assert str(schema.GENETIC.STRAIN) == "Metadata_Strain"
    assert str(schema.IMAGE.IMAGE_NAME) == "Metadata_ImageName"


def test_legacy_python_names_warn_resolve_by_identity_and_stay_out_of_all():
    aliases = {
        "METADATA": schema.IMAGE,
        "GENETIC_METADATA": schema.GENETIC,
        "SAMPLE_METADATA": schema.SAMPLE,
        "PLATE_METADATA": schema.PLATE,
        "CONDITION_METADATA": schema.CONDITION,
        "CULTURE_METADATA": schema.CULTURE,
        "EXPERIMENT_METADATA": schema.EXPERIMENT,
        "STUDY_METADATA": schema.STUDY,
        "ACQUISITION_METADATA": schema.ACQUISITION,
    }
    for legacy_name, canonical_owner in aliases.items():
        assert legacy_name not in schema.__all__
        assert legacy_name not in vars(schema)
        with pytest.warns(DeprecationWarning, match=canonical_owner.__name__):
            assert getattr(schema, legacy_name) is canonical_owner
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from phenotypic.schema import METADATA

    assert METADATA is schema.IMAGE
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)


def test_direct_experimental_tags_package_imports_have_the_same_transition_aliases():
    import phenotypic.schema._experimental_tags as tags

    assert "GENETIC_METADATA" not in tags.__all__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from phenotypic.schema._experimental_tags import GENETIC_METADATA

    assert GENETIC_METADATA is schema.GENETIC
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)

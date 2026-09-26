"""The CATEGORIES vocabulary and Entry.categories normalization."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import phenotypic.schema as schema
from phenotypic.schema import (
    CATEGORIES,
    CategoryEntry,
    Entry,
    MeasurementInfo,
    MetadataInfo,
)

_CATEGORIES_MODULE = (
    Path(__file__).resolve().parents[3] / "src" / "phenotypic" / "schema" / "_categories.py"
)
_STDLIB_ONLY = {"__future__", "re", "dataclasses", "enum", "typing", "collections.abc"}


def test_categories_is_not_a_measurement_info() -> None:
    assert not issubclass(CATEGORIES, MeasurementInfo)


def test_value_is_the_label_with_no_family_prefix() -> None:
    assert CATEGORIES.STARTING_METRICS.value == "StartingMetrics"
    assert CATEGORIES.STARTING_METRICS.label == "StartingMetrics"
    assert str(CATEGORIES.STARTING_METRICS) == "StartingMetrics"
    assert CATEGORIES.STARTING_METRICS.desc


def test_display_name_splits_camel_case() -> None:
    assert CATEGORIES.STARTING_METRICS.display_name == "Starting Metrics"


@pytest.mark.parametrize("category", list(CATEGORIES), ids=lambda c: c.name)
def test_every_display_name_is_readable(category: CATEGORIES) -> None:
    words = category.display_name.split(" ")
    assert "".join(words) == category.label
    assert all(word[:1].isupper() or word[:1].isdigit() for word in words)


def test_anchor_is_stable() -> None:
    assert CATEGORIES.STARTING_METRICS.anchor == "measurement-category-startingmetrics"


def test_members_must_be_category_entries() -> None:
    # __new__ delegates to _validate_entry; an Enum with members cannot be
    # subclassed, so the validator is exercised directly.
    with pytest.raises(TypeError, match="CategoryEntry"):
        CATEGORIES._validate_entry("not-an-entry")


@pytest.mark.parametrize("label", ["startingMetrics", "Starting Metrics", "", "Starting_Metrics"])
def test_category_entry_rejects_non_camel_case_labels(label: str) -> None:
    with pytest.raises(ValueError, match="CamelCase"):
        CategoryEntry(label, "desc")


def test_category_entry_rejects_empty_desc() -> None:
    with pytest.raises(ValueError, match="desc"):
        CategoryEntry("Valid", "   ")


def test_bare_member_is_one_category_not_its_characters() -> None:
    entry = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)
    assert entry.categories == frozenset({CATEGORIES.STARTING_METRICS})


def test_iterable_of_members_normalizes_to_frozenset() -> None:
    entry = Entry("Value", "A value.", categories=[CATEGORIES.STARTING_METRICS] * 2)
    assert entry.categories == frozenset({CATEGORIES.STARTING_METRICS})


def test_default_is_empty() -> None:
    assert Entry("Value", "A value.").categories == frozenset()


def test_raw_string_equal_to_a_member_value_is_rejected() -> None:
    assert "StartingMetrics" == CATEGORIES.STARTING_METRICS  # str enum equality
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories="StartingMetrics")


def test_non_member_element_is_rejected() -> None:
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories=[CATEGORIES.STARTING_METRICS, "Other"])


def test_non_iterable_is_rejected() -> None:
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories=3)  # type: ignore[arg-type]


def test_member_exposes_its_categories() -> None:
    class TAGGED(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "Tagged"

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)
        OTHER = Entry("Other", "Another value.")

    assert TAGGED.VALUE.categories == frozenset({CATEGORIES.STARTING_METRICS})
    assert TAGGED.OTHER.categories == frozenset()


def test_members_finds_tagged_public_members(monkeypatch: pytest.MonkeyPatch) -> None:
    class FUTURE_TAGGED(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "FutureTagged"

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)

    monkeypatch.setattr(schema, "FUTURE_TAGGED", FUTURE_TAGGED, raising=False)
    monkeypatch.setattr(schema, "__all__", [*schema.__all__, "FUTURE_TAGGED"])
    assert FUTURE_TAGGED.VALUE in CATEGORIES.STARTING_METRICS.members()


def test_in_order_follows_declaration_order() -> None:
    assert CATEGORIES.in_order({CATEGORIES.STARTING_METRICS}) == (CATEGORIES.STARTING_METRICS,)


def test_no_metadata_member_carries_a_category() -> None:
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if isinstance(value, type) and issubclass(value, MetadataInfo):
            tagged = [m for m in value if m.categories]
            assert tagged == [], f"{name} metadata members may not be categorized: {tagged}"


def test_categories_module_imports_only_stdlib_at_module_level() -> None:
    tree = ast.parse(_CATEGORIES_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= _STDLIB_ONLY
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module in _STDLIB_ONLY, ast.dump(node)


_STARTING_METRICS_HEADERS = {
    "Size_Area",
    "Size_IntegratedIntensity",
    "Size_Perimeter",
    "Size_ConvexArea",
    "Size_BboxArea",
    "Size_MajorAxisLength",
    "Size_MinorAxisLength",
    "Size_MinFeretDiameter",
    "Size_MaxFeretDiameter",
    "Size_InscribedRadius",
    "Size_MedianRadius",
    "Size_MeanRadius",
    "Size_RobustMeanRadius",
    "Size_MaxRadius",
    "ColorLab_L*Medoid",
    "ColorLab_a*Medoid",
    "ColorLab_b*Medoid",
    "Intensity_IntegratedIntensity",
}


def test_starting_metrics_membership_is_pinned() -> None:
    members = CATEGORIES.STARTING_METRICS.members()
    assert {m.value for m in members} == _STARTING_METRICS_HEADERS
    assert len(members) == 18


def test_every_category_has_a_member() -> None:
    for category in CATEGORIES:
        assert category.members(), f"{category.name} has no members"

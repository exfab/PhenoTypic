"""The CATEGORIES vocabulary and Entry.categories normalization."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from enum import Enum
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
from phenotypic.schema._categories import _CAMEL_BOUNDARY_RE

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


# Digit-bearing labels ("Size2D", "Tier1Traits") are deliberately not pinned:
# how a digit run should split is undecided, and no current label has one.
@pytest.mark.parametrize(
    ("label", "display_name"),
    [
        ("StartingMetrics", "Starting Metrics"),
        ("CIELabColor", "CIE Lab Color"),
        ("QCFlags", "QC Flags"),
    ],
)
def test_camel_boundary_split_is_pinned(label: str, display_name: str) -> None:
    assert _CAMEL_BOUNDARY_RE.sub(" ", label) == display_name


def test_anchor_is_stable() -> None:
    assert CATEGORIES.STARTING_METRICS.anchor == "measurement-category-startingmetrics"


def test_members_must_be_category_entries() -> None:
    # An Enum with members cannot be subclassed, but the class's own __new__
    # (the one that builds members) stays reachable as _new_member_.
    with pytest.raises(TypeError, match="CategoryEntry"):
        CATEGORIES._new_member_(CATEGORIES, "not-an-entry")


@pytest.mark.parametrize(
    "label",
    ["startingMetrics", "Starting Metrics", "", "Starting_Metrics", "StartingMetrics\n"],
)
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


def test_mapping_is_rejected_rather_than_reduced_to_its_keys() -> None:
    with pytest.raises(TypeError, match="mapping"):
        Entry("Value", "A value.", categories={CATEGORIES.STARTING_METRICS: "why"})


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
    # Listed twice, as a compatibility alias would be: members() must
    # deduplicate by class rather than walk the same enum again.
    monkeypatch.setattr(
        schema, "__all__", [*schema.__all__, "FUTURE_TAGGED", "FUTURE_TAGGED"]
    )
    assert CATEGORIES.STARTING_METRICS.members().count(FUTURE_TAGGED.VALUE) == 1


def test_in_order_follows_declaration_order() -> None:
    # CATEGORIES has one member, which cannot tell a sort from no sort, so the
    # classmethod's function is bound to a local two-member enum instead.
    class _Local(str, Enum):
        A = "A"
        B = "B"

    in_order = CATEGORIES.in_order.__func__  # type: ignore[attr-defined]
    assert in_order(_Local, [_Local.B, _Local.A]) == (_Local.A, _Local.B)


def test_no_metadata_member_carries_a_category() -> None:
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if isinstance(value, type) and issubclass(value, MetadataInfo):
            tagged = [m for m in value if m.categories]
            assert tagged == [], f"{name} metadata members may not be categorized: {tagged}"


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_imports(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Every import that executes when the module is imported.

    Walks the whole module (``try``/``if`` blocks and class bodies included)
    but skips function bodies, which run only when called, and the body of an
    ``if TYPE_CHECKING:`` block, which never runs.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _non_stdlib_imports(source: str) -> list[str]:
    bad = []
    for node in _import_time_imports(ast.parse(source)):
        if isinstance(node, ast.Import):
            bad.extend(alias.name for alias in node.names if alias.name not in _STDLIB_ONLY)
        elif node.level != 0 or node.module not in _STDLIB_ONLY:
            bad.append(ast.dump(node))
    return bad


def test_categories_module_imports_only_stdlib_at_module_level() -> None:
    source = _CATEGORIES_MODULE.read_text(encoding="utf-8")
    assert _non_stdlib_imports(source) == []


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("try:\n    import numpy\nexcept ImportError:\n    pass\n", True),
        ("import re\nif re:\n    from . import _sibling\n", True),
        ("class A:\n    import numpy\n", True),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\n", False),
        ("def f():\n    import numpy\n", False),
        ("class A:\n    def f(self):\n        import numpy\n", False),
    ],
)
def test_stdlib_guard_sees_nested_imports(source: str, flagged: bool) -> None:
    assert bool(_non_stdlib_imports(source)) is flagged


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
    # schema.__all__ order, then member order; Task 8's Categories page
    # renders in this order.
    assert [m.value for m in members][:4] == [
        "ColorLab_L*Medoid",
        "ColorLab_a*Medoid",
        "ColorLab_b*Medoid",
        "Intensity_IntegratedIntensity",
    ]


def test_every_category_has_a_member() -> None:
    for category in CATEGORIES:
        assert category.members(), f"{category.name} has no members"


@pytest.mark.parametrize("category", list(CATEGORIES), ids=lambda c: c.name)
def test_category_desc_is_safe_to_emit_verbatim_as_rst(category: CATEGORIES) -> None:
    # The generated Categories page writes each desc into RST unescaped (spec:
    # "verbatim"). A word-initial *, `, | or _ would start inline markup and
    # warn or mis-render while the docs build still exits 0.
    import re

    assert not re.search(r"(?:^|\s)[*`|_]", category.desc), category.desc

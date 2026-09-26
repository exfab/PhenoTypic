"""``MeasurementInfo.category()`` was hard-renamed to ``metric_family()``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phenotypic.schema import SIZE, Entry, MeasurementInfo

_REPO = Path(__file__).resolve().parents[3]

#: Where a leftover could be copied from: shipped code, user docs, the agent
#: skills and the root guide. ``docs/superpowers`` is historical and excluded.
_SCAN_ROOTS = (
    _REPO / "src" / "phenotypic",
    _REPO / "docs" / "source",
    _REPO / ".claude" / "skills",
    _REPO / "CLAUDE.md",
)
_SCAN_SUFFIXES = {".py", ".md", ".rst", ".ipynb"}

#: Literal dotted/def forms, plus the string-keyed forms that only failed at
#: runtime during the rename (``ns["category"]``-style class construction was
#: found that way). The string forms are limited to the uppercase property name
#: and to class-namespace keys, because GUI triage and operation-registry code
#: legitimately use a lowercase ``"category"`` key for unrelated concepts.
_LEGACY_API = re.compile(
    r"def category\(|\.category\(\)|\.CATEGORY\b|def CATEGORY\("
    r"|(?:get|has|set)attr\([^)]*[\"']CATEGORY[\"']"
    r"|\bns\[[\"']category[\"']\]"
)

#: Lines that mention the legacy names on purpose.
_ALLOWED = ("hard-renamed", "_LEGACY_FAMILY_NAMES", "legacy ``CATEGORY``")


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        candidates = [root] if root.is_file() else root.rglob("*")
        files.extend(p for p in candidates if p.is_file() and p.suffix in _SCAN_SUFFIXES)
    return sorted(files)


def test_metric_family_is_the_header_prefix() -> None:
    assert SIZE.metric_family() == "Size"
    assert SIZE.AREA.METRIC_FAMILY == "Size"
    assert SIZE.AREA.value == "Size_Area"


def test_base_class_has_no_legacy_category_api() -> None:
    assert not hasattr(MeasurementInfo, "category")
    assert not hasattr(MeasurementInfo, "CATEGORY")


def test_legacy_category_override_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="metric_family"):

        class LEGACY(MeasurementInfo):
            @classmethod
            def category(cls) -> str:
                return "Legacy"

            VALUE = Entry("Value", "A value.")


def test_legacy_category_on_memberless_subclass_is_refused() -> None:
    with pytest.raises(TypeError, match="metric_family"):

        class LEGACY_BASE(MeasurementInfo):
            @classmethod
            def category(cls) -> str:
                return "Legacy"


def test_legacy_CATEGORY_property_override_is_refused() -> None:
    with pytest.raises(TypeError, match="METRIC_FAMILY"):

        class LEGACY_PROP(MeasurementInfo):
            @classmethod
            def metric_family(cls) -> str:
                return "LegacyProp"

            @property
            def CATEGORY(self) -> str:
                return "LegacyProp"

            VALUE = Entry("Value", "A value.")


def test_member_named_CATEGORY_is_refused_as_a_reserved_name() -> None:
    with pytest.raises(TypeError, match="reserved name"):

        class LEGACY_MEMBER(MeasurementInfo):
            @classmethod
            def metric_family(cls) -> str:
                return "LegacyMember"

            CATEGORY = Entry("Category", "A value.")


def test_scan_covers_docs_skills_and_guides() -> None:
    names = {p.name for p in _scanned_files()}
    assert "SKILL.md" in names
    assert "CLAUDE.md" in names
    assert any(p.suffix == ".md" and "docs" in p.parts for p in _scanned_files())


def test_no_legacy_category_api_left_anywhere() -> None:
    offenders = [
        f"{path.relative_to(_REPO)}:{lineno}"
        for path in _scanned_files()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if _LEGACY_API.search(line) and not any(token in line for token in _ALLOWED)
    ]
    assert offenders == []


def test_rst_table_caption_names_the_metric_family() -> None:
    assert ".. list-table:: Metric family: **Size**" in SIZE.rst_table()

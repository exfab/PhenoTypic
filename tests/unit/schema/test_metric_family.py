"""``MeasurementInfo.category()`` was hard-renamed to ``metric_family()``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phenotypic.schema import SIZE, Entry, MeasurementInfo

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
_LEGACY_API = re.compile(r"def category\(|\.category\(\)|\.CATEGORY\b|def CATEGORY\(")


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


def test_no_legacy_category_api_left_in_src() -> None:
    offenders = [
        f"{path.relative_to(_SRC)}:{lineno}"
        for path in sorted(_SRC.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _LEGACY_API.search(line)
    ]
    assert offenders == []


def test_rst_table_caption_names_the_metric_family() -> None:
    assert ".. list-table:: Metric family: **Size**" in SIZE.rst_table()

"""Regression tests for scripts/check_features_md.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_features_md.py"


@pytest.fixture(scope="module")
def validator():
    """Import scripts/check_features_md.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "check_features_md", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_features_md"] = module
    spec.loader.exec_module(module)
    return module


def test_markdown_row_splitter_keeps_escaped_pipe_inside_cell() -> None:
    """The shared row splitter should honor GitHub Markdown escapes."""
    from scripts._markdown_table import split_markdown_row_cells

    cells = split_markdown_row_cells(
        r"Feature | Status | Test ref | ColumnRef \| None"
    )
    assert cells == [
        "Feature",
        "Status",
        "Test ref",
        "ColumnRef | None",
    ]


def test_parse_tables_keeps_escaped_pipe_inside_feature_cell(validator) -> None:
    """Feature ledger cells may document values containing literal pipes."""
    rows, warnings = validator.parse_tables(
        "\n".join(
            [
                "| Feature | Status | Test ref | Notes |",
                "| ------- | ------ | -------- | ----- |",
                r"| Builder | ✅ shipping | tests/unit/gui/test_builder.py | ColumnRef \| None |",
            ]
        )
    )

    assert warnings == []
    assert rows == [
        {
            "Feature": "Builder",
            "Status": "✅ shipping",
            "Test ref": "tests/unit/gui/test_builder.py",
            "Notes": "ColumnRef | None",
        }
    ]


def test_parse_tables_warns_on_malformed_feature_rows(validator) -> None:
    """Malformed feature rows should stay visible to the CI gate."""
    rows, warnings = validator.parse_tables(
        "\n".join(
            [
                "| Feature | Status | Test ref |",
                "| ------- | ------ | -------- |",
                "| Builder | ✅ shipping | tests/unit/gui/test_builder.py | extra |",
            ]
        )
    )

    assert rows == []
    assert len(warnings) == 1
    assert "malformed feature row" in warnings[0]
    assert "4 cells vs 3 headers" in warnings[0]

"""Unit tests for the filter-panel pure helpers (no Dash runtime)."""

from __future__ import annotations

from phenotypic.gui.results_viewer._filter_panel import (
    _blank_row,
    _normalise_spec,
    set_row_compare,
    set_row_method,
    set_row_range,
    set_row_text,
)
from phenotypic.gui.results_viewer._filter_state import (
    METHOD_COMPARE,
    METHOD_IS_ANY_OF,
    METHOD_RANGE,
)


def test_blank_row_has_all_keys_and_defaults() -> None:
    row = _blank_row()
    for key in (
        "id", "column", "method", "values", "range_min", "range_max",
        "compare_op", "compare_value", "text_pattern", "text_regex",
        "text_case_sensitive",
    ):
        assert key in row
    assert row["method"] == METHOD_IS_ANY_OF
    assert row["values"] == []


def test_normalise_spec_backfills_legacy_rows() -> None:
    rows = _normalise_spec([{"column": "a", "values": ["x"]}])
    assert rows[0]["method"] == METHOD_IS_ANY_OF
    assert rows[0]["range_min"] is None
    assert isinstance(rows[0]["id"], str) and rows[0]["id"]


def test_set_row_method_resets_payload() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    rows[0]["values"] = ["keep-me?"]
    rows[0]["range_min"] = 5.0
    out = set_row_method(rows, idx, METHOD_RANGE)
    assert out[0]["method"] == METHOD_RANGE
    assert out[0]["values"] == []          # payload reset
    assert out[0]["range_min"] is None     # payload reset


def test_set_row_range_writes_bounds() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_range(rows, idx, 1.0, 9.0)
    assert out[0]["range_min"] == 1.0 and out[0]["range_max"] == 9.0


def test_set_row_compare_writes_op_and_value() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_compare(rows, idx, ">=", 0.5)
    assert out[0]["method"] == METHOD_COMPARE or out[0]["compare_op"] == ">="
    assert out[0]["compare_op"] == ">=" and out[0]["compare_value"] == 0.5


def test_set_row_text_writes_pattern_and_flags() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_text(rows, idx, "plate", regex=True, case=False)
    assert out[0]["text_pattern"] == "plate"
    assert out[0]["text_regex"] is True
    assert out[0]["text_case_sensitive"] is False


def test_setters_ignore_unknown_idx() -> None:
    rows = [_blank_row()]
    out = set_row_range(rows, "nope", 1.0, 2.0)
    assert out[0]["range_min"] is None

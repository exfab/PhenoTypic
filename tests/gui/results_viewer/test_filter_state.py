"""Unit tests for :mod:`phenotypic.gui.results_viewer._filter_state`.

Exercises the pure-data layer between Dash ``dcc.Store`` payloads and
polars filter expressions: ``FilterRow``, ``FilterSpec.from_store``,
``FilterSpec.to_store``, and ``FilterSpec.apply_to``. The tests build
small in-memory polars frames so they run in milliseconds without any
Dash machinery.
"""

from __future__ import annotations

import logging

import polars as pl

from phenotypic.gui.results_viewer._filter_state import FilterRow, FilterSpec


def _make_frame() -> pl.DataFrame:
    """Return a small frame with two metadata-style columns.

    Two columns ``a`` and ``b`` give us enough to exercise both
    single-column OR-of-values and multi-column AND-across-rows logic.
    """

    return pl.DataFrame(
        {
            "a": ["x", "y", "z", "x"],
            "b": ["1", "2", "1", "3"],
        }
    )


def test_empty_spec_is_passthrough() -> None:
    """An empty ``FilterSpec`` returns the input frame unchanged in row count."""

    df = _make_frame()
    out = FilterSpec(rows=[]).apply_to(df)
    assert out.height == df.height


def test_from_store_handles_none_and_empty_list() -> None:
    """``from_store`` treats ``None`` and ``[]`` identically as no rows."""

    assert FilterSpec.from_store(None).rows == []
    assert FilterSpec.from_store([]).rows == []


def test_from_store_drops_malformed_and_coerces_values() -> None:
    """Entries lacking ``column`` are dropped; integer values cast to str."""

    payload = [
        {"column": "a", "values": [1, 2, "3"]},
        {"values": ["only-values"]},  # missing column → dropped
        "not-a-dict",  # type: ignore[list-item]
        {"column": "b", "values": None},  # None → empty list
    ]
    spec = FilterSpec.from_store(payload)  # type: ignore[arg-type]
    assert len(spec.rows) == 2
    assert spec.rows[0].column == "a"
    assert spec.rows[0].values == ["1", "2", "3"]
    assert spec.rows[1].column == "b"
    assert spec.rows[1].values == []


def test_to_store_round_trip_equals_original() -> None:
    """``from_store(to_store(spec))`` yields a spec with equivalent row contents."""

    original = FilterSpec(
        rows=[
            FilterRow(column="a", values=["x", "y"]),
            FilterRow(column="b", values=["1"]),
        ]
    )
    rebuilt = FilterSpec.from_store(original.to_store())
    assert len(rebuilt.rows) == len(original.rows)
    for src, dst in zip(original.rows, rebuilt.rows, strict=True):
        assert src.column == dst.column
        assert src.values == dst.values


def test_single_column_filter_or_within_values() -> None:
    """A single row keeps rows whose column value is in ``values``."""

    df = _make_frame()
    spec = FilterSpec(rows=[FilterRow(column="a", values=["x", "y"])])
    out = spec.apply_to(df)
    assert sorted(out.get_column("a").to_list()) == ["x", "x", "y"]


def test_multi_column_filter_and_across_rows() -> None:
    """Two rows on different columns must both match (AND semantics)."""

    df = _make_frame()
    spec = FilterSpec(
        rows=[
            FilterRow(column="a", values=["x"]),
            FilterRow(column="b", values=["1"]),
        ]
    )
    out = spec.apply_to(df)
    # Only the first row in `_make_frame()` matches both clauses.
    assert out.height == 1
    assert out.get_column("a").to_list() == ["x"]
    assert out.get_column("b").to_list() == ["1"]


def test_empty_values_list_is_no_op_for_that_row() -> None:
    """A row with ``values=[]`` is skipped; the rest of the frame is preserved."""

    df = _make_frame()
    spec = FilterSpec(rows=[FilterRow(column="a", values=[])])
    out = spec.apply_to(df)
    assert out.height == df.height


def test_empty_column_string_is_no_op_for_that_row() -> None:
    """A row whose ``column`` is the empty string is treated as unset."""

    df = _make_frame()
    spec = FilterSpec(rows=[FilterRow(column="", values=["x"])])
    out = spec.apply_to(df)
    assert out.height == df.height


def test_mismatched_values_reduce_to_zero_rows() -> None:
    """Filtering on a value that no row satisfies produces an empty frame."""

    df = _make_frame()
    spec = FilterSpec(rows=[FilterRow(column="a", values=["nonexistent"])])
    out = spec.apply_to(df)
    assert out.height == 0


def test_missing_column_logs_warning_and_skips(caplog) -> None:
    """An unknown column is skipped with a WARNING log; the frame is unchanged."""

    df = _make_frame()
    spec = FilterSpec(
        rows=[FilterRow(column="ColumnDoesNotExist", values=["x"])]
    )
    with caplog.at_level(
        logging.WARNING, logger="phenotypic.gui.results_viewer._filter_state"
    ):
        out = spec.apply_to(df)
    assert out.height == df.height
    assert any(
        "ColumnDoesNotExist" in record.getMessage()
        and record.levelno == logging.WARNING
        for record in caplog.records
    ), f"expected a WARNING about ColumnDoesNotExist, got: {caplog.records!r}"


def test_string_coercion_uses_polars_default_cast() -> None:
    """Document the observed cast behaviour for numeric columns.

    The implementation casts the column to ``pl.String`` before
    ``is_in``. Polars' default ``Float64 -> String`` cast renders
    ``100.0`` as the literal string ``"100.0"``. So filtering with
    ``["100", "200"]`` (the bare integer-style strings a user typically
    pastes) matches *zero* rows because the cast preserves the trailing
    ``.0``. Filtering with the literal ``["100.0", "200.0"]`` matches.

    This test pins that exact behaviour so a future change is
    intentional.
    """

    df = pl.DataFrame({"Size_Area": [100.0, 200.0, 300.0]})

    no_match = FilterSpec(
        rows=[FilterRow(column="Size_Area", values=["100", "200"])]
    ).apply_to(df)
    assert no_match.height == 0

    matches = FilterSpec(
        rows=[FilterRow(column="Size_Area", values=["100.0", "200.0"])]
    ).apply_to(df)
    assert matches.height == 2
    assert sorted(matches.get_column("Size_Area").to_list()) == [100.0, 200.0]


from phenotypic.gui.results_viewer._filter_state import (
    COMPARE_OPS,
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_IS_NONE_OF,
    METHOD_RANGE,
    _coerce_float,
)
from phenotypic.schema import METADATA


def test_legacy_row_defaults_to_is_any_of() -> None:
    """A pre-method store row keeps working as an is_any_of list filter."""
    spec = FilterSpec.from_store([{"column": "a", "values": ["x"]}])
    assert spec.rows[0].method == METHOD_IS_ANY_OF
    assert spec.rows[0].values == ["x"]


def test_coerce_float_handles_blanks_and_numbers() -> None:
    assert _coerce_float("") is None
    assert _coerce_float(None) is None
    assert _coerce_float("3.5") == 3.5
    assert _coerce_float(7) == 7.0
    assert _coerce_float("not-a-number") is None


def test_from_store_reads_range_and_compare_and_contains() -> None:
    payload = [
        {"column": "Size_Area", "method": METHOD_RANGE,
         "range_min": "100", "range_max": "5000"},
        {"column": "Shape_Circularity", "method": METHOD_COMPARE,
         "compare_op": ">=", "compare_value": "0.85"},
        {"column": str(METADATA.IMAGE_NAME), "method": METHOD_CONTAINS,
         "text_pattern": "plate_02", "text_regex": False,
         "text_case_sensitive": True},
    ]
    spec = FilterSpec.from_store(payload)
    assert spec.rows[0].method == METHOD_RANGE
    assert spec.rows[0].range_min == 100.0 and spec.rows[0].range_max == 5000.0
    assert spec.rows[1].method == METHOD_COMPARE
    assert spec.rows[1].compare_op == ">=" and spec.rows[1].compare_value == 0.85
    assert spec.rows[2].method == METHOD_CONTAINS
    assert spec.rows[2].text_pattern == "plate_02"
    assert spec.rows[2].text_case_sensitive is True


def test_invalid_compare_op_coerced_to_none() -> None:
    spec = FilterSpec.from_store(
        [{"column": "a", "method": METHOD_COMPARE, "compare_op": "~=",
          "compare_value": "1"}]
    )
    assert spec.rows[0].compare_op is None


def test_to_store_round_trips_all_methods() -> None:
    original = FilterSpec.from_store(
        [
            {"column": "Size_Area", "method": METHOD_RANGE, "range_min": 1.0,
             "range_max": None},
            {"column": "n", "method": METHOD_IS_NONE_OF, "values": ["1"]},
        ]
    )
    rebuilt = FilterSpec.from_store(original.to_store())
    assert rebuilt.rows[0].method == METHOD_RANGE
    assert rebuilt.rows[0].range_min == 1.0 and rebuilt.rows[0].range_max is None
    assert rebuilt.rows[1].method == METHOD_IS_NONE_OF
    assert rebuilt.rows[1].values == ["1"]


def test_compare_ops_set_is_ordering_only() -> None:
    assert COMPARE_OPS == frozenset({">", ">=", "<", "<="})


def _numeric_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Size_Area": [50.0, 150.0, 1000.0, 6000.0],
            "name": ["plate_01", "Plate_02", "ctrl_02", "x"],
            "rep": ["1", "2", "3", "4"],
        }
    )


def test_is_none_of_excludes_listed_values() -> None:
    df = _make_frame()
    spec = FilterSpec.from_store(
        [{"column": "a", "method": METHOD_IS_NONE_OF, "values": ["x"]}]
    )
    out = spec.apply_to(df)
    assert "x" not in out.get_column("a").to_list()
    assert out.height == 2  # y, z


def test_range_between_inclusive_optional_bounds() -> None:
    df = _numeric_frame()
    both = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": 100,
          "range_max": 1000}]
    ).apply_to(df)
    assert sorted(both.get_column("Size_Area").to_list()) == [150.0, 1000.0]

    only_min = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": 1000,
          "range_max": None}]
    ).apply_to(df)
    assert sorted(only_min.get_column("Size_Area").to_list()) == [1000.0, 6000.0]

    only_max = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": None,
          "range_max": 150}]
    ).apply_to(df)
    assert sorted(only_max.get_column("Size_Area").to_list()) == [50.0, 150.0]


def test_range_both_bounds_blank_is_no_op() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE}]
    ).apply_to(df)
    assert out.height == df.height


def test_compare_operators() -> None:
    df = _numeric_frame()
    gt = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_COMPARE, "compare_op": ">",
          "compare_value": 150}]
    ).apply_to(df)
    assert sorted(gt.get_column("Size_Area").to_list()) == [1000.0, 6000.0]

    le = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_COMPARE, "compare_op": "<=",
          "compare_value": 150}]
    ).apply_to(df)
    assert sorted(le.get_column("Size_Area").to_list()) == [50.0, 150.0]


def test_contains_literal_case_sensitive_and_insensitive() -> None:
    df = _numeric_frame()
    cs = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "plate",
          "text_regex": False, "text_case_sensitive": True}]
    ).apply_to(df)
    assert cs.get_column("name").to_list() == ["plate_01"]

    ci = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "plate",
          "text_regex": False, "text_case_sensitive": False}]
    ).apply_to(df)
    assert sorted(ci.get_column("name").to_list()) == ["Plate_02", "plate_01"]


def test_contains_regex() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": r"_0\d$",
          "text_regex": True, "text_case_sensitive": True}]
    ).apply_to(df)
    assert sorted(out.get_column("name").to_list()) == ["Plate_02", "ctrl_02", "plate_01"]


def test_contains_blank_pattern_is_no_op() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "  "}]
    ).apply_to(df)
    assert out.height == df.height


def test_invalid_regex_skips_row_without_raising(caplog) -> None:
    df = _numeric_frame()
    spec = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "(",
          "text_regex": True}]
    )
    out = spec.apply_to(df)  # must not raise
    assert out.height == df.height


def test_range_on_mixed_column_drops_non_numeric() -> None:
    df = pl.DataFrame({"mix": ["10", "2", "x"]})
    out = FilterSpec.from_store(
        [{"column": "mix", "method": METHOD_RANGE, "range_min": 1, "range_max": 100}]
    ).apply_to(df)
    assert sorted(out.get_column("mix").to_list()) == ["10", "2"]

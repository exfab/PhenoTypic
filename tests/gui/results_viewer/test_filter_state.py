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

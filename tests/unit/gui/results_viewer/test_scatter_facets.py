"""Facet ordering and the two caps that bound a grid."""

from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._facets import (
    plan_facets,
    sort_facet_values,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec


def test_numeric_looking_values_sort_numerically() -> None:
    """Grid_ColNum is a String column with values 0..11.

    A plain string sort gives 0, 1, 10, 11, 2 -- which renders as a
    scrambled grid and reads like a rendering bug rather than a sort bug.
    """
    assert sort_facet_values(["10", "2", "0", "11", "1"]) == [
        "0",
        "1",
        "2",
        "10",
        "11",
    ]


def test_non_numeric_values_sort_lexically() -> None:
    assert sort_facet_values(["b", "a", "c"]) == ["a", "b", "c"]


def test_mixed_values_fall_back_to_lexical() -> None:
    """If any value fails to parse, every value sorts as a string."""
    assert sort_facet_values(["10", "a", "2"]) == ["10", "2", "a"]


def test_the_grid_is_capped_by_the_product_not_per_axis() -> None:
    """A 12x12 selection is 144 panels; no context budget survives it."""
    df = pl.DataFrame(
        {
            "r": [str(i) for i in range(12) for _ in range(12)],
            "c": [str(j) for _ in range(12) for j in range(12)],
        }
    )
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec, cap=16)

    assert len(plan.rows) * len(plan.cols) <= 16
    assert plan.truncated is True
    assert plan.total == 144


def test_an_uncapped_grid_is_not_marked_truncated() -> None:
    df = pl.DataFrame({"r": ["0", "1"], "c": ["0", "1"]})
    spec = FigureSpec(x_col="x", y_col="y", row_col="r", col_col="c")
    plan = plan_facets(df, spec, cap=16)
    assert plan.truncated is False and plan.total == 4

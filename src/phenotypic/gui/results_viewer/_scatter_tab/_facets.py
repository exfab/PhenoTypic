"""Facet planning: which values become rows and columns, in what order."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from phenotypic.gui._config import SCATTER_FACET_CAP
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

#: The column name the derived frame index is written to.
COMPUTED_FRAME_INDEX = "Computed_FrameIndex"


@dataclass(frozen=True)
class FacetPlan:
    """The grid a figure will draw.

    Args:
        rows: Ordered row values, already capped.
        cols: Ordered column values, already capped.
        truncated: Whether the cap removed any panel.
        total: Panels the uncapped selection would have produced.
    """

    rows: list[str]
    cols: list[str]
    truncated: bool
    total: int


def sort_facet_values(values: list[str]) -> list[str]:
    """Order facet values numerically when every value parses.

    Grid and many metadata columns are ``String`` even when their values
    are numbers, so a plain sort orders ``Grid_ColNum`` as 0, 1, 10, 11,
    2. Falls back to a lexical sort the moment any value is non-numeric,
    so a mixed column is ordered consistently rather than half
    numerically.

    Args:
        values: Distinct facet values as strings.

    Returns:
        The same values, ordered.
    """
    try:
        return sorted(values, key=lambda v: (float(v), v))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def plan_facets(
    df: pl.DataFrame, spec: FigureSpec, cap: int = SCATTER_FACET_CAP
) -> FacetPlan:
    """Choose the grid's rows and columns, capped by their product.

    The cap bounds ``rows * cols``, not either axis alone: a 12-value row
    axis crossed with a 12-value column axis is 144 panels. Over-cap keeps
    the first panels in facet-value order -- deterministic, and
    independent of how the data happens to be distributed -- and flags
    ``truncated`` so the caller can surface "showing first N of M" rather
    than silently dropping panels.

    Args:
        df: The plottable frame.
        spec: The figure's configuration.
        cap: Maximum number of panels.

    Returns:
        A :class:`FacetPlan`.
    """

    def _values(col: str | None) -> list[str]:
        # The [""] fallback is load-bearing, not defensive: an all-null
        # column yields [] from drop_nulls().unique(), and an empty axis
        # would collapse the grid to zero panels rather than one.
        if col is None or col not in df.columns:
            return [""]
        raw = df[col].drop_nulls().unique().cast(pl.String).to_list()
        return sort_facet_values(raw) or [""]

    rows, cols = _values(spec.row_col), _values(spec.col_col)
    total = len(rows) * len(cols)
    if total <= cap:
        return FacetPlan(rows=rows, cols=cols, truncated=False, total=total)

    kept_rows = rows
    kept_cols = cols
    while len(kept_rows) * len(kept_cols) > cap:
        if len(kept_rows) >= len(kept_cols) and len(kept_rows) > 1:
            kept_rows = kept_rows[:-1]
        elif len(kept_cols) > 1:
            kept_cols = kept_cols[:-1]
        else:
            break
    return FacetPlan(
        rows=kept_rows, cols=kept_cols, truncated=True, total=total
    )


def derive_frame_index(
    df: pl.DataFrame,
    plate_col: str = "Metadata_PlateID",
    time_col: str = "Metadata_ImageDatetime",
) -> pl.DataFrame:
    """Rank each image chronologically within its plate, zero-based.

    Needed because ``Metadata_FrameIndex`` is often unpopulated and
    ``Metadata_Timepoint`` can be a constant. Ranks distinct timestamps,
    so every colony in one image shares a frame. Rows with a null
    timestamp get a null index and are excluded from the plot rather than
    ranked zero -- the verification fixture has 81 of them.

    Args:
        df: A plottable frame.
        plate_col: Column identifying the plate.
        time_col: Column carrying the capture timestamp.

    Returns:
        ``df`` with a ``Computed_FrameIndex`` Int32 column appended.
    """
    if plate_col not in df.columns or time_col not in df.columns:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias(COMPUTED_FRAME_INDEX)
        )
    ranked = (
        df.select([plate_col, time_col])
        .unique()
        .drop_nulls()
        .sort([plate_col, time_col])
        .with_columns(
            pl.col(time_col)
            .cum_count()
            .over(plate_col)
            .sub(1)
            .cast(pl.Int32)
            .alias(COMPUTED_FRAME_INDEX)
        )
    )
    return df.join(ranked, on=[plate_col, time_col], how="left")

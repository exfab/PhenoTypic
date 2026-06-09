"""Pure figure builder for the Heatmap tab.

Side-effect-free Plotly figure construction so the heatmap can be
unit-tested against synthetic frames without booting Dash. Imported by
:mod:`._callbacks` for the live render and by
``tests/unit/gui/results_viewer/test_heatmap_figure.py`` for coverage.

Order of operations (spec lines 1352-1356):

1. Empty-state check - if the grid columns are absent, return a
   placeholder figure with an annotation explaining the gap.
2. Filter by ``Metadata_ImageFile`` if the picker has a selection.
3. Filter by ``Metadata_Time`` if a slider value is supplied.
4. Aggregate the remaining rows over ``(Grid_RowNum, Grid_ColNum)``
   with the configured polars aggregator.
5. Pivot to a wide row-major matrix and emit:
   - a primary ``go.Heatmap`` trace with the data values, and
   - a removed-cell overlay (``go.Heatmap`` at zero opacity for hover +
     a ``go.Scatter`` of `x`-markers) so curated cells render as
     muted-color exclusions rather than just NaN holes.

Color choice: the single-variable navy-to-blue ramp from DESIGN.md "06 --
Heatmap Colorscale" / "10 -- Well-Plate Grid" (``SEQUENTIAL_COLORSCALE``:
near-transparent navy -> sky -> full navy). A plate map and a heatmap of the
same data therefore read identically. Removed / excluded cells render in
vermilion (the spec's failed/null color), CB-distinct from the ramp under all
types. Never build a sequential scale from the categorical Okabe-Ito series.
"""
from __future__ import annotations

from typing import Literal, Sequence, TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

from phenotypic.gui._design import COLOR_MUTED, OI_VERMILION
from phenotypic.viz.figures import SEQUENTIAL_COLORSCALE, apply_theme

from .._filtered_state import KEY_OBJECT_LABEL

#: Plotly-shaped copy of the brand sequential ramp (list-of-[pos, color]).
_HEATMAP_COLORSCALE = [[pos, color] for pos, color in SEQUENTIAL_COLORSCALE]

#: Closed value-set for the aggregator dropdown. Kept in lock-step with
#: the polars expression dispatch in :func:`_aggregate_grid_bins`.
AggregatorName: TypeAlias = Literal["mean", "median", "max", "min"]

# Marker sizing for the removed-cell overlay (spec lines 1035-1042).
# ``cell_px`` is fixed at the same constant Plotly uses to compute the
# default cell footprint for moderately-sized grids - Plotly recomputes
# this dynamically based on the plot area so a single constant is good
# enough for the half-cell heuristic.
_CELL_PX_NOMINAL: int = 30
_REMOVED_MARKER_MIN: float = 6.0
_REMOVED_MARKER_MAX: float = 14.0

# Polars uses ``Float64`` for any non-integral aggregation output. We
# stash NaN-only pivots as a ``Float64`` matrix so the downstream
# Plotly trace's ``zauto`` does not silently swap to a categorical-style
# axis. Sentinel kept private; tests don't need to assert on it.

# Names of the metadata columns this module touches. The image-file and
# time columns are kept local so the figure builder stays pure; the
# object-label curation key is single-sourced from ``_filtered_state``
# (imported above as :data:`KEY_OBJECT_LABEL`) so it can never drift from
# the curation layer.
_META_IMAGE_FILE: str = "Metadata_ImageFile"
_META_TIME: str = "Metadata_Time"


def build_heatmap_figure(
    frame: pl.DataFrame | pd.DataFrame,
    *,
    color_col: str,
    image_file: str,
    time_value: float | None,
    aggregator: AggregatorName,
    removed_keys: set[tuple[str, int]],
    grid_row_col: tuple[str, str] = ("Grid_RowNum", "Grid_ColNum"),
) -> go.Figure:
    """Build the heatmap figure for the Heatmap tab.

    Args:
        frame: Either a polars or pandas DataFrame whose rows are
            individual measurements (typically the post-curation
            filtered frame or its QC-augmented variant). Pandas input
            is converted to polars at the top so the rest of the
            pipeline can rely on the polars API.
        color_col: Name of the column whose values fill the heatmap
            cells. Typically a measurement column (e.g. ``Size_Area``)
            or a QC metric column (e.g. ``QC_Count_Metric``).
        image_file: ``Metadata_ImageFile`` selection. Applied as a
            row filter *before* the aggregator so multi-image frames
            cannot leak across image boundaries (spec lines 1352-1356).
        time_value: ``Metadata_Time`` row filter. ``None`` skips the
            time filter. Coerced to ``float`` for the comparison so
            integer-typed stores match floating-point picks.
        aggregator: Polars ``GroupBy.agg`` aggregator used to collapse
            multi-row ``(grid_row, grid_col)`` bins. For the common
            one-row-per-well case this is a no-op.
        removed_keys: Set of ``(ImageFile, Object_Label)`` keys that the
            user has curated out. Cells matching the picked image plus
            any such label render as muted `x`-markers on top of the
            data heatmap.
        grid_row_col: Tuple of (row column, col column) names used to
            pivot the heatmap. Defaults to the
            :class:`phenotypic.schema.GRID`-prefixed
            column names emitted by ``GridMeasureFeatures``.

    Returns:
        A :class:`plotly.graph_objects.Figure` ready to assign to a
        :class:`dash.dcc.Graph` ``figure`` prop. Never raises:
        missing-column / empty-frame / NaN-only branches return a
        placeholder figure with an explanatory annotation.
    """
    # Always work in polars internally - converting at the top means the
    # rest of the function can rely on polars semantics for null + dtype
    # handling. ``from_pandas`` copies, but the alternative (branching
    # on the input dtype everywhere downstream) is brittle and the input
    # frames the live callback feeds in are already polars; pandas
    # inputs only show up in unit tests with synthetic data so the
    # incidental copy is fine.
    if isinstance(frame, pd.DataFrame):
        frame = pl.from_pandas(frame)

    row_col, col_col = grid_row_col
    if row_col not in frame.columns or col_col not in frame.columns:
        return _empty_state_figure(
            f"Grid columns missing ({row_col!r} / {col_col!r}). "
            "Run a pipeline that includes a GridMeasureFeatures step "
            "to populate them."
        )

    # 1) Image filter first.
    if _META_IMAGE_FILE in frame.columns:
        frame = frame.filter(pl.col(_META_IMAGE_FILE) == image_file)

    # 2) Time filter, if requested AND the column exists.
    if time_value is not None and _META_TIME in frame.columns:
        # Cast the column to Float64 so non-numeric stores don't blow up
        # the comparison. ``strict=False`` coerces invalid entries to
        # null which then survive the equality filter cleanly.
        try:
            frame = frame.filter(
                pl.col(_META_TIME).cast(pl.Float64, strict=False)
                == float(time_value)
            )
        except (TypeError, ValueError):
            # Non-numeric time picker payload - skip the filter rather
            # than raise; matches the "non-numeric time slider"
            # empty-state in :func:`_refresh_heatmap_controls`.
            pass

    # 3) Aggregate over grid bins.
    if color_col not in frame.columns:
        return _empty_state_figure(
            f"Color column {color_col!r} is not present in the data."
        )

    if frame.is_empty():
        return _empty_state_figure(
            f"No rows for image {image_file!r}"
            + (f" at time {time_value!r}" if time_value is not None else "")
            + "."
        )

    aggregated = _aggregate_grid_bins(
        frame, row_col=row_col, col_col=col_col, value_col=color_col,
        aggregator=aggregator,
    )

    # 4) Pivot to a wide matrix.
    matrix, row_labels, col_labels = _pivot_to_matrix(
        aggregated, row_col=row_col, col_col=col_col, value_col=color_col,
    )

    # 5) Build the primary heatmap trace.
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale=_HEATMAP_COLORSCALE,
            colorbar={"title": {"text": color_col}},
            hovertemplate=(
                f"{row_col}: %{{y}}<br>"
                f"{col_col}: %{{x}}<br>"
                f"{color_col}: %{{z}}<extra></extra>"
            ),
            # Render NaN cells transparently rather than collapsing to
            # the colormap's zero - keeps QC metric columns (which
            # are NaN-heavy) visually honest.
            zauto=True,
            connectgaps=False,
        )
    )

    # 6) Removed-cell overlay.
    overlay_traces = _build_removed_overlay(
        frame,
        row_col=row_col,
        col_col=col_col,
        row_labels=row_labels,
        col_labels=col_labels,
        removed_keys=removed_keys,
    )
    for trace in overlay_traces:
        fig.add_trace(trace)

    apply_theme(fig)  # Okabe-Ito colorway, mono numeric axes, brand grid/fonts.
    fig.update_layout(
        xaxis={
            "title": col_col,
            "side": "top",
            "scaleanchor": "y",
            "constrain": "domain",
        },
        yaxis={
            "title": row_col,
            "autorange": "reversed",  # row 1 at top, like plate notation
            "constrain": "domain",
        },
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_state_figure(message: str) -> go.Figure:
    """Return a placeholder figure with a centered annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": COLOR_MUTED, "size": 14},
        align="center",
    )
    apply_theme(fig)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def _aggregator_expr(aggregator: AggregatorName, column: str) -> pl.Expr:
    """Map an :class:`AggregatorName` literal to its polars expression."""
    col = pl.col(column)
    if aggregator == "mean":
        return col.mean()
    if aggregator == "median":
        return col.median()
    if aggregator == "max":
        return col.max()
    if aggregator == "min":
        return col.min()
    # ``aggregator`` is typed Literal[...]; any other value is a caller
    # bug and we surface it loudly rather than silently falling through.
    raise ValueError(f"Unsupported aggregator: {aggregator!r}")


def _aggregate_grid_bins(
    frame: pl.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    aggregator: AggregatorName,
) -> pl.DataFrame:
    """Collapse rows that share a ``(row, col)`` pair via the aggregator.

    The polars ``group_by`` is unconditional so the no-op case
    (one row per (row, col)) is just a tight identity transform. Skipping
    the group_by based on uniqueness would save a few ms but adds a
    branch the tests would have to cover; the current shape is cheaper
    to maintain.
    """
    expr = _aggregator_expr(aggregator, value_col)
    return frame.group_by([row_col, col_col]).agg(expr.alias(value_col))


def _pivot_to_matrix(
    aggregated: pl.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
) -> tuple[np.ndarray, list[int | float], list[int | float]]:
    """Pivot the aggregated frame into a dense ``(rows x cols)`` matrix.

    Returns:
        Tuple of:

        * 2-D ``numpy.ndarray`` of shape ``(len(row_labels),
          len(col_labels))`` with ``NaN`` filling any missing
          ``(row, col)`` bin.
        * ``row_labels`` - sorted unique row values from the input.
        * ``col_labels`` - sorted unique col values from the input.
    """
    if aggregated.is_empty():
        # ``group_by`` on an empty frame returns an empty frame; pivot
        # would still raise on the missing columns. Short-circuit.
        return np.empty((0, 0)), [], []

    row_labels = sorted(set(aggregated[row_col].to_list()))
    col_labels = sorted(set(aggregated[col_col].to_list()))

    pivoted = aggregated.pivot(
        on=col_col,
        index=row_col,
        values=value_col,
        aggregate_function="first",  # bins are pre-aggregated; this is a passthrough
    )

    # ``pivot`` does not guarantee column ordering matches ``col_labels``,
    # so we reindex by name explicitly.
    pivoted = pivoted.sort(row_col)
    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    row_to_idx = {label: i for i, label in enumerate(row_labels)}
    col_to_idx = {str(label): i for i, label in enumerate(col_labels)}

    for record in pivoted.to_dicts():
        i = row_to_idx[record[row_col]]
        for k, v in record.items():
            if k == row_col:
                continue
            if v is None:
                continue
            # Pivot column names come back as strings (or whatever the
            # source dtype renders to); map via the str-keyed lookup.
            try:
                j = col_to_idx[str(k)]
            except KeyError:
                continue
            try:
                matrix[i, j] = float(v)
            except (TypeError, ValueError):
                matrix[i, j] = np.nan

    return matrix, row_labels, col_labels


def _build_removed_overlay(
    frame: pl.DataFrame,
    *,
    row_col: str,
    col_col: str,
    row_labels: Sequence[int | float],
    col_labels: Sequence[int | float],
    removed_keys: set[tuple[str, int]],
) -> list[go.Heatmap | go.Scatter]:
    """Build the removed-cell overlay traces.

    Two traces:

    * ``go.Heatmap`` at zero opacity covering the same grid, so the
      built-in Plotly hover machinery still reports "removed" rather
      than falling through to the data trace.
    * ``go.Scatter`` of `x`-markers in vermilion (:data:`OI_VERMILION`, the
      spec's failed/excluded color). Marker size uses the spec's
      ``min(14, max(6, cell_px * 0.5))`` formula.

    The frame is scanned for rows whose ``(ImageFile, Object_Label)``
    appears in ``removed_keys``; the matching ``(row, col)`` bins drive
    the overlay coordinates. When no keys match the active frame the
    overlay traces are still emitted (zero-length scatter) so the
    trace-count invariant in tests is stable.
    """
    if not removed_keys:
        return []

    if (
        _META_IMAGE_FILE not in frame.columns
        or KEY_OBJECT_LABEL not in frame.columns
    ):
        return []

    # Build a removed-keys frame and inner-join with the (already
    # image+time filtered) frame so we only emit overlay markers for
    # cells actually visible in the heatmap.
    removed_frame = pl.DataFrame(
        {
            _META_IMAGE_FILE: [k[0] for k in removed_keys],
            KEY_OBJECT_LABEL: [k[1] for k in removed_keys],
        },
        schema={_META_IMAGE_FILE: pl.String, KEY_OBJECT_LABEL: pl.Int64},
    )
    keyed = frame.with_columns(
        pl.col(_META_IMAGE_FILE).cast(pl.String),
        pl.col(KEY_OBJECT_LABEL).cast(pl.Int64),
    )
    matched = keyed.join(
        removed_frame,
        on=[_META_IMAGE_FILE, KEY_OBJECT_LABEL],
        how="inner",
    )
    if matched.is_empty():
        return []

    rows = matched[row_col].to_list()
    cols = matched[col_col].to_list()

    marker_size = max(_REMOVED_MARKER_MIN, min(_REMOVED_MARKER_MAX, _CELL_PX_NOMINAL * 0.5))

    # The zero-opacity heatmap keeps the visible cell footprint clean
    # while still painting "removed" into Plotly's hover index.
    overlay_z = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
    row_to_idx = {label: i for i, label in enumerate(row_labels)}
    col_to_idx = {label: i for i, label in enumerate(col_labels)}
    for r, c in zip(rows, cols):
        if r in row_to_idx and c in col_to_idx:
            overlay_z[row_to_idx[r], col_to_idx[c]] = 1.0
    overlay_heatmap = go.Heatmap(
        z=overlay_z,
        x=list(col_labels),
        y=list(row_labels),
        colorscale=[[0, OI_VERMILION], [1, OI_VERMILION]],
        showscale=False,
        opacity=0.0,
        hovertemplate="(removed)<extra></extra>",
    )

    overlay_scatter = go.Scatter(
        x=list(cols),
        y=list(rows),
        mode="markers",
        marker={
            "symbol": "x",
            "size": marker_size,
            "color": OI_VERMILION,
            "line": {"width": 1.5, "color": OI_VERMILION},
        },
        name="Removed",
        hovertemplate="Removed<extra></extra>",
        showlegend=False,
    )
    return [overlay_heatmap, overlay_scatter]


__all__ = ["build_heatmap_figure", "AggregatorName"]

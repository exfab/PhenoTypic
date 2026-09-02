"""Pure Plotly figure construction for the Scatter tab.

Side-effect free and Dash-free so it can be unit-tested against synthetic
frames without booting a server, following ``_heatmap_tab/_figure.py``.

Three properties of this module are correctness requirements rather than
polish, and none of them announces itself when broken:

* **The trace type is switched for export.** WebGL export through kaleido
  is environment-dependent: with byte-identical inputs it produced 624
  non-white pixels on one compute node -- the count for a figure with
  *zero* traces -- and 47,549 on two others, against 46,886 for SVG
  everywhere. No warning, exit code 0, a well-formed PDF either way, and
  the mechanism behind the blank node is unidentified. A renderer that
  works on most machines is worse than one that fails on all of them,
  because nothing signals which one you got. The screen keeps gl, because
  SVG cannot draw this project's point counts and a live figure that
  renders wrong is visible immediately.
* **``CUSTOMDATA_COL`` is carried through, never recomputed.** The value
  is a positional index into ``OutputRoot.master_df``, while the frame
  this builder receives is a filtered, re-sorted slice of it. Deriving the
  index here instead resolves every click to a real but wrong colony.
* **Sharing is expressed by the ``share_axes`` branch alone.**
  ``update_yaxes(range=...)`` writes to every axis regardless of what
  ``shared_yaxes`` was passed to ``make_subplots`` (measured), so removing
  the branch shares every figure and nothing raises.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from phenotypic.gui._design import OI_GREY, OKABE_ITO
from phenotypic.gui.results_viewer._scatter_tab._facets import (
    FacetPlan,
    sort_facet_values,
)
from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec

#: Column carrying each point's positional index into ``master_df``.
CUSTOMDATA_COL = "_scatter_row_index"

#: Boolean column marking a colony the user has removed by curation.
#:
#: Joined into the frame in memory at render time by the callback layer,
#: from ``STORE_REMOVED_KEYS`` -- it is not a measurement and never
#: reaches disk. Underscore-prefixed like ``CUSTOMDATA_COL`` so it cannot
#: collide with a measurement header. Absent when no curation state
#: exists, and this module treats absence as "nothing is removed".
REMOVED_COL = "_scatter_removed"

#: Legend entry for the curation series. One entry for the whole figure,
#: not one per facet.
REMOVED_LABEL = "removed by curation"

#: Symbol for removed colonies. Also ``_SYMBOLS[4]``, so a shape channel
#: with five values draws its fifth as an x too -- the grey is what
#: separates them, which is why the colour is not also shared.
_REMOVED_SYMBOL = "x"

#: Colours for the hue channel, in DESIGN.md "06" series order.
#:
#: Sliced to the six *categorical* series deliberately: ``OKABE_ITO``
#: index 6 is vermilion, which ``_design.py`` and DESIGN.md reserve for
#: error/alert and forbid as a data series. Spec 9 caps the hue control at
#: eight values, so seven or eight hues reuse a colour here -- shape is a
#: separate channel and does not rescue it. The reuse is the lesser fault:
#: it is a legibility limit the palette itself is reporting, where
#: borrowing the alert colour would make a normal series read as a failure.
_SERIES_COLORS: tuple[str, ...] = OKABE_ITO[:6]

#: Marker symbols in the order the shape channel consumes them.
_SYMBOLS: tuple[str, ...] = ("circle", "square", "triangle-up", "diamond", "x")

#: Fraction of the data span added to each end of a shared axis.
_AXIS_PAD_FRACTION = 0.05

#: Separator between the parts of a facet title or a legend label.
_LABEL_JOIN = " · "


def _grouping_column(df: pl.DataFrame, col: str | None) -> str | None:
    """Return ``col`` when it can actually group, else None.

    Distinguishes "this role is unset" from "this role names a column
    whose every value is null". Both make :func:`plan_facets` return
    ``[""]``, but only the first means "draw one panel holding
    everything" -- and neither may be inferred from the *value* ``""``,
    which a real metadata column is free to contain.

    Args:
        df: The frame being drawn.
        col: A column name, or None when the role is unset.

    Returns:
        The column name when it exists and holds a non-null value, else
        None.
    """
    if col is None or col not in df.columns:
        return None
    return col if df[col].drop_nulls().len() > 0 else None


def _series_values(df: pl.DataFrame, col: str | None) -> list[str | None]:
    """Distinct values of a legend channel, ordered, or ``[None]``.

    ``[None]`` means "one unsplit series", which is what an unset role and
    an all-null column both come to.

    Args:
        df: The frame being drawn.
        col: The hue or shape column, already resolved, or None.

    Returns:
        Ordered string values, or ``[None]`` for a single series.
    """
    if col is None:
        return [None]
    values = df[col].drop_nulls().unique().cast(pl.String).to_list()
    ordered: list[str | None] = list(sort_facet_values(values))
    return ordered or [None]


def _axis_range(df: pl.DataFrame, col: str) -> tuple[float, float] | None:
    """Padded (min, max) over the whole frame, or None if not numeric.

    Args:
        df: The frame being drawn.
        col: The column to measure.

    Returns:
        A padded ``(low, high)`` pair, or None when the column is absent,
        non-numeric, or holds no finite value.
    """
    if col not in df.columns:
        return None
    series = df[col]
    if not (series.dtype.is_numeric() or series.dtype == pl.String):
        return None
    try:
        # Utf8 -> Float64 with strict=False yields nulls for unparseable
        # values (measured, polars 1.41). The except is narrow on purpose:
        # a bare `except Exception` would turn a real error into a silent
        # "no shared range", which reads as a rendering choice.
        values = series.cast(pl.Float64, strict=False).drop_nulls()
    except pl.exceptions.PolarsError:
        return None
    values = values.filter(values.is_finite())
    low, high = values.min(), values.max()
    # Real narrowing rather than a `type: ignore`. `Series.min()` is typed
    # as any Python literal or None -- None on an empty series, and a
    # `str`/`date` on a series of some other dtype. The isinstance is what
    # proves to both the reader and the checker that this one is neither,
    # and it subsumes the emptiness check.
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    lo, hi = float(low), float(high)
    pad = (hi - lo) * _AXIS_PAD_FRACTION or 1.0
    return (lo - pad, hi + pad)


def _join_label(*parts: str) -> str:
    """Join the non-empty parts of a title or legend label."""
    return _LABEL_JOIN.join(p for p in parts if p)


def _split_on_curation(
    df: pl.DataFrame, *, show_removed: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate live colonies from ones removed by curation.

    The flag column is optional and Boolean. A frame without it -- any
    caller holding no curation state, which is every caller before the
    Scatter tab was wired up -- comes back unsplit, so absence changes
    nothing. A non-Boolean column is read as absent rather than coerced:
    the column is produced in memory by the callback layer, so a wrong
    dtype is a defect in the producer, and guessing at it would hide
    colonies on the strength of a guess.

    Args:
        df: The plottable frame.
        show_removed: Whether removed colonies are drawn at all.

    Returns:
        ``(live, removed)``. ``removed`` is empty when the flag column is
        absent or when ``show_removed`` is False -- in the second case the
        rows are dropped from both halves, which is what the toggle means.
    """
    empty = df.clear()
    if df.schema.get(REMOVED_COL) != pl.Boolean:
        return df, empty
    flag = pl.col(REMOVED_COL).fill_null(False)
    live = df.filter(~flag)
    return live, (df.filter(flag) if show_removed else empty)


def _filter_cell(
    frame: pl.DataFrame,
    row_col: str | None,
    row_value: str,
    col_col: str | None,
    col_value: str,
) -> pl.DataFrame:
    """Narrow a frame to one facet cell.

    Args:
        frame: The frame to narrow.
        row_col: The resolved facet-row column, or None to not filter.
        row_value: The row value this cell draws.
        col_col: The resolved facet-column column, or None to not filter.
        col_value: The column value this cell draws.

    Returns:
        The rows belonging to this cell.
    """
    if row_col is not None:
        frame = frame.filter(pl.col(row_col).cast(pl.String) == row_value)
    if col_col is not None:
        frame = frame.filter(pl.col(col_col).cast(pl.String) == col_value)
    return frame


def _marker_trace(
    trace_cls: type[go.Scatter] | type[go.Scattergl],
    part: pl.DataFrame,
    spec: FigureSpec,
    *,
    label: str,
    showlegend: bool,
    color: str,
    symbol: str,
) -> go.Scatter | go.Scattergl:
    """Build one marker series.

    Args:
        trace_cls: ``go.Scatter`` for export, ``go.Scattergl`` for screen.
        part: The rows this series draws.
        spec: Marker sizing and the x/y columns.
        label: Series name, shared with its legend group.
        showlegend: Whether this series carries the legend entry.
        color: Marker fill.
        symbol: Marker symbol.

    Returns:
        A configured trace, not yet added to a figure.
    """
    return trace_cls(
        x=part[spec.x_col].to_list(),
        y=part[spec.y_col].to_list(),
        mode="markers",
        name=label,
        legendgroup=label,
        showlegend=showlegend,
        # Carried through verbatim: this indexes master_df, not this slice.
        customdata=[[i] for i in part[CUSTOMDATA_COL].to_list()],
        marker=dict(
            size=spec.marker_size,
            opacity=spec.marker_opacity,
            color=color,
            symbol=symbol,
            line=dict(width=0),
        ),
    )


def _subplot_titles(
    spec: FigureSpec,
    plan: FacetPlan,
    row_col: str | None,
    col_col: str | None,
) -> list[str]:
    """One title per cell, naming whichever facet roles are in use.

    Args:
        spec: The figure's configuration.
        plan: The grid being drawn.
        row_col: The resolved facet-row column, or None.
        col_col: The resolved facet-column column, or None.

    Returns:
        Titles in row-major order, one per cell.
    """
    return [
        _join_label(
            f"{spec.row_col}={r_val}" if row_col else "",
            f"{spec.col_col}={c_val}" if col_col else "",
        )
        for r_val in plan.rows or [""]
        for c_val in plan.cols or [""]
    ]


def build_scatter_figure(
    df: pl.DataFrame,
    spec: FigureSpec,
    plan: FacetPlan,
    *,
    for_export: bool = False,
) -> go.Figure:
    """Build one section's faceted scatter figure.

    Args:
        df: The plottable frame for ONE section, carrying
            ``CUSTOMDATA_COL`` and optionally the Boolean ``REMOVED_COL``.
            Without ``REMOVED_COL`` nothing is treated as removed.
        spec: Roles, sizes and scales. ``show_removed`` decides whether
            curation-removed colonies draw as a grey x series or are
            dropped; with no ``REMOVED_COL`` it has no effect.
        plan: The facet grid to draw.
        for_export: When True, use SVG ``go.Scatter`` traces. kaleido
            renders ``Scattergl`` as blank axes with exit code 0 and no
            warning, so the export pass MUST substitute the trace type.

    Returns:
        A ``plotly.graph_objects.Figure`` with one trace per non-empty
        (facet cell, hue, shape) combination, plus at most one grey-x
        curation series per cell.
    """
    trace_cls = go.Scatter if for_export else go.Scattergl
    n_rows, n_cols = max(len(plan.rows), 1), max(len(plan.cols), 1)

    live, removed = _split_on_curation(df, show_removed=spec.show_removed)
    # Everything the axes and the legend channels are derived from is the
    # data that will actually be drawn: hiding removed colonies must not
    # leave the shared range sized for points nobody can see.
    drawn = pl.concat([live, removed]) if removed.height else live

    # Whether a role filters at all is a property of the COLUMN, never of
    # the facet value: `plan_facets` uses "" to mean "no values", but "" is
    # also a value a metadata column may hold. Reading the value as the
    # sentinel makes that panel draw the whole frame.
    row_col = _grouping_column(drawn, spec.row_col)
    col_col = _grouping_column(drawn, spec.col_col)
    # Hue and shape are read off the LIVE rows: a removed colony is not a
    # member of any hue series, so a hue value carried only by removed rows
    # must not open an empty legend entry.
    hue_col = _grouping_column(live, spec.hue_col)
    shape_col = _grouping_column(live, spec.shape_col)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=spec.share_axes,
        shared_yaxes=spec.share_axes,
        subplot_titles=_subplot_titles(spec, plan, row_col, col_col),
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    hues = _series_values(live, hue_col)
    shapes = _series_values(live, shape_col)

    # Track which series already have a legend entry. Keying on the series
    # rather than on "is this the first cell" is what stops a hue absent
    # from cell (0,0) from vanishing out of the legend on a sparse frame --
    # the common case at 23 strains over 36 images, not a corner.
    legended: set[str] = set()
    for r_i, r_val in enumerate(plan.rows or [""], start=1):
        for c_i, c_val in enumerate(plan.cols or [""], start=1):

            cell = _filter_cell(live, row_col, r_val, col_col, c_val)

            # Drawn first so live colonies sit on top of the exclusions
            # rather than under them.
            removed_cell = _filter_cell(
                removed, row_col, r_val, col_col, c_val
            )
            if removed_cell.height:
                fig.add_trace(
                    _marker_trace(
                        trace_cls,
                        removed_cell,
                        spec,
                        label=REMOVED_LABEL,
                        showlegend=REMOVED_LABEL not in legended,
                        color=OI_GREY,
                        symbol=_REMOVED_SYMBOL,
                    ),
                    row=r_i,
                    col=c_i,
                )
                legended.add(REMOVED_LABEL)

            for h_i, hue in enumerate(hues):
                for s_i, shape in enumerate(shapes):
                    part = cell
                    if hue_col is not None and hue is not None:
                        part = part.filter(
                            pl.col(hue_col).cast(pl.String) == hue
                        )
                    if shape_col is not None and shape is not None:
                        part = part.filter(
                            pl.col(shape_col).cast(pl.String) == shape
                        )
                    if part.height == 0:
                        continue
                    label = _join_label(
                        f"{spec.hue_col}={hue}" if hue is not None else "",
                        f"{spec.shape_col}={shape}"
                        if shape is not None
                        else "",
                    ) or spec.y_col
                    fig.add_trace(
                        _marker_trace(
                            trace_cls,
                            part,
                            spec,
                            label=label,
                            showlegend=label not in legended,
                            # Indexed by the GLOBAL hue order, not by a
                            # per-cell counter: a hue absent from one facet
                            # would otherwise shift every colour after it,
                            # and the legend would describe a different
                            # figure from the one on screen.
                            color=_SERIES_COLORS[h_i % len(_SERIES_COLORS)],
                            symbol=_SYMBOLS[s_i % len(_SYMBOLS)],
                        ),
                        row=r_i,
                        col=c_i,
                    )
                    legended.add(label)

    # This branch is what `share_axes` MEANS -- see the module docstring.
    if spec.share_axes:
        x_rng = _axis_range(drawn, spec.x_col)
        y_rng = _axis_range(drawn, spec.y_col)
        if x_rng:
            fig.update_xaxes(range=list(x_rng))
        if y_rng:
            fig.update_yaxes(range=list(y_rng))

    legend: dict[str, object] = {"font": {"size": spec.sizes["legend"]}}
    if for_export:
        # Spec 9: on export the floating legend leaves its corner and lays
        # out along the bottom of every page.
        legend.update(
            orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5
        )
    fig.update_layout(
        font=dict(size=spec.sizes["axis"]),
        legend=legend,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = spec.sizes["facet"]
    fig.update_xaxes(tickfont=dict(size=spec.sizes["tick"]))
    fig.update_yaxes(tickfont=dict(size=spec.sizes["tick"]))
    # Label the outer edge only: repeating the axis title under every panel
    # costs the space small panels are already short of.
    fig.update_xaxes(title_text=spec.x_col, row=n_rows)
    fig.update_yaxes(title_text=spec.y_col, col=1)
    return fig

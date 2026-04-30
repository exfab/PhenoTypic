"""Pure rendering helpers for the colony-view grid.

The colony-view tab lays out a 2D table of per-colony crops indexed by
two metadata columns chosen via dropdowns. This module owns the
*pure* rendering and column-introspection logic — every Dash callback
binding lives in :mod:`._callbacks`, and every CSS / JS asset lives in
:mod:`phenotypic.gui.results_viewer._assets`.

The three public helpers:

- :func:`selectable_axis_columns` — filter the master frame's columns
  down to those that make a sensible grid axis (low cardinality,
  metadata-flavoured).
- :func:`compute_max_bbox_size` — derive a uniform crop size from the
  largest bounding box in the (filtered) frame so every tile shares a
  canvas.
- :func:`build_grid` — render the actual ``html.Div`` component tree,
  alongside the row-major flat list of per-cell keys consumed by the
  shift+click range-selection callback.

Plus :func:`expand_range` for resolving shift+click slices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import html
from dash.development.base_component import Component

from phenotypic.gui.results_viewer._ids import (
    colony_cell_count_badge_id,
    colony_cell_remove_btn_id,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Column-name prefixes that mark a measurement (rather than metadata or
#: grid context). Columns starting with one of these are excluded from
#: :func:`selectable_axis_columns` because they have unbounded cardinality
#: and don't make sense as a grid axis.
_MEASUREMENT_PREFIXES: tuple[str, ...] = (
    "Bbox_",
    "Shape_",
    "Intensity_",
    "TextureGray_",
    "SymmetricRadius_",
    "GridSpatial_",
)

#: Per-object identifier; high cardinality, never a meaningful axis.
_OBJECT_LABEL_COL = "ObjectLabel"

#: Sort buckets — Metadata_ first, then Grid_, then everything else.
_METADATA_PREFIX = "Metadata_"
_GRID_PREFIX = "Grid_"

#: Minimum crop side length, even on degenerate (tiny / empty) frames.
_MIN_CROP_SIZE = 64


# ---------------------------------------------------------------------------
# Column introspection
# ---------------------------------------------------------------------------


def selectable_axis_columns(
    df: pl.DataFrame,
    column_value_sets: Mapping[str, list[str]],
    max_cardinality: int = 50,
) -> list[str]:
    """Return columns suitable as a colony-grid axis.

    A column is suitable iff:

    - cardinality (unique non-null values) is in ``[2, max_cardinality]``;
    - name does not start with one of the measurement prefixes
      (``Bbox_``, ``Shape_``, ``Intensity_``, ``TextureGray_``,
      ``SymmetricRadius_``, ``GridSpatial_``);
    - name is not ``ObjectLabel`` (per-object identifier — too high
      cardinality and not a meaningful axis).

    The returned list is sorted in three buckets: ``Metadata_*`` first
    (alphabetic within), then ``Grid_*`` (alphabetic), then everything
    else (alphabetic).

    Args:
        df: The frame to inspect (typically the master frame after the
            filter sidebar has been applied).
        column_value_sets: Mapping from column name to its sorted unique
            string values, as exposed by :attr:`OutputRoot.column_value_sets`.
            Used to read cardinality without re-scanning ``df``.
        max_cardinality: Upper bound on accepted cardinalities. Defaults to
            50 — large enough for time-courses or replicate counts, small
            enough to keep the grid tractable.

    Returns:
        Column names in the bucketed sort order described above.
    """
    suitable: list[str] = []
    for col in df.columns:
        if col == _OBJECT_LABEL_COL:
            continue
        if any(col.startswith(prefix) for prefix in _MEASUREMENT_PREFIXES):
            continue
        # Prefer the precomputed value set for cardinality; fall back to a
        # lazy scan of df if the column is missing from the mapping.
        try:
            cardinality = len(column_value_sets[col])
        except KeyError:
            cardinality = (
                df.get_column(col).drop_nulls().unique().len()
                if col in df.columns
                else 0
            )
        if cardinality < 2 or cardinality > max_cardinality:
            continue
        suitable.append(col)

    def _bucket(name: str) -> int:
        if name.startswith(_METADATA_PREFIX):
            return 0
        if name.startswith(_GRID_PREFIX):
            return 1
        return 2

    suitable.sort(key=lambda name: (_bucket(name), name))
    return suitable


# ---------------------------------------------------------------------------
# Crop sizing
# ---------------------------------------------------------------------------


def compute_max_bbox_size(df: pl.DataFrame, padding: int = 8) -> int:
    """Return a uniform crop side length covering every bbox in ``df``.

    Computes ``max(Bbox_MaxRR - Bbox_MinRR, Bbox_MaxCC - Bbox_MinCC)`` over
    the frame and adds ``2 * padding`` so colonies never butt up against
    the tile edge. Returns at least :data:`_MIN_CROP_SIZE` (64 px) to
    avoid degenerate tiny crops on frames where every colony is a single
    pixel.

    Args:
        df: Frame containing ``Bbox_MinRR``, ``Bbox_MaxRR``, ``Bbox_MinCC``,
            and ``Bbox_MaxCC``. Empty frames or frames missing any of the
            four columns return :data:`_MIN_CROP_SIZE`.
        padding: Per-side padding added to both sides of the maximum
            bbox extent. Defaults to 8 px.

    Returns:
        Side length of the square crop, in pixels, no smaller than 64.
    """
    required = ("Bbox_MinRR", "Bbox_MaxRR", "Bbox_MinCC", "Bbox_MaxCC")
    if df.is_empty() or not all(col in df.columns for col in required):
        return _MIN_CROP_SIZE

    extents = df.select(
        (pl.col("Bbox_MaxRR") - pl.col("Bbox_MinRR")).alias("rr"),
        (pl.col("Bbox_MaxCC") - pl.col("Bbox_MinCC")).alias("cc"),
    )
    max_rr = extents.get_column("rr").max()
    max_cc = extents.get_column("cc").max()
    if max_rr is None or max_cc is None:
        return _MIN_CROP_SIZE
    # polars `.max()` returns a broad runtime union; the bbox columns are
    # numeric ints so SupportsInt holds in practice.
    bbox_max = max(int(max_rr), int(max_cc))  # type: ignore[arg-type]
    return max(_MIN_CROP_SIZE, bbox_max + 2 * padding)


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def _format_axis_value(value: object) -> str:
    """Render an axis value as a header label, preserving numeric ordering.

    Polars' default sort order for the column dtype is honoured upstream
    by ``unique().sort()``; this helper just stringifies the result.
    """
    if value is None:
        return ""
    return f"{value}"


def _representative_per_cell(
    df: pl.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
) -> pl.DataFrame:
    """Pick the representative colony per (x, y) cell.

    Sorts by ``(x, y, ObjectLabel)`` so the ``first`` aggregate picks the
    smallest-label colony deterministically. Also reports the per-cell
    count so callers can render a ``N=k`` badge when a cell aggregates
    multiple colonies.

    Args:
        df: Filtered master frame. Must contain ``x_axis_col``, ``y_axis_col``,
            ``Metadata_ImageFile``, ``Metadata_Dataset``, and ``ObjectLabel``.
        x_axis_col: Column projected onto the grid's X-axis.
        y_axis_col: Column projected onto the grid's Y-axis.

    Returns:
        A frame with columns ``x_axis_col``, ``y_axis_col``,
        ``Metadata_ImageFile``, ``Metadata_Dataset``, ``ObjectLabel``, and
        ``count`` (number of colonies in the cell).
    """
    # If an axis column happens to also be one of the columns we want to
    # carry through with `pl.first(...)`, polars complains about a
    # duplicate output column. Skip those agg-firsts here — the axis
    # values are already available on the resulting frame as the
    # group-by keys.
    forwarded_cols = ["Metadata_ImageFile", "Metadata_Dataset", _OBJECT_LABEL_COL]
    aggs = [
        pl.first(col).alias(col)
        for col in forwarded_cols
        if col not in (x_axis_col, y_axis_col)
    ]
    aggs.append(pl.len().alias("count"))
    return (
        df.sort([x_axis_col, y_axis_col, _OBJECT_LABEL_COL])
        .group_by([x_axis_col, y_axis_col], maintain_order=True)
        .agg(*aggs)
    )


def _build_axis_label(value: object, *, axis: str) -> Component:
    """Render an X/Y axis header label for the corner row/column."""
    return html.Div(
        _format_axis_value(value),
        className=f"colony-axis-label colony-axis-label--{axis}",
        style={
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "0.75rem",
            "color": "#003660",
            "textAlign": "center",
            "alignSelf": "center",
            "padding": "0.25rem",
        },
    )


def _build_cell(
    *,
    image_file: str,
    label: int,
    dataset: str,
    count: int,
    max_size: int,
    has_overlay: bool,
    is_removed: bool,
    is_selected: bool,
) -> Component:
    """Render the chrome + crop for a single grid cell.

    Args:
        image_file: ``Metadata_ImageFile`` of the representative colony.
        label: ``ObjectLabel`` of the representative colony.
        dataset: ``Metadata_Dataset`` of the representative colony.
        count: Number of colonies aggregated into this cell.
        max_size: Crop side length, in pixels.
        has_overlay: Whether the source overlay PNG exists on disk; if not,
            a striped placeholder is rendered instead of an ``<img>``.
        is_removed: Whether the representative colony is in the curated
            removal set. Toggles the icon and dims the crop.
        is_selected: Whether the cell is in the active multi-select.

    Returns:
        A component ready to drop into the grid container.
    """
    classes = ["colony-cell"]
    if is_selected:
        classes.append("is-selected")
    if is_removed:
        classes.append("is-removed")

    if has_overlay:
        crop_url = f"/crops/{dataset}/{image_file}/{label}.png?size={max_size}"
        crop_node: Component = html.Img(
            src=crop_url,
            className="colony-cell-img",
            style={
                "width": f"{max_size}px",
                "height": f"{max_size}px",
                "display": "block",
                "opacity": "0.3" if is_removed else "1",
                "objectFit": "cover",
            },
        )
    else:
        crop_node = html.Div(
            className="colony-cell-placeholder",
            style={
                "width": f"{max_size}px",
                "height": f"{max_size}px",
                "backgroundImage": (
                    "repeating-linear-gradient(45deg, "
                    "rgba(0,54,96,0.05) 0px, rgba(0,54,96,0.05) 8px, "
                    "rgba(0,54,96,0.10) 8px, rgba(0,54,96,0.10) 16px)"
                ),
                "border": "1px dashed rgba(0,54,96,0.25)",
            },
        )

    # CSS-styled span playing the role of a checkbox. We don't use a real
    # <input> because Dash 4 doesn't expose html.Input, and the JS layer
    # only needs `data-key` + the class name to wire up the click event.
    # Visual checked state is driven by the `is-checked` modifier class.
    checkbox_class = "colony-cell-checkbox"
    if is_selected:
        checkbox_class += " is-checked"
    # `data-*` HTML attributes can't be expressed in Dash's typed Span
    # kwargs, so unpack via an Any-typed dict to bypass the stub mismatch.
    extra_props: dict[str, Any] = {"data-key": f"{image_file}::{label}"}
    checkbox_inner = html.Span(
        "",
        className=checkbox_class,
        **extra_props,
    )
    checkbox = html.Span(
        checkbox_inner,
        className="colony-cell-checkbox-wrap",
        style={
            "position": "absolute",
            "top": "4px",
            "left": "4px",
            "zIndex": "2",
        },
    )

    remove_btn = dbc.Button(
        "↺" if is_removed else "✕",
        id=colony_cell_remove_btn_id(image_file, label),
        color="danger" if not is_removed else "secondary",
        outline=True,
        size="sm",
        className="colony-cell-remove-btn",
        style={
            "position": "absolute",
            "top": "4px",
            "right": "4px",
            "zIndex": "2",
            "padding": "0 0.4rem",
            "lineHeight": "1.2",
        },
        title="Restore colony" if is_removed else "Remove colony",
    )

    children: list[Component] = [crop_node, checkbox, remove_btn]

    if count > 1:
        # Render the count badge as a button so a future wave can layer in
        # an expand-on-click drilldown without changing the DOM shape.
        children.append(
            dbc.Button(
                f"N={count}",
                id=colony_cell_count_badge_id(image_file, label),
                color="light",
                size="sm",
                className="colony-cell-count-badge",
                style={
                    "position": "absolute",
                    "bottom": "4px",
                    "right": "4px",
                    "zIndex": "2",
                    "padding": "0 0.35rem",
                    "fontSize": "0.65rem",
                    "lineHeight": "1.2",
                    "fontFamily": "'DM Mono', monospace",
                },
            )
        )

    return html.Div(
        children,
        className=" ".join(classes),
        style={
            "position": "relative",
            "width": f"{max_size}px",
            "height": f"{max_size}px",
        },
    )


def build_grid(
    df: pl.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
    max_size: int,
    removed_keys: set[tuple[str, int]],
    selected_keys: set[tuple[str, int]],
    output_root: OutputRoot,
) -> tuple[Component, list[tuple[str, int]]]:
    """Render the colony-grid component and its row-major key order.

    Layout:

    - **Top row**: X-axis value labels (one column header per unique X value).
    - **Left column**: Y-axis value labels (one row header per unique Y value).
    - **Each cell**: a representative colony crop (smallest ``ObjectLabel``
      among rows matching the cell's ``(x_value, y_value)``) plus chrome
      (× / ↺ button, multi-select checkbox, optional ``N=k`` badge).

    Per-cell visual state:

    - Crop dimmed (``opacity: 0.3``) and ``↺`` icon when the cell's
      representative colony is in ``removed_keys``.
    - Outer cell ``html.Div`` gets ``is-selected`` when the cell's
      representative colony is in ``selected_keys``.
    - If :meth:`OutputRoot.has_overlay` is False for the cell's
      ``(dataset, stem)`` pair, a striped placeholder div is rendered
      instead of the ``<img>``.

    Args:
        df: Filtered master frame (after :class:`FilterSpec.apply_to`).
        x_axis_col: Column projected onto the X-axis.
        y_axis_col: Column projected onto the Y-axis.
        max_size: Side length, in pixels, of every crop tile.
        removed_keys: Set of ``(image_file, label)`` keys currently in
            the curated removal set.
        selected_keys: Set of ``(image_file, label)`` keys currently in
            the active multi-select.
        output_root: Validated handle on the output root, used to answer
            :meth:`OutputRoot.has_overlay` per cell.

    Returns:
        A tuple ``(component, grid_order)``. ``grid_order`` is the
        row-major flat list of representative ``(image_file, label)``
        keys (Y-axis outer, X-axis inner) in the same iteration order as
        the rendered cells. Consumed by the selection-range callback to
        resolve shift+click slices.
    """
    if df.is_empty() or x_axis_col not in df.columns or y_axis_col not in df.columns:
        return html.Div("No colonies match the active filter.", className="text-muted"), []

    x_values = (
        df.get_column(x_axis_col).drop_nulls().unique().sort().to_list()
    )
    y_values = (
        df.get_column(y_axis_col).drop_nulls().unique().sort().to_list()
    )
    if not x_values or not y_values:
        return html.Div("No colonies match the active filter.", className="text-muted"), []

    representatives = _representative_per_cell(df, x_axis_col, y_axis_col)

    # Index the representative frame for O(1) per-cell lookup.
    cell_index: dict[tuple[object, object], dict[str, object]] = {}
    for row in representatives.iter_rows(named=True):
        cell_index[(row[x_axis_col], row[y_axis_col])] = {
            "image_file": row["Metadata_ImageFile"],
            "dataset": row["Metadata_Dataset"],
            "label": row[_OBJECT_LABEL_COL],
            "count": row["count"],
        }

    # Build the grid, walking Y outer / X inner so the row-major key list
    # mirrors the visible reading order (left-to-right within a row).
    children: list[Component] = []

    # Empty top-left corner.
    children.append(html.Div(className="colony-grid-corner"))
    # X-axis header row.
    for x_value in x_values:
        children.append(_build_axis_label(x_value, axis="x"))

    grid_order: list[tuple[str, int]] = []
    for y_value in y_values:
        children.append(_build_axis_label(y_value, axis="y"))
        for x_value in x_values:
            entry = cell_index.get((x_value, y_value))
            if entry is None:
                # Empty cell — render a blank placeholder to keep grid alignment.
                children.append(
                    html.Div(
                        className="colony-cell colony-cell--empty",
                        style={
                            "width": f"{max_size}px",
                            "height": f"{max_size}px",
                            "background": "rgba(0,54,96,0.03)",
                            "borderRadius": "4px",
                        },
                    )
                )
                continue
            image_file = str(entry["image_file"])
            dataset = str(entry["dataset"])
            # polars row dicts type values as `object`; the columns are
            # known-int upstream so SupportsInt holds.
            label = int(entry["label"])  # type: ignore[call-overload]
            count = int(entry["count"])  # type: ignore[call-overload]
            key = (image_file, label)
            grid_order.append(key)
            children.append(
                _build_cell(
                    image_file=image_file,
                    label=label,
                    dataset=dataset,
                    count=count,
                    max_size=max_size,
                    has_overlay=output_root.has_overlay(dataset, image_file),
                    is_removed=key in removed_keys,
                    is_selected=key in selected_keys,
                )
            )

    grid = html.Div(
        children,
        id="colony-grid-css-grid",
        className="colony-grid",
        style={
            "display": "grid",
            "gridTemplateColumns": "auto " + " ".join([f"{max_size}px"] * len(x_values)),
            "gridTemplateRows": "auto " + " ".join([f"{max_size}px"] * len(y_values)),
            "gap": "8px",
            "padding": "0.5rem",
        },
    )
    return grid, grid_order


# ---------------------------------------------------------------------------
# Range expansion
# ---------------------------------------------------------------------------


def expand_range(
    grid_order: list[tuple[str, int]],
    anchor: tuple[str, int],
    target: tuple[str, int],
) -> list[tuple[str, int]]:
    """Return the contiguous slice of ``grid_order`` between two keys.

    Direction-agnostic: the slice always runs from the lower of the two
    indices to the higher (inclusive) regardless of which key was
    originally clicked first.

    Args:
        grid_order: Row-major flat list of cell keys, as returned by
            :func:`build_grid`.
        anchor: Key of the cell that started the range (the most recent
            non-shift click).
        target: Key of the shift-clicked cell.

    Returns:
        Inclusive slice of ``grid_order`` between the two keys.

    Raises:
        ValueError: If either ``anchor`` or ``target`` is not in
            ``grid_order``.
    """
    try:
        a = grid_order.index(anchor)
    except ValueError as exc:
        raise ValueError(f"anchor {anchor!r} is not in grid_order") from exc
    try:
        b = grid_order.index(target)
    except ValueError as exc:
        raise ValueError(f"target {target!r} is not in grid_order") from exc
    lo, hi = (a, b) if a <= b else (b, a)
    return grid_order[lo : hi + 1]


__all__ = [
    "selectable_axis_columns",
    "compute_max_bbox_size",
    "build_grid",
    "expand_range",
]

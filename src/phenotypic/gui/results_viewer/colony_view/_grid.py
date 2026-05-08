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
from dash import dcc, html
from dash.development.base_component import Component
from flask import current_app, has_app_context

from phenotypic.gui._config import CFG_URL_PREFIX, MOUNT_HOME
from phenotypic.gui._design import (
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
)
from phenotypic.gui.results_viewer._ids import (
    colony_cell_count_badge_id,
    colony_cell_popover_body_id,
    colony_cell_popover_data_id,
    colony_cell_remove_btn_id,
)
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
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

#: Per-object identifier alias — sourced from :mod:`._filtered_state` so
#: the curation key columns and the colony grid stay in sync.
_OBJECT_LABEL_COL = KEY_OBJECT_LABEL


def _url_prefix() -> str:
    """Return the active app's mount-point prefix.

    Read from ``flask.current_app.config["pheno_url_prefix"]`` when a
    request context is active. Falls back to ``"/"`` for the standalone
    case (or when called outside a request, e.g. unit tests building a
    grid without spinning up a Flask app).
    """
    if has_app_context():
        return current_app.config.get(CFG_URL_PREFIX, MOUNT_HOME)
    return MOUNT_HOME

#: Sort buckets — Metadata_ first, then Grid_, then everything else.
_METADATA_PREFIX = "Metadata_"
_GRID_PREFIX = "Grid_"

#: Minimum crop side length, even on degenerate (tiny / empty) frames.
_MIN_CROP_SIZE = 64

#: Vertical room reserved beneath every cell for the multi-colony stack
#: tab to peek out of. Applied to every tile (even single-colony ones)
#: so the grid stays evenly spaced regardless of which cells aggregate.
_STACK_TAB_OFFSET = 14


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
    # Collect the full per-cell list of `(image_file, dataset, label)` tuples
    # via `pl.col(...)` aggregates that return polars lists. Aliased to
    # `_members_*` so the names never collide with group-by keys (one of
    # the forwarded columns may also be an axis column when the user
    # picks e.g. Metadata_Dataset as an axis).
    aggs = [
        pl.col(KEY_IMAGE_FILE).alias("_members_image_file"),
        pl.col("Metadata_Dataset").alias("_members_dataset"),
        pl.col(_OBJECT_LABEL_COL).alias("_members_label"),
        pl.len().alias("count"),
    ]
    return (
        df.sort([x_axis_col, y_axis_col, _OBJECT_LABEL_COL])
        .group_by([x_axis_col, y_axis_col], maintain_order=True)
        .agg(*aggs)
    )


def _build_axis_label(value: object, *, axis: str, max_width_px: int) -> Component:
    """Render an X/Y axis header label for the corner row/column.

    ``max_width_px`` caps the label so long values wrap inside their grid
    track instead of stretching the column (Y) or spilling past the cell
    width (X). Combined with ``minWidth: 0`` and ``overflowWrap: anywhere``
    so the wrap is robust against long unbroken tokens like dataset stems
    (``Run_12-01_29_26``).
    """
    return html.Div(
        _format_axis_value(value),
        className=f"colony-axis-label colony-axis-label--{axis}",
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "color": COLOR_NAVY,
            "textAlign": "center",
            "alignSelf": "center",
            "padding": "0.25rem",
            "maxWidth": f"{max_width_px}px",
            "minWidth": 0,
            "whiteSpace": "normal",
            "wordBreak": "break-word",
            "overflowWrap": "anywhere",
            "lineHeight": "1.2",
        },
    )


def _build_cell(
    *,
    image_file: str,
    label: int,
    dataset: str,
    count: int,
    max_size: int,
    display_size: int,
    has_overlay: bool,
    is_removed: bool,
    is_selected: bool,
    members: list[tuple[str, str, int]] | None = None,
) -> Component:
    """Render the chrome + crop for a single grid cell.

    Args:
        image_file: ``Metadata_ImageFile`` of the representative colony.
        label: ``ObjectLabel`` of the representative colony.
        dataset: ``Metadata_Dataset`` of the representative colony.
        count: Number of colonies aggregated into this cell.
        max_size: Server crop side length, in pixels (used in the URL so
            the PNG is generated at full resolution covering the colony's
            bbox).
        display_size: CSS render size, in pixels. The browser scales the
            ``<img>`` to this size; ``object-fit`` keeps colonies centered
            without distortion.
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
        prefix = _url_prefix()
        crop_url = f"{prefix}crops/{dataset}/{image_file}/{label}.png?size={max_size}"
        crop_node: Component = html.Img(
            src=crop_url,
            className="colony-cell-img",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
                "display": "block",
                "opacity": "0.3" if is_removed else "1",
                "objectFit": "cover",
            },
        )
    else:
        crop_node = html.Div(
            className="colony-cell-placeholder",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
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
        title=(
            "add colony to measurements"
            if is_removed
            else "remove colony from measurements"
        ),
    )

    # Image card: the framed display_size×display_size area carrying the
    # crop, checkbox, and remove button. Sits in front of the stack tab
    # via z-index so the tab can peek out from beneath the bottom edge.
    frame = html.Div(
        [crop_node, checkbox, remove_btn],
        className="colony-cell-frame",
    )

    children: list[Component] = [frame]

    if count > 1:
        badge_id = colony_cell_count_badge_id(image_file, label)
        children.append(
            dbc.Button(
                f"N={count}",
                id=badge_id,
                className="colony-cell-stack-tab",
                title=f"click to expand all {count} colonies in this cell",
                n_clicks=0,
            )
        )
        if members:
            children.extend(
                _build_stack_popover(
                    target_id=badge_id,
                    image_file=image_file,
                    label=label,
                    members=members,
                    crop_size=max_size,
                    display_size=display_size,
                )
            )

    return html.Div(
        children,
        className=" ".join(classes),
        style={
            "position": "relative",
            "width": f"{display_size}px",
            "height": f"{display_size + _STACK_TAB_OFFSET}px",
            "overflow": "visible",
        },
    )


def _build_stack_popover(
    *,
    target_id: Mapping[str, Any],
    image_file: str,
    label: int,
    members: list[tuple[str, str, int]],
    crop_size: int,
    display_size: int,
) -> list[Component]:
    """Render the click-to-expand stack popover with a deferred body.

    The popover anchors to the cell's ``N=k`` badge and ships an empty
    body plus a co-located ``dcc.Store`` carrying the cell's members and
    sizes. The first time the badge is clicked, a pattern-matched
    callback (see :func:`build_stack_popover_rows`) reads the store and
    populates the body. The ``<img>`` elements never enter the DOM until
    the user actually opens the stack — strictly stronger than native
    ``loading="lazy"`` because there is nothing for the browser to fetch.

    Returns a pair ``[popover, store]`` so the caller can splice both
    siblings into the cell tree.
    """
    body_id = colony_cell_popover_body_id(image_file, label)
    data_id = colony_cell_popover_data_id(image_file, label)
    popover = dbc.Popover(
        dbc.PopoverBody(
            [],
            id=body_id,
            style={
                "maxHeight": "60vh",
                "overflowY": "auto",
                "padding": "0.5rem",
            },
        ),
        target=target_id,
        trigger="legacy",
        placement="right",
        hide_arrow=False,
        style={"zIndex": "1080"},
    )
    store = dcc.Store(
        id=data_id,
        data={
            "members": [[im, ds, lbl] for im, ds, lbl in members],
            "crop_size": int(crop_size),
            "display_size": int(display_size),
        },
    )
    return [popover, store]


def build_stack_popover_rows(
    members: list[tuple[str, str, int]],
    *,
    crop_size: int,
    display_size: int,
    removed_keys: set[tuple[str, int]],
) -> list[Component]:
    """Render the per-member rows that populate a stack popover body.

    Called from a pattern-matched populate-on-click callback when the
    user first opens a multi-colony cell's badge. Each member colony
    renders as a small ``<img>`` with its label beneath; removed
    colonies are dimmed.
    """
    rows: list[Component] = []
    prefix = _url_prefix()
    for image_file, dataset, label in members:
        is_removed = (image_file, label) in removed_keys
        crop_url = f"{prefix}crops/{dataset}/{image_file}/{label}.png?size={crop_size}"
        rows.append(
            html.Div(
                [
                    html.Img(
                        src=crop_url,
                        style={
                            "width": f"{display_size}px",
                            "height": f"{display_size}px",
                            "objectFit": "cover",
                            "display": "block",
                            "borderRadius": "3px",
                            "opacity": "0.3" if is_removed else "1",
                        },
                    ),
                    html.Div(
                        f"label {label}"
                        + ("  (removed)" if is_removed else ""),
                        style={
                            "fontFamily": FONT_FAMILY_MONO,
                            "fontSize": FONT_SIZE_CAPTION,
                            "color": COLOR_NAVY,
                            "textAlign": "center",
                            "marginTop": "0.15rem",
                        },
                    ),
                ],
                style={"marginBottom": "0.4rem"},
            )
        )
    return rows


def build_grid(
    df: pl.DataFrame,
    x_axis_col: str,
    y_axis_col: str,
    max_size: int,
    removed_keys: set[tuple[str, int]],
    selected_keys: set[tuple[str, int]],
    output_root: OutputRoot,
    display_size: int | None = None,
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
        max_size: Side length, in pixels, of every server-side crop tile.
            Used in the crop URL so the PNG always covers the colony bbox
            at full resolution. Independent of how the tile is sized in
            the browser.
        removed_keys: Set of ``(image_file, label)`` keys currently in
            the curated removal set.
        selected_keys: Set of ``(image_file, label)`` keys currently in
            the active multi-select.
        output_root: Validated handle on the output root, used to answer
            :meth:`OutputRoot.has_overlay` per cell.
        display_size: CSS render size, in pixels, for each tile. Defaults
            to ``max_size`` (no scaling). Pass a smaller value to shrink
            the grid into the viewport without re-cropping; the browser
            scales the ``<img>`` and ``object-fit: cover`` keeps the
            colony centred.

    Returns:
        A tuple ``(component, grid_order)``. ``grid_order`` is the
        row-major flat list of representative ``(image_file, label)``
        keys (Y-axis outer, X-axis inner) in the same iteration order as
        the rendered cells. Consumed by the selection-range callback to
        resolve shift+click slices.
    """
    if display_size is None:
        display_size = max_size

    if df.is_empty() or x_axis_col not in df.columns or y_axis_col not in df.columns:
        return html.Div("No colonies match the active filter.", className="text-muted"), []

    if x_axis_col == y_axis_col:
        # polars rejects ``group_by([col, col])`` with a duplicate-column
        # error; head off the crash with a friendly message instead.
        return (
            html.Div(
                "Pick distinct X and Y axis columns to render the grid.",
                className="text-muted",
            ),
            [],
        )

    x_values = (
        df.get_column(x_axis_col).drop_nulls().unique().sort().to_list()
    )
    y_values = (
        df.get_column(y_axis_col).drop_nulls().unique().sort().to_list()
    )
    if not x_values or not y_values:
        return html.Div("No colonies match the active filter.", className="text-muted"), []

    representatives = _representative_per_cell(df, x_axis_col, y_axis_col)

    # Index the representative frame for O(1) per-cell lookup. The
    # representative is the first member (smallest ObjectLabel) and the
    # `members` list carries every colony in the cell so the click-to-
    # expand popover can render the full mini-stack.
    cell_index: dict[tuple[object, object], dict[str, object]] = {}
    for row in representatives.iter_rows(named=True):
        members_image_file = list(row["_members_image_file"])
        members_dataset = list(row["_members_dataset"])
        members_label = list(row["_members_label"])
        members = [
            (str(im), str(ds), int(lbl))
            for im, ds, lbl in zip(
                members_image_file,
                members_dataset,
                members_label,
                strict=True,
            )
        ]
        cell_index[(row[x_axis_col], row[y_axis_col])] = {
            "image_file": members_image_file[0] if members_image_file else None,
            "dataset": members_dataset[0] if members_dataset else None,
            "label": members_label[0] if members_label else None,
            "count": row["count"],
            "members": members,
        }

    # Build the grid, walking Y outer / X inner so the row-major key list
    # mirrors the visible reading order (left-to-right within a row).
    children: list[Component] = []

    # Y-label column gets the same width as a cell so long dataset stems
    # wrap into multiple lines instead of widening the entire grid.
    y_label_width = display_size

    # Empty top-left corner.
    children.append(html.Div(className="colony-grid-corner"))
    # X-axis header row.
    for x_value in x_values:
        children.append(_build_axis_label(x_value, axis="x", max_width_px=display_size))

    grid_order: list[tuple[str, int]] = []
    for y_value in y_values:
        children.append(
            _build_axis_label(y_value, axis="y", max_width_px=y_label_width)
        )
        for x_value in x_values:
            entry = cell_index.get((x_value, y_value))
            if entry is None:
                # Empty cell — render a blank placeholder to keep grid alignment.
                children.append(
                    html.Div(
                        className="colony-cell colony-cell--empty",
                        style={
                            "width": f"{display_size}px",
                            "height": f"{display_size + _STACK_TAB_OFFSET}px",
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
            members = entry.get("members") or []
            # ``members`` came in already typed as ``list[tuple[str, str, int]]``
            # from the index; cast for the local helper.
            typed_members = [
                (str(m[0]), str(m[1]), int(m[2])) for m in members  # type: ignore[index]
            ]
            children.append(
                _build_cell(
                    image_file=image_file,
                    label=label,
                    dataset=dataset,
                    count=count,
                    max_size=max_size,
                    display_size=display_size,
                    has_overlay=output_root.has_overlay(dataset, image_file),
                    is_removed=key in removed_keys,
                    is_selected=key in selected_keys,
                    members=typed_members,
                )
            )

    grid = html.Div(
        children,
        id="colony-grid-css-grid",
        className="colony-grid",
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"minmax(0, {y_label_width}px) "
                + " ".join([f"{display_size}px"] * len(x_values))
            ),
            "gridTemplateRows": (
                "auto "
                + " ".join([f"{display_size + _STACK_TAB_OFFSET}px"] * len(y_values))
            ),
            "gap": "8px",
            "padding": "0.5rem",
            # Shrink-wrap to the column widths so the grid sits flush
            # against the container's left edge instead of stretching to
            # block-level width and floating its tracks.
            "width": "max-content",
            "justifySelf": "start",
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

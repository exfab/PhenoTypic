"""Callbacks for the QC Review sub-view.

Wires the master–detail walkthrough described in spec §D.2–D.6 on top of
the pure data layer (:mod:`._data`) and the per-module review-progress
store (:mod:`._review_state`). The Dash-coupled glue lives here; the
load-bearing logic (artifact slicing, summary stats, recompute frame,
review state) is tested through those modules directly.

Callback map:

* **module switch / re-sort** → rebuild worklist + summary + frozen order
  store, render the first group into the detail pane.
* **select group** (worklist row click) → render the detail header +
  faceted tile gallery for that group.
* **per-tile remove / bulk remove+restore** → mutate the shared
  :class:`~phenotypic.gui.results_viewer._filtered_state.FilteredMeasurements`
  removal set (same store the colony view writes).
* **mark reviewed / next** → mark progress, and *if the group was
  curated*, run an in-session per-group recompute (``run_qc`` only — never
  ``finalize_*``) on the post-applied + metadata-joined frame, then update
  the group's metric/badge **in place** (no reorder).

Critical invariants (spec §D risk refinements):

* Recompute reads ``measurements.parquet`` and anti-joins the live
  removal set (:func:`._data.build_recompute_frame`) — never
  ``master − removed``.
* ``removed_keys`` is read under the ``FilteredMeasurements`` lock so the
  recomputed ``qc/`` reflects a coherent state-at-mark-reviewed.
* The summary header counts NaN/insufficient groups separately from
  ``pass`` and uses a robust median (handled in :func:`._data.summary_stats`).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import dash
import polars as pl
from dash import ALL, MATCH, Input, Output, State, callback_context, html, no_update
from dash.development.base_component import Component
from flask import current_app

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_OUTPUT_ROOT,
    CFG_QC_PIPELINE,
    CFG_URL_PREFIX,
    MOUNT_HOME,
    QC_CROPS_URL_SEGMENT,
    TILE_DIM_DEFAULT,
    stepped_alpha_from_trigger,
)
from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
    OI_GREEN_TEXT,
    OI_ORANGE_TEXT,
    OI_VERMILION_TEXT,
)
from phenotypic.gui._shared.tiles import build_tile_grid
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._filtered_state import (
    decode_removed_keys_payload,
)
from phenotypic.gui.results_viewer._qc_tab.review import _data, _ids as rids
from phenotypic.gui.results_viewer._qc_tab.review._layout import (
    _SUMMARY_HEADER_HEIGHT,
    clamp_sidebar_width,
    collapsed_sidebar_style,
    expanded_sidebar_style,
)
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState,
    decode_group_key,
    encode_group_key,
)

logger = logging.getLogger(__name__)

#: Bootstrap badge colours by QC status (mirrors the Configure card map);
#: ``insufficient`` is its own neutral colour so a NaN group never reads
#: as a green ``pass``.
_BADGE_COLOR_BY_STATUS: dict[str, str] = {
    "fail": "danger",
    "warn": "warning",
    "pass": "success",
    "insufficient": "secondary",
}


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------


def _url_prefix() -> str:
    """Return the active app's mount-point prefix (``/`` standalone)."""
    return current_app.config.get(CFG_URL_PREFIX, MOUNT_HOME)


def _qc_crop_url(
    dataset: str,
    image_file: str,
    label: int,
    crop_size: int,
    *,
    dim_alpha: float = 0.0,
) -> str:
    """Build a QC-gallery crop ``<img>`` src for one tile.

    Points at the QC crop route mounted under
    :data:`QC_CROPS_URL_SEGMENT` (see
    :func:`phenotypic.gui._shared.tiles.register_crop_route`).

    Args:
        dataset: ``Metadata_Dataset`` of the tile's colony.
        image_file: ``Metadata_ImageFile`` of the tile's colony.
        label: ``Object_Label`` of the tile's colony.
        crop_size: Server crop side length, in pixels (``?size=``).
        dim_alpha: Tile-spotlight strength forwarded to the crop route as
            ``&dim=``. ``0.0`` (default) is today's full-context crop.
            Bound per-render via :func:`functools.partial` so the 4-arg
            ``url_builder`` protocol :func:`build_tile_grid` expects is
            preserved.
    """
    prefix = _url_prefix()
    return (
        f"{prefix}{QC_CROPS_URL_SEGMENT}/{dataset}/{image_file}/"
        f"{label}.png?size={crop_size}&dim={dim_alpha}"
    )


def _review_tile_remove_button(
    image_file: str, label: int, is_removed: bool
) -> Component:
    """Build a Review-gallery per-tile remove/restore button."""
    return dbc.Button(
        "↺" if is_removed else "✕",
        id=rids.review_tile_remove_btn_id(image_file, label),
        color="secondary" if is_removed else "danger",
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
            "restore colony to measurements"
            if is_removed
            else "remove colony from measurements"
        ),
    )


# ---------------------------------------------------------------------------
# Rendering helpers (pure: state in, components out)
# ---------------------------------------------------------------------------


def _render_summary_header(
    stats: dict[str, Any], reviewed: int, colonies_removed: int
) -> Component:
    """Render the per-module summary stat tiles as one horizontal row.

    ``insufficient`` is shown as its own tile so a no-signal group never
    inflates the ``pass`` count (spec §D risk refinement). The tiles are
    wrapped in a horizontal flex row so they read left-to-right across the
    top of the Review pane (a plain list of block ``Div``s would stack
    vertically and eat the whole column).
    """
    median = stats.get("median_metric")
    median_text = "N/A" if median is None else f"{median:.3f}"
    tiles = [
        ("Total", stats.get("total", 0), COLOR_NAVY),
        ("Fail", stats.get("fail", 0), OI_VERMILION_TEXT),
        ("Warn", stats.get("warn", 0), OI_ORANGE_TEXT),
        ("Pass", stats.get("pass", 0), OI_GREEN_TEXT),
        ("Insufficient", stats.get("insufficient", 0), COLOR_MUTED),
        ("Reviewed", reviewed, COLOR_NAVY),
        ("Removed", colonies_removed, COLOR_MUTED),
        ("Median metric", median_text, COLOR_NAVY),
    ]
    tile_nodes = [
        html.Div(
            [
                html.Div(
                    str(value),
                    style={
                        "fontWeight": 600,
                        "color": color,
                        "fontSize": "1.1rem",
                        "fontFamily": FONT_FAMILY_MONO,
                    },
                ),
                html.Div(
                    label,
                    style={"color": COLOR_MUTED, "fontSize": FONT_SIZE_CAPTION},
                ),
            ],
            # ``flex: 0 0 auto`` keeps each tile at its intrinsic width so
            # they pack side-by-side and wrap to at most a row or two —
            # never stretching to full width / one-per-line at a narrow
            # viewport (the reported bug).
            style={
                "flex": "0 0 auto",
                "textAlign": "center",
                "minWidth": "70px",
            },
        )
        for label, value, color in tiles
    ]
    return html.Div(
        tile_nodes,
        style={
            "display": "flex",
            "flexDirection": "row",
            "flexWrap": "wrap",
            "alignItems": "center",
            "gap": "1rem 1.25rem",
        },
    )


def _render_worklist_rows(
    worklist,  # polars.DataFrame
    instance_id: str,
    groupby_cols: list[str],
    review_state: ReviewState,
    deltas: dict[str, dict[str, Any]],
    selected_encoded: str | None,
) -> list[Component]:
    """Render the worklist sidebar rows (frozen order, reviewed dimmed)."""
    rows: list[Component] = []
    for record in worklist.iter_rows(named=True):
        key_values = tuple(record.get(col) for col in groupby_cols)
        encoded = encode_group_key(key_values)
        is_reviewed = review_state.is_reviewed(instance_id, key_values)
        delta = deltas.get(encoded, {})
        # Prefer the in-session recompute's after-metric/status when this
        # group has been recomputed, so a full re-render (module switch /
        # ↻ Re-sort) carries the recomputed value, not the frozen-frame one.
        metric, status = _row_metric_status(record, delta)
        rows.append(
            _render_worklist_row(
                instance_id=instance_id,
                encoded=encoded,
                key_values=key_values,
                metric=metric,
                status=status,
                is_reviewed=is_reviewed,
                is_selected=encoded == selected_encoded,
                moved=bool(delta.get("moved")),
            )
        )
    return rows


def _row_metric_status(
    record: dict[str, Any], delta: dict[str, Any]
) -> tuple[Any, str]:
    """Resolve a row's display metric + status, preferring a recompute delta.

    The frozen worklist frame carries the metric/status committed by the
    last ``run_qc`` artifact write. When an in-session recompute has
    produced a delta for this group, its ``after`` metric and
    ``status_after`` are the authoritative current values, so they win.

    Args:
        record: The group's frozen summary row.
        delta: The group's recompute delta (``{}`` when never recomputed).

    Returns:
        ``(metric, status)`` for display.
    """
    if delta:
        return delta.get("after"), str(
            delta.get("status_after", record.get("status"))
        )
    return record.get("metric"), str(record.get("status"))


def render_worklist_row_metric_cell(
    metric: Any, status: str, *, moved: bool = False
) -> list[Component]:
    """Build the metric-span + status-badge children of one worklist-row cell.

    Module-level + pure so the in-place metric/badge update callback
    (:func:`_register_worklist_row_metric_callback`) and the initial
    row render share **one** rendering of the cell — and so the recompute
    update is unit-testable without booting Dash. The status drives the
    badge colour, so swapping in the recompute ``after`` status here flips
    the badge in place (never leaving a stale colour beside a new number).
    The ``⤳`` "changed after recompute" hint lives **inside** this cell so
    the in-place update can add it without re-rendering the whole row.

    Args:
        metric: The group's metric value (``None`` / NaN renders ``insuf.``).
        status: The group's QC status (``fail`` / ``warn`` / ``pass`` /
            ``insufficient``) — drives the badge colour.
        moved: Whether this group's metric changed in an in-session
            recompute (appends the ``⤳`` hint).

    Returns:
        The ``[Span(metric_text), Badge(status)[, Span(⤳)]]`` children list.
    """
    children: list[Component] = [
        html.Span(
            f" {_format_metric(metric)} ",
            style={"fontFamily": FONT_FAMILY_MONO},
        ),
        dbc.Badge(
            status,
            color=_BADGE_COLOR_BY_STATUS.get(status, "secondary"),
            className="ms-1",
            style={"fontFamily": FONT_FAMILY_MONO},
        ),
    ]
    if moved:
        children.append(
            html.Span(
                " ⤳",
                title="metric changed after recompute",
                style={"color": COLOR_MUTED},
            )
        )
    return children


def _render_worklist_row(
    *,
    instance_id: str,
    encoded: str,
    key_values: tuple[Any, ...],
    metric: Any,
    status: str,
    is_reviewed: bool,
    is_selected: bool,
    moved: bool,
) -> Component:
    """Render one worklist row."""
    label = " / ".join("∅" if v is None else str(v) for v in key_values)
    children: list[Component] = [
        html.Span(label, style={"fontFamily": FONT_FAMILY_MONO}),
        html.Span(
            id=rids.worklist_row_metric_id(instance_id, encoded),
            children=render_worklist_row_metric_cell(metric, status, moved=moved),
            style={"marginLeft": "auto"},
        ),
    ]
    if is_reviewed:
        children.insert(0, html.Span("✓ ", style={"color": OI_GREEN_TEXT}))
    return html.Button(
        children,
        id=rids.worklist_row_id(instance_id, encoded),
        n_clicks=0,
        className="qc-worklist-row d-flex align-items-center w-100",
        style=_worklist_row_style(is_selected=is_selected, is_reviewed=is_reviewed),
    )


def _worklist_row_style(*, is_selected: bool, is_reviewed: bool) -> dict[str, str]:
    """Return the visual state for a Review worklist row."""
    return {
        "gap": "0.4rem",
        "padding": "0.35rem 0.5rem",
        "border": "none",
        "borderBottom": f"1px solid {COLOR_BORDER}",
        "background": "rgba(0,54,96,0.06)" if is_selected else "transparent",
        "opacity": "0.55" if is_reviewed else "1",
        "fontSize": FONT_SIZE_LABEL,
        "textAlign": "left",
        "cursor": "pointer",
    }


def _worklist_row_styles_for_selection(
    row_ids: list[dict[str, Any]],
    *,
    selected_encoded: str,
    review_state: ReviewState,
) -> list[dict[str, str]]:
    """Return updated row styles for a selected encoded group key."""
    styles: list[dict[str, str]] = []
    for row_id in row_ids:
        encoded = str(row_id.get("key", ""))
        instance_id = str(row_id.get("instance", ""))
        styles.append(
            _worklist_row_style(
                is_selected=encoded == selected_encoded,
                is_reviewed=review_state.is_reviewed(
                    instance_id, decode_group_key(encoded)
                ),
            )
        )
    return styles


def _format_metric(metric: Any) -> str:
    """Format a metric value for display (``nan`` → ``insuf.``)."""
    if metric is None:
        return "insuf."
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return str(metric)
    if value != value:  # NaN
        return "insuf."
    return f"{value:.3f}"


def _render_detail_header(
    key_values: tuple[Any, ...],
    record: dict[str, Any],
    delta: dict[str, Any],
    n_removed: int,
) -> Component:
    """Render the detail-pane group header (key, metric delta, status, n)."""
    label = " / ".join("∅" if v is None else str(v) for v in key_values)
    status = str(record.get("status"))
    n_members = record.get("n_members")

    before = delta.get("before")
    after = delta.get("after")
    if before is not None and after is not None:
        metric_node: Component = html.Span(
            [
                html.Span(_format_metric(before), style={"color": COLOR_MUTED}),
                html.Span(" → "),
                html.Span(_format_metric(after), style={"fontWeight": 600}),
            ]
        )
    else:
        metric_node = html.Span(
            _format_metric(record.get("metric")), style={"fontWeight": 600}
        )

    return html.Div(
        [
            html.Span(label, className="fw-semibold me-3",
                      style={"fontFamily": FONT_FAMILY_MONO}),
            dbc.Badge(status, color=_BADGE_COLOR_BY_STATUS.get(status, "secondary"),
                      className="me-3"),
            html.Span(["metric: ", metric_node], className="me-3"),
            html.Span(f"n={n_members}", className="me-3",
                      style={"color": COLOR_MUTED}),
            html.Span(f"removed={n_removed}", style={"color": COLOR_MUTED}),
        ],
        style={
            "padding": "0.5rem 0",
            "borderBottom": f"1px solid {COLOR_BORDER}",
            "marginBottom": "0.5rem",
        },
    )


def _render_faceted_gallery(
    facets: list[tuple[Any, list[tuple[str, str, int]]]],
    *,
    removed: set[tuple[str, int]],
    crop_size: int,
    display_size: int,
    has_overlay,
    dim_alpha: float = 0.0,
) -> Component:
    """Render the faceted tile gallery: one row per timepoint facet.

    Each facet row is a flat :func:`build_tile_grid` gallery; when there is
    a single ``None`` facet (not a time-course), this collapses to one
    unlabelled gallery.

    Args:
        facets: ``(timepoint, keys)`` pairs (one per facet row).
        removed: ``(image_file, label)`` keys currently removed.
        crop_size: Server crop side length, in pixels.
        display_size: CSS render size, in pixels, for each tile.
        has_overlay: ``(dataset, image_file) -> bool`` overlay probe.
        dim_alpha: Tile-spotlight strength threaded onto each crop URL as
            ``&dim=`` via a :func:`functools.partial` over
            :func:`_qc_crop_url`. ``0.0`` (default) keeps the full-context
            crop.
    """
    url_builder = functools.partial(_qc_crop_url, dim_alpha=dim_alpha)
    rows: list[Component] = []
    single_facet = len(facets) == 1 and facets[0][0] is None
    for timepoint, keys in facets:
        gallery, _order = build_tile_grid(
            keys,
            url_builder,
            selected=set(),
            removed=removed,
            crop_size=crop_size,
            display_size=display_size,
            has_overlay=has_overlay,
            remove_button_builder=_review_tile_remove_button,
        )
        if single_facet:
            rows.append(gallery)
        else:
            rows.append(
                html.Div(
                    [
                        html.Div(
                            f"t = {timepoint}" if timepoint is not None else "t = ?",
                            style={
                                "fontFamily": FONT_FAMILY_MONO,
                                "fontSize": FONT_SIZE_CAPTION,
                                "color": COLOR_NAVY,
                                "marginTop": "0.25rem",
                            },
                        ),
                        gallery,
                    ]
                )
            )
    return html.Div(rows)


# ---------------------------------------------------------------------------
# Shared state plumbing used by multiple callbacks
# ---------------------------------------------------------------------------


def _output_root():
    """Return the active ``OutputRoot`` (or ``None`` outside a viewer app)."""
    return current_app.config.get(CFG_OUTPUT_ROOT)


def _filtered_state():
    """Return the active ``FilteredMeasurements`` (or ``None``)."""
    return current_app.config.get(CFG_FILTERED_STATE)


def _removed_keys_locked() -> set[tuple[str, int]]:
    """Snapshot the removal set under the ``FilteredMeasurements`` lock.

    Reading under the lock makes the recompute see a coherent
    state-at-mark-reviewed rather than a set mid-mutated by a concurrent
    curation callback (spec §D risk refinement).
    """
    filtered = _filtered_state()
    if filtered is None:
        return set()
    with filtered._lock:
        return set(filtered.removed_keys)


def _load_review_state() -> ReviewState:
    """Load the per-module review state for the active output root."""
    output_root = _output_root()
    if output_root is None:
        return ReviewState(path=Path("review_state.json"))
    return ReviewState.load(output_root.root)


def _metric_for_group(
    summary_df: pl.DataFrame | None,
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
) -> Any:
    """Read one group's metric from a (re)loaded summary frame, or ``None``."""
    if summary_df is None:
        return None
    record = _data.group_record(summary_df, instance_id, groupby_cols, key_values)
    return None if record is None else record.get("metric")


def _metric_status_for_group(
    summary_df: pl.DataFrame | None,
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
) -> tuple[Any, str | None]:
    """Read one group's ``(metric, status)`` from a (re)loaded summary frame."""
    if summary_df is None:
        return None, None
    record = _data.group_record(summary_df, instance_id, groupby_cols, key_values)
    if record is None:
        return None, None
    status = record.get("status")
    return record.get("metric"), None if status is None else str(status)


def _recompute_after_curation(
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
    metric_before: Any,
) -> dict[str, Any] | None:
    """Run an in-session per-group recompute and return its before→after delta.

    Reads the curated post-applied frame, runs ``run_qc`` (only — never
    ``finalize_*``, which would wipe ``review_state.json``), reloads the
    rewritten summary, and reports this group's new metric. Returns
    ``None`` (no-op) when no pipeline is available.

    Args:
        instance_id: The module being recomputed.
        groupby_cols: The module's group-key columns.
        key_values: The recomputed group's key.
        metric_before: The group's metric prior to this recompute.

    Returns:
        ``{"before", "after", "status_after", "moved"}`` for the group, or
        ``None``. ``status_after`` is the recomputed QC status straight
        from the rewritten artifact (so the worklist badge flips to the
        authoritative new status — no GUI-side threshold re-derivation).
    """
    output_root = _output_root()
    pipeline = current_app.config.get(CFG_QC_PIPELINE)
    if output_root is None or pipeline is None or not pipeline.get_qc():
        return None

    from phenotypic.tools_._qc_recipe._runner import run_qc

    removed = _removed_keys_locked()
    frame = _data.build_recompute_frame(output_root, removed)
    try:
        run_qc(frame, pipeline, Path(output_root.root))
    except Exception:  # noqa: BLE001 - recompute failure must not crash curation
        logger.warning("In-session QC recompute failed", exc_info=True)
        return None

    new_summary = _data.load_qc_summary(output_root)
    metric_after, status_after = _metric_status_for_group(
        new_summary, instance_id, groupby_cols, key_values
    )
    moved = not _metrics_equal(metric_before, metric_after)
    return {
        "before": metric_before,
        "after": metric_after,
        "status_after": status_after,
        "moved": moved,
    }


def _metrics_equal(a: Any, b: Any) -> bool:
    """Compare two metric values treating NaN==NaN and within float tol."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    a_nan, b_nan = fa != fa, fb != fb
    if a_nan or b_nan:
        return a_nan and b_nan
    return abs(fa - fb) <= 1e-9


def register_review_callbacks(app: dash.Dash) -> None:
    """Register the QC Review sub-view callbacks on *app*.

    Requires the same Flask app config as the Configure callbacks plus
    :data:`CFG_QC_PIPELINE` (for in-session recompute). Safe to call once
    from :func:`.._callbacks.register_qc_callbacks`.

    Args:
        app: The Dash application that will own the callbacks.
    """

    # -----------------------------------------------------------------
    # A. Module picker options (refresh on recipe-revision tick).
    # -----------------------------------------------------------------
    @app.callback(
        Output(rids.QC_REVIEW_MODULE_PICKER_ID, "options"),
        Output(rids.QC_REVIEW_MODULE_PICKER_ID, "value"),
        Input(viewer_ids.STORE_QC_RECIPE_REVISION, "data"),
        State(rids.QC_REVIEW_MODULE_PICKER_ID, "value"),
    )
    def _populate_module_picker(
        _revision: int | None, current: str | None
    ) -> tuple[list[dict[str, str]], str | None]:
        """Populate the module picker from the committed qc_summary artifact."""
        output_root = _output_root()
        if output_root is None:
            return [], None
        summary = _data.load_qc_summary(output_root)
        options = _data.module_options(summary)
        values = {opt["value"] for opt in options}
        value = current if current in values else (
            options[0]["value"] if options else None
        )
        return options, value

    # -----------------------------------------------------------------
    # B. Module switch / re-sort → worklist + summary + frozen order +
    #    initial selection.
    # -----------------------------------------------------------------
    @app.callback(
        Output(rids.QC_REVIEW_WORKLIST_ID, "children"),
        Output(rids.QC_REVIEW_SUMMARY_HEADER_ID, "children"),
        Output(rids.STORE_QC_WORKLIST_ORDER, "data"),
        Output(rids.STORE_QC_SELECTED_GROUP, "data"),
        Output(rids.QC_REVIEW_EMPTY_STATE_ID, "style"),
        Output(rids.QC_REVIEW_MODULE_CHIPS_ID, "children"),
        Input(rids.QC_REVIEW_MODULE_PICKER_ID, "value"),
        Input(rids.QC_REVIEW_RESORT_BTN_ID, "n_clicks"),
        Input(rids.QC_REVIEW_SHOW_FILTER_ID, "value"),
        State(rids.STORE_QC_SELECTED_GROUP, "data"),
        State(rids.STORE_QC_RECOMPUTE_DELTAS, "data"),
    )
    def _render_worklist(
        instance_id: str | None,
        _resort_clicks: int | None,
        show_filter: str,
        selected_encoded: str | None,
        deltas: dict[str, dict[str, Any]] | None,
    ):
        output_root = _output_root()
        if output_root is None or not instance_id:
            return [], [], [], None, {"display": "block", "padding": "2rem",
                                      "textAlign": "center"}, []

        summary = _data.load_qc_summary(output_root)
        if summary is None:
            return [], [], [], None, {"display": "block", "padding": "2rem",
                                      "textAlign": "center"}, []

        groupby_cols = _data.groupby_cols_for(summary, instance_id)
        worklist = _data.module_worklist(summary, instance_id)
        review_state = _load_review_state()
        deltas = deltas or {}

        visible = _apply_show_filter(
            worklist, instance_id, groupby_cols, review_state, show_filter
        )

        order = [
            encode_group_key(tuple(r.get(c) for c in groupby_cols))
            for r in visible.iter_rows(named=True)
        ]
        # Keep the prior selection if still visible, else first visible.
        if selected_encoded not in order:
            selected_encoded = order[0] if order else None

        rows = _render_worklist_rows(
            visible, instance_id, groupby_cols, review_state, deltas,
            selected_encoded,
        )

        stats = _data.summary_stats(_data.module_worklist(summary, instance_id))
        removed = _removed_keys_locked()
        members = _data.load_qc_members(output_root)
        colonies_removed = _count_removed_in_module(
            members, instance_id, removed
        )
        header = _render_summary_header(
            stats, review_state.reviewed_count(instance_id), colonies_removed
        )
        chips = _module_chips(summary, instance_id, groupby_cols)
        empty_style = {"display": "none"}
        return rows, header, order, selected_encoded, empty_style, chips

    # -----------------------------------------------------------------
    # C. Group selection → detail header + faceted gallery.
    #    Fires on worklist row click AND on selection-store change. A row
    #    click that changes the selection only writes the store; the
    #    resulting store-input echo renders the detail once (module switch
    #    and mark/next render through the same store-input path).
    # -----------------------------------------------------------------
    @app.callback(
        Output(rids.QC_REVIEW_DETAIL_HEADER_ID, "children"),
        Output(rids.QC_REVIEW_GALLERY_ID, "children"),
        Output(rids.STORE_QC_SELECTED_GROUP, "data", allow_duplicate=True),
        Output(
            {"type": "qc-worklist-row", "instance": ALL, "key": ALL},
            "style",
            allow_duplicate=True,
        ),
        Input({"type": "qc-worklist-row", "instance": ALL, "key": ALL}, "n_clicks"),
        Input(rids.STORE_QC_SELECTED_GROUP, "data"),
        Input(viewer_ids.STORE_TILE_DIM_ALPHA, "data"),
        State(rids.QC_REVIEW_MODULE_PICKER_ID, "value"),
        State(rids.STORE_QC_RECOMPUTE_DELTAS, "data"),
        State({"type": "qc-worklist-row", "instance": ALL, "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _render_detail(
        _row_clicks: list[int | None],
        selected_encoded: str | None,
        dim_alpha: float | None,
        instance_id: str | None,
        deltas: dict[str, dict[str, Any]] | None,
        row_ids: list[dict[str, Any]] | None,
    ):
        triggered = callback_context.triggered_id
        is_row_click = (
            isinstance(triggered, dict)
            and triggered.get("type") == "qc-worklist-row"
        )
        if is_row_click:
            clicked_key = triggered.get("key")
            # A row click that *changes* the selection only needs to update
            # the store: that write re-fires this callback on the store-input
            # path, which renders the detail once. Rendering here too would
            # paint the identical pane twice. (A re-click of the already-open
            # group leaves the store unchanged and so never echoes, so we
            # still fall through and render to refresh it.)
            if clicked_key != selected_encoded:
                return no_update, no_update, clicked_key, no_update
            selected_encoded = clicked_key
        if not instance_id or not selected_encoded:
            return [], [], no_update, no_update

        output_root = _output_root()
        summary = _data.load_qc_summary(output_root)
        members = _data.load_qc_members(output_root)
        if summary is None or members is None:
            return [], [], selected_encoded, no_update

        groupby_cols = _data.groupby_cols_for(summary, instance_id)
        key_values = decode_group_key(selected_encoded)
        record = _data.group_record(summary, instance_id, groupby_cols, key_values)
        if record is None:
            return [], [], selected_encoded, no_update

        dataset_by_image = _data.dataset_by_image_map(output_root)
        keys = _data.group_member_keys(
            members, instance_id, groupby_cols, key_values, dataset_by_image
        )
        time_by_key = _data.time_by_key_map(output_root)
        facets = _data.facet_keys_by_timepoint(keys, time_by_key)

        removed = _removed_keys_locked()
        n_removed = sum(1 for _ds, im, lbl in keys if (im, lbl) in removed)
        deltas = deltas or {}
        delta = deltas.get(selected_encoded, {})

        alpha = TILE_DIM_DEFAULT if dim_alpha is None else float(dim_alpha)
        header = _render_detail_header(key_values, record, delta, n_removed)
        gallery = _render_faceted_gallery(
            facets,
            removed=removed,
            crop_size=_crop_size_for(keys, output_root),
            display_size=120,
            has_overlay=output_root.has_overlay,
            dim_alpha=alpha,
        )
        # Record last-visited group for this module.
        review_state = _load_review_state()
        review_state.set_last(instance_id, key_values)
        row_styles = _worklist_row_styles_for_selection(
            row_ids or [],
            selected_encoded=selected_encoded,
            review_state=review_state,
        )
        return header, gallery, selected_encoded, row_styles

    # -----------------------------------------------------------------
    # D. Tile-spotlight ``dim`` stepper → shared store. Writes the same
    #    STORE_TILE_DIM_ALPHA the colony stepper writes (allow_duplicate)
    #    so both toolbars drive one strength; the readout sync + both
    #    galleries' renders subscribe to the store.
    # -----------------------------------------------------------------
    @app.callback(
        Output(viewer_ids.STORE_TILE_DIM_ALPHA, "data", allow_duplicate=True),
        Input(rids.QC_REVIEW_DIM_MINUS, "n_clicks"),
        Input(rids.QC_REVIEW_DIM_PLUS, "n_clicks"),
        State(viewer_ids.STORE_TILE_DIM_ALPHA, "data"),
        prevent_initial_call=True,
    )
    def _step_qc_review_dim(
        _minus_clicks: int | None,
        _plus_clicks: int | None,
        current: float | None,
    ) -> float:
        """Step the shared spotlight strength on a Review ``−``/``+`` click.

        Thin adapter over the pure, Dash-free
        :func:`stepped_alpha_from_trigger` helper (direction from
        ``dash.ctx.triggered_id``; clamp/round inside the helper).
        """
        return stepped_alpha_from_trigger(
            dash.ctx.triggered_id,
            current,
            plus_id=rids.QC_REVIEW_DIM_PLUS,
            minus_id=rids.QC_REVIEW_DIM_MINUS,
        )

    _register_curation_callbacks(app)
    _register_review_progress_callbacks(app)
    _register_worklist_row_metric_callback(app)
    _register_sidebar_callbacks(app)


# ---------------------------------------------------------------------------
# Worklist / summary helpers
# ---------------------------------------------------------------------------


def _apply_show_filter(
    worklist: pl.DataFrame,
    instance_id: str,
    groupby_cols: list[str],
    review_state: ReviewState,
    show_filter: str,
) -> pl.DataFrame:
    """Filter the frozen worklist by the toolbar's Show selector.

    ``unreviewed`` hides groups already marked reviewed; ``fail_warn``
    keeps only failing/warning groups; ``all`` keeps everything. Order is
    always preserved (the worklist is already worst-first / frozen).
    """
    if show_filter == rids.QC_SHOW_FAIL_WARN and "status" in worklist.columns:
        return worklist.filter(pl.col("status").is_in(["fail", "warn"]))
    if show_filter == rids.QC_SHOW_UNREVIEWED:
        keep_mask = [
            not review_state.is_reviewed(
                instance_id, tuple(r.get(c) for c in groupby_cols)
            )
            for r in worklist.iter_rows(named=True)
        ]
        if not any(keep_mask):
            return worklist.clear()
        return worklist.filter(pl.Series(keep_mask))
    return worklist


def _count_removed_in_module(
    members: pl.DataFrame | None,
    instance_id: str,
    removed: set[tuple[str, int]],
) -> int:
    """Count distinct removed colonies that belong to this module's members."""
    if members is None or members.is_empty() or not removed:
        return 0
    slice_df = members.filter(pl.col("instance_id") == instance_id)
    count = 0
    seen: set[tuple[str, int]] = set()
    for image_file, label in zip(
        slice_df.get_column("Metadata_ImageFile").to_list(),
        slice_df.get_column("Object_Label").to_list(),
    ):
        key = (str(image_file), int(label))
        if key in removed and key not in seen:
            seen.add(key)
            count += 1
    return count


def _module_chips(
    summary: pl.DataFrame, instance_id: str, groupby_cols: list[str]
) -> list[Component]:
    """Render the read-only ``class`` + ``groupby`` chips for the module."""
    record = summary.filter(pl.col("instance_id") == instance_id).head(1)
    cls = (
        str(record.get_column("class")[0])
        if not record.is_empty()
        else "?"
    )
    chips: list[Component] = [
        dbc.Badge(cls, color="light", text_color="dark", className="me-1")
    ]
    if groupby_cols:
        chips.append(
            html.Span(
                f"groupby: {', '.join(groupby_cols)}",
                style={"marginLeft": "0.5rem"},
            )
        )
    return chips


def _crop_size_for(keys: list[tuple[str, str, int]], output_root) -> int:
    """Pick a server crop side length covering the group's bounding boxes.

    Reuses the colony-view sizing convention (max bbox extent + padding,
    floored) by reading the bbox columns from the master frame for the
    group's members. Falls back to a sensible default when bbox columns
    are absent.
    """
    default = 160
    master = output_root.master_df
    bbox_cols = ("Bbox_MinRR", "Bbox_MaxRR", "Bbox_MinCC", "Bbox_MaxCC")
    if not all(c in master.columns for c in bbox_cols) or not keys:
        return default
    member_keys = {(im, lbl) for _ds, im, lbl in keys}
    subset = master.filter(
        pl.struct(["Metadata_ImageFile", "Object_Label"]).map_elements(
            lambda s: (str(s["Metadata_ImageFile"]), int(s["Object_Label"]))
            in member_keys,
            return_dtype=pl.Boolean,
        )
    )
    if subset.is_empty():
        return default
    extents = subset.select(
        (pl.col("Bbox_MaxRR") - pl.col("Bbox_MinRR")).alias("rr"),
        (pl.col("Bbox_MaxCC") - pl.col("Bbox_MinCC")).alias("cc"),
    )
    max_rr = extents.get_column("rr").max()
    max_cc = extents.get_column("cc").max()
    if max_rr is None or max_cc is None:
        return default
    return max(default, int(max(int(max_rr), int(max_cc))) + 16)


# ---------------------------------------------------------------------------
# Curation (shared FilteredMeasurements removal set)
# ---------------------------------------------------------------------------


def toggle_review_tile(filtered, image_file: str, label: int) -> list[list]:
    """Toggle one colony's removal and return the new ``STORE_REMOVED_KEYS``.

    Module-level (not a callback closure) so the
    :meth:`FilteredMeasurements.mutate_and_payload` contract — the action
    receives the state instance — is unit-testable without booting Dash.

    Args:
        filtered: The shared :class:`FilteredMeasurements`.
        image_file: ``Metadata_ImageFile`` of the colony.
        label: ``Object_Label`` of the colony.

    Returns:
        The updated removed-keys payload.
    """
    return filtered.mutate_and_payload(
        lambda state: state.toggle(image_file, label)
    )


def bulk_review_curation(
    filtered, remove: bool, selected: list[tuple[str, int]]
) -> list[list]:
    """Remove or restore the selected colonies; return the new payload.

    Args:
        filtered: The shared :class:`FilteredMeasurements`.
        remove: ``True`` to remove the selection, ``False`` to restore it.
        selected: The ``(image_file, label)`` keys to act on.

    Returns:
        The updated removed-keys payload.
    """

    def _apply(state) -> None:
        if remove:
            state.remove_many(selected)
        else:
            state.restore_many(selected)

    return filtered.mutate_and_payload(_apply)


def _register_curation_callbacks(app: dash.Dash) -> None:
    """Register Review per-tile + bulk remove/restore callbacks.

    These mutate the **same** ``FilteredMeasurements`` removal set and the
    **same** ``STORE_REMOVED_KEYS`` store the colony view writes, so a
    colony removed in Review is removed everywhere (spec §D.4). They share
    the JS multi-select layer via ``STORE_COLONY_SELECTION`` (Review tiles
    carry the same ``data-key`` checkbox class as colony tiles). The
    mutation bodies live in :func:`toggle_review_tile` /
    :func:`bulk_review_curation` so the ``mutate_and_payload`` contract is
    unit-tested.
    """

    @app.callback(
        Output(viewer_ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Input(
            {"type": "qc-review-tile-remove", "image_file": MATCH, "label": MATCH},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _toggle_review_tile(_clicks: int | None):
        """Toggle one colony's removal from a Review-gallery tile button."""
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        filtered = _filtered_state()
        if filtered is None:
            return no_update
        raw_label = triggered.get("label")
        if raw_label is None:
            return no_update
        return toggle_review_tile(
            filtered, str(triggered.get("image_file")), int(raw_label)
        )

    @app.callback(
        Output(viewer_ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Output(viewer_ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Input(rids.QC_REVIEW_BULK_REMOVE_BTN_ID, "n_clicks"),
        Input(rids.QC_REVIEW_BULK_RESTORE_BTN_ID, "n_clicks"),
        State(viewer_ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _bulk_review_curation(
        _remove_clicks: int | None,
        _restore_clicks: int | None,
        selection_payload: Any,
    ):
        """Apply remove/restore to the multi-selected Review tiles, then clear."""
        triggered = callback_context.triggered_id
        filtered = _filtered_state()
        if filtered is None or triggered is None:
            return no_update, no_update
        selected = decode_removed_keys_payload(
            (selection_payload or {}).get("selected")
        )
        if not selected:
            return no_update, no_update
        payload = bulk_review_curation(
            filtered, triggered == rids.QC_REVIEW_BULK_REMOVE_BTN_ID, selected
        )
        return payload, {"selected": []}


# ---------------------------------------------------------------------------
# Review-progress callbacks (mark reviewed / next + recompute)
# ---------------------------------------------------------------------------


def _register_review_progress_callbacks(app: dash.Dash) -> None:
    """Register mark-reviewed / next callbacks with per-group recompute.

    On mark-reviewed (or advancing past) a group **that was curated**, an
    in-session recompute runs (``run_qc`` only) on the post-applied frame
    minus removals; the group's metric/badge update in place via the
    recompute-deltas store (consumed by the worklist + detail callbacks) —
    the queue order never changes here (only ↻ Re-sort reorders it).
    """

    @app.callback(
        Output(rids.STORE_QC_RECOMPUTE_DELTAS, "data", allow_duplicate=True),
        Output(rids.STORE_QC_SELECTED_GROUP, "data", allow_duplicate=True),
        Input(rids.QC_REVIEW_MARK_REVIEWED_BTN_ID, "n_clicks"),
        Input(rids.QC_REVIEW_PREV_BTN_ID, "n_clicks"),
        Input(rids.QC_REVIEW_NEXT_BTN_ID, "n_clicks"),
        State(rids.QC_REVIEW_MODULE_PICKER_ID, "value"),
        State(rids.STORE_QC_SELECTED_GROUP, "data"),
        State(rids.STORE_QC_WORKLIST_ORDER, "data"),
        State(rids.STORE_QC_RECOMPUTE_DELTAS, "data"),
        prevent_initial_call=True,
    )
    def _mark_or_next(
        _mark_clicks: int | None,
        _prev_clicks: int | None,
        _next_clicks: int | None,
        instance_id: str | None,
        selected_encoded: str | None,
        order: list[str] | None,
        deltas: dict[str, dict[str, Any]] | None,
    ):
        triggered = callback_context.triggered_id
        if not instance_id or not selected_encoded:
            return no_update, no_update

        deltas = dict(deltas or {})
        if triggered == rids.QC_REVIEW_PREV_BTN_ID:
            if not order:
                return deltas, selected_encoded
            return deltas, _previous_group(order, selected_encoded)

        output_root = _output_root()
        if output_root is None:
            return no_update, no_update

        summary = _data.load_qc_summary(output_root)
        groupby_cols = (
            _data.groupby_cols_for(summary, instance_id) if summary is not None
            else []
        )
        key_values = decode_group_key(selected_encoded)
        review_state = _load_review_state()

        # Did this group get curated? (any member currently removed)
        curated = _group_has_removed_members(
            output_root, instance_id, groupby_cols, key_values
        )

        # Mark reviewed (explicit button, or auto on advancing a curated group).
        is_next = triggered == rids.QC_REVIEW_NEXT_BTN_ID
        if triggered == rids.QC_REVIEW_MARK_REVIEWED_BTN_ID or (
            is_next and curated
        ):
            review_state.mark_reviewed(instance_id, key_values)

        # Recompute only when changes were made (spec §D.5).
        if curated:
            metric_before = _metric_for_group(
                summary, instance_id, groupby_cols, key_values
            )
            delta = _recompute_after_curation(
                instance_id, groupby_cols, key_values, metric_before
            )
            if delta is not None:
                deltas[selected_encoded] = delta

        # Advance to the next unreviewed group in the frozen order on "next".
        next_encoded = selected_encoded
        if is_next and order:
            next_encoded = _next_unreviewed(
                order, selected_encoded, instance_id, review_state
            )

        return deltas, next_encoded


def _group_has_removed_members(
    output_root,
    instance_id: str,
    groupby_cols: list[str],
    key_values: tuple[Any, ...],
) -> bool:
    """Return ``True`` if any of the group's member colonies are removed."""
    members = _data.load_qc_members(output_root)
    if members is None:
        return False
    dataset_by_image = _data.dataset_by_image_map(output_root)
    keys = _data.group_member_keys(
        members, instance_id, groupby_cols, key_values, dataset_by_image
    )
    removed = _removed_keys_locked()
    return any((im, lbl) in removed for _ds, im, lbl in keys)


def _next_unreviewed(
    order: list[str],
    current_encoded: str,
    instance_id: str,
    review_state: ReviewState,
) -> str:
    """Return the next not-yet-reviewed encoded key after the current one.

    Wraps within the frozen order; falls back to the current key when
    every other group is already reviewed.
    """
    if current_encoded not in order:
        return current_encoded
    start = order.index(current_encoded)
    n = len(order)
    for offset in range(1, n + 1):
        candidate = order[(start + offset) % n]
        if candidate == current_encoded:
            break
        if not review_state.is_reviewed(
            instance_id, decode_group_key(candidate)
        ):
            return candidate
    return current_encoded


def _previous_group(order: list[str], current_encoded: str) -> str:
    """Return the previous encoded key in frozen visible order, wrapping."""
    if not order or current_encoded not in order:
        return current_encoded
    start = order.index(current_encoded)
    return order[(start - 1) % len(order)]


# ---------------------------------------------------------------------------
# In-place worklist-row metric/badge update (after recompute)
# ---------------------------------------------------------------------------


def worklist_row_metric_update(
    delta: dict[str, Any] | None, fallback_status: str | None = None
) -> list[Component] | Any:
    """Return the in-place metric-cell children for a recompute delta, or no-op.

    Module-level + pure so the per-row update callback is unit-testable
    without booting Dash. Given a group's recompute ``delta``, renders the
    ``after`` metric + recomputed ``status_after`` badge (with the ``⤳``
    changed hint when ``moved``) so the frozen worklist row reflects the
    recompute **in place** — same span, no reorder, no full-list flash.
    Returns ``dash.no_update`` when there is no delta for this row (so a
    recompute on group A never repaints group B's cell).

    Args:
        delta: This group's recompute delta
            (``{"after", "status_after", "moved"}``), or ``None`` / ``{}``
            when the group was never recomputed.
        fallback_status: Status to use when the delta omits
            ``status_after`` (a partial recompute) — keeps the badge from
            blanking. ``None`` falls back to ``"insufficient"`` (neutral).

    Returns:
        The new cell children list, or ``dash.no_update``.
    """
    if not delta:
        return no_update
    status = delta.get("status_after") or fallback_status or "insufficient"
    return render_worklist_row_metric_cell(
        delta.get("after"), str(status), moved=bool(delta.get("moved"))
    )


def _register_worklist_row_metric_callback(app: dash.Dash) -> None:
    """Register the per-row in-place metric/badge update (spec §D.5).

    A ``MATCH`` callback keyed on the worklist row's ``key`` (encoded group
    key) listens to :data:`STORE_QC_RECOMPUTE_DELTAS` and rewrites **only**
    that row's metric-cell ``children`` (metric span + status badge + ⤳
    hint) when a delta exists for it. Targeting the per-row metric span —
    not the whole worklist — preserves the frozen order, the scroll
    position, and the current selection: no row is re-created, so the
    sticky sidebar never jumps and the open group stays open. Rows with no
    delta short-circuit to ``no_update`` (their cell is untouched).
    """

    @app.callback(
        Output(
            {"type": "qc-worklist-row-metric", "instance": MATCH, "key": MATCH},
            "children",
        ),
        Input(rids.STORE_QC_RECOMPUTE_DELTAS, "data"),
        prevent_initial_call=True,
    )
    def _update_worklist_row_metric(deltas: dict[str, dict[str, Any]] | None):
        """Update one worklist row's metric/badge in place from its delta."""
        triggered_output = callback_context.outputs_list
        encoded = _encoded_key_from_output(triggered_output)
        if encoded is None:
            return no_update
        delta = (deltas or {}).get(encoded)
        return worklist_row_metric_update(delta)


def _encoded_key_from_output(outputs_list: Any) -> str | None:
    """Recover the matched row's encoded group key from the callback output id.

    A ``MATCH`` output's ``outputs_list`` carries the concrete id the
    wildcard resolved to; the ``key`` field is the encoded group key the
    row was rendered with. Returns ``None`` if the shape is unexpected (so
    the callback degrades to a no-op rather than raising).
    """
    entry = outputs_list
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not isinstance(entry, dict):
        return None
    component_id = entry.get("id")
    if not isinstance(component_id, dict):
        return None
    key = component_id.get("key")
    return key if isinstance(key, str) else None


# ---------------------------------------------------------------------------
# Sidebar collapse / expand
# ---------------------------------------------------------------------------


def sidebar_layout_state(
    collapsed: bool, width_px: object
) -> tuple[dict[str, str], dict[str, str], str]:
    """Return (sidebar wrapper style, worklist style, chevron glyph) for a state.

    Pure + module-level so the layout logic is unit-testable without a
    Dash app. Combines BOTH the collapse flag and the user's dragged width
    (the JS splitter persists px to ``STORE_QC_SIDEBAR_WIDTH``) into one
    worklist style, so a single callback owns ``worklist.style`` and the
    two stores can never fight over it:

    * collapsed → worklist hidden, wrapper shrinks to a thin chevron rail
      (the detail/gallery pane, ``flex: 1 1 auto``, reclaims the freed
      width); chevron ``▶`` (click to expand).
    * expanded → worklist shown at the clamped ``width_px``; chevron ``◀``.

    The width is applied even when collapsed (display:none), so expanding
    restores the user's dragged width rather than the default.

    Args:
        collapsed: Whether the sidebar is collapsed.
        width_px: The persisted sidebar width (clamped via
            :func:`clamp_sidebar_width`).

    Returns:
        ``(sidebar_style, worklist_style, chevron_text)``.
    """
    width = clamp_sidebar_width(width_px)
    worklist_style: dict[str, str] = {
        "width": f"{width}px",
        "overflow": "auto",
        "maxHeight": f"calc(100vh - {_SUMMARY_HEADER_HEIGHT} - 2rem)",
        "padding": "0.5rem",
        "display": "none" if collapsed else "block",
    }
    if collapsed:
        return collapsed_sidebar_style(), worklist_style, "▶"
    return expanded_sidebar_style(), worklist_style, "◀"


def _register_sidebar_callbacks(app: dash.Dash) -> None:
    """Register the worklist sidebar collapse + resize callback.

    A SINGLE callback owns ``worklist.style`` so the collapse flag and the
    dragged width never fight over it. Fires on the chevron click (which
    flips :data:`STORE_QC_SIDEBAR_COLLAPSED`) and on
    :data:`STORE_QC_SIDEBAR_WIDTH` changes (the JS drag-splitter persists
    the dragged px on mouse-up). The detail/gallery pane reclaims any
    freed width automatically via its ``flex: 1 1 auto`` sizing.
    """

    @app.callback(
        Output(rids.STORE_QC_SIDEBAR_COLLAPSED, "data"),
        Output(rids.QC_REVIEW_SIDEBAR_ID, "style"),
        Output(rids.QC_REVIEW_WORKLIST_ID, "style"),
        Output(rids.QC_REVIEW_SIDEBAR_TOGGLE_ID, "children"),
        Input(rids.QC_REVIEW_SIDEBAR_TOGGLE_ID, "n_clicks"),
        Input(rids.STORE_QC_SIDEBAR_WIDTH, "data"),
        State(rids.STORE_QC_SIDEBAR_COLLAPSED, "data"),
        prevent_initial_call=True,
    )
    def _apply_sidebar_layout(
        _clicks: int | None,
        width_px: object,
        collapsed: bool | None,
    ):
        # Only the chevron toggles collapsed; a width-store change (drag)
        # keeps the current collapsed state.
        triggered = callback_context.triggered_id
        new_collapsed = (
            not bool(collapsed)
            if triggered == rids.QC_REVIEW_SIDEBAR_TOGGLE_ID
            else bool(collapsed)
        )
        sidebar_style, worklist_style, glyph = sidebar_layout_state(
            new_collapsed, width_px
        )
        return new_collapsed, sidebar_style, worklist_style, glyph


__all__ = [
    "register_review_callbacks",
    "toggle_review_tile",
    "bulk_review_curation",
    "sidebar_layout_state",
    "render_worklist_row_metric_cell",
    "worklist_row_metric_update",
    "_previous_group",
]


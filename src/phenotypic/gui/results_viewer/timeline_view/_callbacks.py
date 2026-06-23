"""Dash callbacks for the results-viewer Timeline tab.

Wires the focus-and-navigate overlay matrix to the SAME filter slice the
colony tab uses (``FilterSpec.from_store(...).apply_to(master_df)``):

1. **Axis dropdowns** — Y (row) via the uncapped ``selectable_axis_columns``
   (``max_cardinality=None``, spec §16.5); X (time) via ``selectable_time_columns``
   (name/dtype-gated, uncapped, spec §15.2). Both react to the filter spec so
   the offcanvas filters the timeline too.
2. **Render** — gated on ``active_tab == TAB_TIMELINE_ID`` (mirroring the Error
   tab's gate) so the polars + Component work never runs for a hidden subtree.
   Builds the grid via :func:`build_timeline_grid_component`, toggles the
   empty-state + large-axis warning, and sets the warning text.
3. **Tile-size stepper** — reuses ``stepped_timeline_tile_size_from_trigger``.
4. **Clientside attach** — calls ``window.__phenotypicTimeline.attach("timeline-grid")``
   after each grid render + on tab activation; the byte-identical controller
   resets focus to the first populated cell and renders the centered window.
5. **Pop-out** — the controller writes the focused cell's ``<dataset/stem>#<nonce>``
   into the hidden ``.timeline-popout-bridge``; a server callback strips the
   nonce (Phase 2 convention), decodes ``(dataset, stem)``, opens the modal, and
   a clientside callback mounts an OSD viewer at the existing ``/tiles`` DZI URL.
"""
from __future__ import annotations

import logging
from typing import Any

import dash
from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from phenotypic.gui._config import (
    CFG_URL_PREFIX,
    MOUNT_HOME,
    TIMELINE_TILE_SIZE_DEFAULT,
    stepped_timeline_tile_size_from_trigger,
)
from phenotypic.gui.browse._callbacks import strip_popout_nonce
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns
from phenotypic.gui.results_viewer.timeline_view import _ids as ids
from phenotypic.gui.results_viewer.timeline_view._grid import (
    is_large_time_axis,
    selectable_time_columns,
)
from phenotypic.gui.results_viewer.timeline_view._layout import (
    build_timeline_grid_component,
)
from phenotypic.gui.results_viewer.timeline_view._thumb_routes import decode_cell_ref

logger = logging.getLogger(__name__)

__all__ = ["register_callbacks"]

#: Clientside callbacks need an Output sink even when the JS returns nothing
#: useful; we write a throwaway data-attr on the grid container (it never
#: disturbs the controller's transform/highlight mutations on the same node).
_ATTACH_SINK_ATTR = "data-attach-sync"


def _prefer_default(options: list[str], current: str | None) -> str | None:
    """Keep ``current`` if still valid, else fall back to the first option."""
    if current in options:
        return current
    return options[0] if options else None


def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register every Timeline-tab callback on ``app``.

    Args:
        app: The Dash application that will own the callbacks.
        output_root: Validated handle on the CLI output directory; passed by
            closure into the dropdown + render callbacks.
    """
    df = output_root.master_df
    column_value_sets = output_root.column_value_sets

    def _slice(filter_payload: Any):
        try:
            spec = FilterSpec.from_store(filter_payload)
            return spec.apply_to(df)
        except Exception:
            logger.exception("FilterSpec.apply_to failed in timeline-view.")
            return df

    # ----------------------------------------------------------------------
    # 1. Axis dropdowns (filter-aware)
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.TIMELINE_Y_DROPDOWN, "options"),
        Output(ids.TIMELINE_X_DROPDOWN, "options"),
        Output(ids.TIMELINE_Y_DROPDOWN, "value"),
        Output(ids.TIMELINE_X_DROPDOWN, "value"),
        Input(viewer_ids.STORE_FILTER_SPEC, "data"),
        State(ids.TIMELINE_Y_DROPDOWN, "value"),
        State(ids.TIMELINE_X_DROPDOWN, "value"),
    )
    def _populate_axis_dropdowns(
        filter_payload: Any,
        current_y: str | None,
        current_x: str | None,
    ) -> tuple[
        list[dict[str, str]], list[dict[str, str]], str | None, str | None
    ]:
        """Refresh Y/X axis dropdown options when the filter spec changes.

        Y reuses the uncapped ``selectable_axis_columns`` so a high-cardinality
        grouping (e.g. ``Metadata_PlateNum``) stays selectable; X uses
        ``selectable_time_columns`` (the first time option, sorted to prefer a
        ``Metadata_*`` time-like name, defaults the X axis).
        """
        filtered = _slice(filter_payload)
        y_cols = selectable_axis_columns(
            filtered, column_value_sets, max_cardinality=None
        )
        x_cols = selectable_time_columns(filtered, column_value_sets)
        y_options = [{"label": c, "value": c} for c in y_cols]
        x_options = [{"label": c, "value": c} for c in x_cols]
        return (
            y_options,
            x_options,
            _prefer_default(y_cols, current_y),
            _prefer_default(x_cols, current_x),
        )

    # ----------------------------------------------------------------------
    # 2. Render the grid (gated on the Timeline tab being active)
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.TIMELINE_GRID, "children"),
        Output(ids.TIMELINE_EMPTY_STATE, "style"),
        Output(ids.TIMELINE_LARGE_AXIS_WARNING, "children"),
        Output(ids.TIMELINE_LARGE_AXIS_WARNING, "is_open"),
        Input(viewer_ids.STORE_FILTER_SPEC, "data"),
        Input(ids.TIMELINE_Y_DROPDOWN, "value"),
        Input(ids.TIMELINE_X_DROPDOWN, "value"),
        Input(ids.TIMELINE_STORE_TILE_SIZE, "data"),
        Input(viewer_ids.TABS_ID, "active_tab"),
    )
    def _render_timeline_grid(
        filter_payload: Any,
        y_axis: str | None,
        x_axis: str | None,
        tile_size: int | None,
        active_tab: str | None,
    ) -> tuple[Any, Any, Any, Any]:
        """Rebuild the timeline grid + empty-state + warning on any data change.

        Short-circuits with ``no_update`` off-tab so we don't pay the polars +
        Component work for a hidden subtree (mirrors the colony/Error gates).
        """
        if active_tab != viewer_ids.TAB_TIMELINE_ID:
            return no_update, no_update, no_update, no_update

        filtered = _slice(filter_payload)
        size = int(tile_size) if tile_size else TIMELINE_TILE_SIZE_DEFAULT
        prefix = app.server.config.get(CFG_URL_PREFIX, MOUNT_HOME)
        component, show_empty, n_time = build_timeline_grid_component(
            output_root,
            filtered,
            row_col=y_axis,
            time_col=x_axis,
            tile_size=size,
            url_prefix=prefix,
        )
        empty_style = {"display": "block"} if show_empty else {"display": "none"}
        large = is_large_time_axis(n_time)
        warning_text = (
            f"This time axis has {n_time} points — rendering may be dense; "
            "time-bucketing is not yet available."
            if large
            else ""
        )
        return component, empty_style, warning_text, large

    # ----------------------------------------------------------------------
    # 3. Tile-size stepper
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.TIMELINE_TILE_SIZE_READOUT, "children"),
        Output(ids.TIMELINE_STORE_TILE_SIZE, "data"),
        Input(ids.TIMELINE_TILE_SIZE_MINUS, "n_clicks"),
        Input(ids.TIMELINE_TILE_SIZE_PLUS, "n_clicks"),
        State(ids.TIMELINE_STORE_TILE_SIZE, "data"),
        prevent_initial_call=True,
    )
    def _step_tile_size(
        _minus: int | None, _plus: int | None, current: int | None
    ) -> tuple[str, int]:
        """Step the rendered tile size on a ``−``/``+`` click."""
        size = stepped_timeline_tile_size_from_trigger(
            ctx.triggered_id,
            current,
            plus_id=ids.TIMELINE_TILE_SIZE_PLUS,
            minus_id=ids.TIMELINE_TILE_SIZE_MINUS,
        )
        return f"{size} px", size

    # ----------------------------------------------------------------------
    # 4. Clientside attach — re-fire the focus-navigate controller after each
    #    grid render AND on tab activation. attach() is idempotent + the
    #    <body> MutationObserver re-attaches on a fresh Dash re-render, so this
    #    explicit poke covers the first paint + any data-driven re-render.
    # ----------------------------------------------------------------------
    app.clientside_callback(
        """
        function(_children, _activeTab) {
            if (window.__phenotypicTimeline) {
                window.__phenotypicTimeline.attach("%s");
            }
            return "";
        }
        """
        % ids.TIMELINE_GRID,
        Output(ids.TIMELINE_GRID, _ATTACH_SINK_ATTR),
        Input(ids.TIMELINE_GRID, "children"),
        Input(viewer_ids.TABS_ID, "active_tab"),
    )

    # ----------------------------------------------------------------------
    # 5. Pop-out — server side: decode the bridge value → open + store
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.TIMELINE_POPOUT_MODAL, "is_open"),
        Output(ids.TIMELINE_POPOUT_STORE, "data"),
        Input(ids.TIMELINE_POPOUT_INPUT, "value"),
    )
    def _open_popout(raw_value: str | None) -> tuple[Any, Any]:
        """Open the pop-out for the cell whose ref the controller bridged.

        The hidden bridge ``dcc.Input(value="")`` fires this on first load with
        ``""`` — guard so the modal never flickers open empty. The controller
        appends a ``#<nonce>`` suffix so a same-cell re-open still changes the
        value; strip it (Phase 2 convention) before decoding ``(dataset, stem)``.
        """
        if not raw_value:
            raise PreventUpdate
        identity = strip_popout_nonce(raw_value)
        dataset, stem = decode_cell_ref(identity)
        if not (dataset and stem):
            raise PreventUpdate
        return True, {"dataset": dataset, "stem": stem}

    # ----------------------------------------------------------------------
    # 5b. Pop-out — clientside: mount the OSD deep-zoom viewer at the DZI URL
    # ----------------------------------------------------------------------
    # Reuses the viewer's existing /tiles DZI route + OSD bridge
    # (window.__phenotypicResultsViewer.mountViewer). appPrefix already ends in
    # "/", so the URL is "tiles/…" not "/tiles/…" — matching applyImageSelection.
    app.clientside_callback(
        """
        function(payload) {
            const ns = window.__phenotypicResultsViewer;
            if (!ns || !ns.mountViewer || !payload
                || !payload.dataset || !payload.stem) {
                return window.dash_clientside.no_update;
            }
            const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
                && window.__phenotypicAppPrefix.length > 0)
                ? window.__phenotypicAppPrefix : "/";
            const dziUrl = appPrefix + "tiles/"
                + encodeURIComponent(payload.dataset) + "/"
                + encodeURIComponent(payload.stem) + ".dzi";
            // Defer one frame so the modal body's OSD div is in the DOM.
            requestAnimationFrame(function () {
                ns.mountViewer("%s", dziUrl);
            });
            return "";
        }
        """
        % ids.TIMELINE_POPOUT_OSD,
        Output(ids.TIMELINE_POPOUT_OSD_SYNC, "data"),
        Input(ids.TIMELINE_POPOUT_STORE, "data"),
    )

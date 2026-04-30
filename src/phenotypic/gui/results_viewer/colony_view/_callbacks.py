"""Dash callbacks driving the colony-view tab.

Owns every callback whose Output is confined to the colony-view subtree:

1. **Render** — rebuild the grid component when the filter spec, removal
   set, selection, axis dropdowns, refresh button, or active tab change.
2. **Axis dropdown population** — keep the X/Y axis dropdown options in
   sync with the filtered frame's selectable columns.
3. **Single-cell remove** — toggle a colony's curated-removal state
   when its per-cell × / ↺ button fires.
4. **Selection-delta consumer** — fold the JS-emitted shift/click delta
   into the canonical multi-select store (with anchor tracking).
5. **Bulk-bar visibility / count label** — show or hide the bulk action
   bar based on the active selection size.
6. **Bulk Remove / Restore** — apply :meth:`FilteredMeasurements.remove_many`
   or :meth:`restore_many` to the entire selection, then clear it.
7. **Bulk Clear** — reset the selection store without touching the
   curated set.

The selection-store payload shape is ``{"anchor": [img, label] | None,
"selected": list[[img, label]]}`` — tuples must be lists for JSON
serialisation through ``dcc.Store``.
"""

from __future__ import annotations

import logging
from typing import Any

import dash
from dash import ALL, Input, Output, State, callback_context, no_update

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._filtered_state import (
    FilteredMeasurements,
    decode_removed_keys_payload,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    build_grid,
    compute_max_bbox_size,
    expand_range,
    selectable_axis_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


# Empty selection payload — used by Clear, layout-change reset, and the
# post-bulk-action emission. Module-level so all three sites agree.
_EMPTY_SELECTION: dict[str, Any] = {"anchor": None, "selected": []}


def _coerce_anchor(payload: Any) -> tuple[str, int] | None:
    """Coerce a single ``[img, label]`` payload into a tuple, or None."""
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    try:
        return (str(payload[0]), int(payload[1]))
    except (TypeError, ValueError):
        return None


def _selection_payload(
    anchor: tuple[str, int] | None, selected: list[tuple[str, int]]
) -> dict[str, Any]:
    """Build the canonical selection-store payload (lists, not tuples)."""
    return {
        "anchor": [anchor[0], anchor[1]] if anchor is not None else None,
        "selected": [[img, label] for img, label in selected],
    }


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(
    app: dash.Dash,
    output_root: OutputRoot,
    filtered_state: FilteredMeasurements,
) -> None:
    """Register every callback owned by the colony-view tab.

    Args:
        app: The Dash application that will own the callbacks.
        output_root: Validated handle on the CLI output directory; passed
            by closure into every callback that needs to slice
            ``master_df`` or call :meth:`OutputRoot.has_overlay`.
        filtered_state: The viewer's curation layer; mutated in place by
            the single-cell and bulk action callbacks. Each public mutator
            on :class:`FilteredMeasurements` auto-saves, so callbacks do
            not need to call :meth:`FilteredMeasurements.save` manually.
    """
    df = output_root.master_df
    column_value_sets = output_root.column_value_sets

    # ----------------------------------------------------------------------
    # 1. Render the grid
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.COLONY_GRID_CONTAINER_ID, "children"),
        Output(ids.COLONY_CROP_SIZE_INFO_ID, "children"),
        Output(ids.STORE_COLONY_GRID_ORDER, "data"),
        Input(ids.STORE_FILTER_SPEC, "data"),
        Input(ids.STORE_REMOVED_KEYS, "data"),
        Input(ids.COLONY_X_AXIS_DROPDOWN_ID, "value"),
        Input(ids.COLONY_Y_AXIS_DROPDOWN_ID, "value"),
        Input(ids.COLONY_BTN_REFRESH_ID, "n_clicks"),
        Input(ids.COLONY_TILE_SIZE_SLIDER_ID, "value"),
        Input(ids.TABS_ID, "active_tab"),
    )
    def _render_colony_grid(
        filter_payload: Any,
        removed_payload: Any,
        x_axis: str | None,
        y_axis: str | None,
        refresh_clicks: int | None,
        tile_size: int | None,
        active_tab: str | None,
    ) -> tuple[Any, Any, Any]:
        """Rebuild the grid whenever any of its data inputs change.

        Selection-only changes (which fire frequently as the user
        shift-clicks) DO NOT trigger this callback; instead a
        clientside callback toggles the ``is-selected`` class on the
        existing DOM. This keeps the heavy polars + Component synthesis
        work off the click hot path.

        Short-circuits with ``no_update`` when the colony tab is not
        active so we don't pay the polars work for a hidden subtree.
        """
        del refresh_clicks  # button is a trigger; n_clicks value is unused.

        if active_tab != ids.TAB_COLONY_ID:
            return no_update, no_update, no_update

        if not x_axis or not y_axis:
            return [], "", []

        try:
            spec = FilterSpec.from_store(filter_payload)
            filtered_df = spec.apply_to(df)
        except Exception:
            logger.exception("FilterSpec.apply_to failed in colony-view render.")
            filtered_df = df

        max_size = compute_max_bbox_size(filtered_df)
        # Display size: clamp the slider value into the slider's own range
        # then cap at the server crop size so the browser never upscales
        # the PNG beyond its native resolution.
        slider_value = int(tile_size) if tile_size else 150
        slider_value = max(64, min(400, slider_value))
        display_size = min(slider_value, max_size)
        removed_keys = set(decode_removed_keys_payload(removed_payload))

        # Selection styling is applied by the JS lifecycle layer, so we
        # always render the grid as if nothing were selected. This
        # keeps the click hot path off the server.
        component, grid_order = build_grid(
            filtered_df,
            x_axis,
            y_axis,
            max_size,
            removed_keys,
            set(),
            output_root,
            display_size=display_size,
        )
        info = (
            f"crop {max_size}px → {display_size}px "
            f"({filtered_df.height} colonies)"
        )
        order_payload = [[img, label] for img, label in grid_order]
        return component, info, order_payload

    # ----------------------------------------------------------------------
    # 2. Axis dropdown population
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.COLONY_X_AXIS_DROPDOWN_ID, "options"),
        Output(ids.COLONY_Y_AXIS_DROPDOWN_ID, "options"),
        Output(ids.COLONY_X_AXIS_DROPDOWN_ID, "value"),
        Output(ids.COLONY_Y_AXIS_DROPDOWN_ID, "value"),
        Input(ids.STORE_FILTER_SPEC, "data"),
        State(ids.COLONY_X_AXIS_DROPDOWN_ID, "value"),
        State(ids.COLONY_Y_AXIS_DROPDOWN_ID, "value"),
    )
    def _populate_axis_dropdowns(
        filter_payload: Any,
        current_x: str | None,
        current_y: str | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str | None, str | None]:
        """Refresh axis dropdown options when the filter spec changes.

        Preserves the current dropdown value when it's still valid;
        otherwise picks a sensible default (first column for X, second
        for Y).
        """
        try:
            spec = FilterSpec.from_store(filter_payload)
            filtered_df = spec.apply_to(df)
        except Exception:
            logger.exception(
                "FilterSpec.apply_to failed while populating colony-view dropdowns."
            )
            filtered_df = df

        columns = selectable_axis_columns(filtered_df, column_value_sets)
        options = [{"label": col, "value": col} for col in columns]

        if current_x in columns:
            x_value: str | None = current_x
        elif columns:
            x_value = columns[0]
        else:
            x_value = None

        if current_y in columns:
            y_value: str | None = current_y
        elif len(columns) >= 2:
            y_value = columns[1]
        elif columns:
            y_value = columns[0]
        else:
            y_value = None

        return options, options, x_value, y_value

    # ----------------------------------------------------------------------
    # 3. Single-cell remove (pattern-matching ALL)
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Input({"type": "colony-cell-remove-btn", "image_file": ALL, "label": ALL}, "n_clicks"),
        State(ids.STORE_REMOVED_KEYS, "data"),
        prevent_initial_call=True,
    )
    def _toggle_single_cell_removal(
        n_clicks_list: list[int | None],
        removed_payload: Any,
    ) -> Any:
        """Toggle a colony's curated-removal state on a per-cell button click.

        Inspects ``callback_context.triggered_id`` to recover the firing
        cell's ``image_file`` / ``label`` and flips its removal state.
        Each :class:`FilteredMeasurements` mutator auto-saves, so we
        only need to re-emit the store payload after the mutation.
        """
        del removed_payload  # filtered_state holds the source of truth.
        triggered = callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        # Skip the initial-mount fire where every n_clicks is 0/None.
        if not any(n for n in n_clicks_list if n):
            return no_update

        raw_image_file = triggered.get("image_file")
        raw_label = triggered.get("label")
        if raw_image_file is None or raw_label is None:
            return no_update
        try:
            image_file = str(raw_image_file)
            label = int(raw_label)
        except (TypeError, ValueError):
            return no_update

        # Mutate + emit under the same lock so a concurrent bulk action
        # can't slip a divergent payload between us and the next render.
        return filtered_state.mutate_and_payload(
            lambda s: s.toggle(image_file, label)
        )

    # ----------------------------------------------------------------------
    # 4. Selection-delta consumer
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Input(ids.STORE_COLONY_SELECTION_DELTA, "data"),
        State(ids.STORE_COLONY_SELECTION, "data"),
        State(ids.STORE_COLONY_GRID_ORDER, "data"),
        prevent_initial_call=True,
    )
    def _consume_selection_delta(
        delta: Any,
        current_selection: Any,
        grid_order_payload: Any,
    ) -> Any:
        """Fold a JS-emitted click into the canonical selection store.

        Delta shape: ``{"key": [image_file, label], "shift": bool, "ts": int}``.
        Single-toggle: add or remove the key, set anchor to the just-clicked
        key when it lands in the selection (else clear the anchor).
        Shift+click: union the current selection with the inclusive slice
        of ``grid_order`` between the existing anchor and the delta key,
        preserving the anchor.
        """
        if not isinstance(delta, dict):
            return no_update
        delta_key = _coerce_anchor(delta.get("key"))
        if delta_key is None:
            return no_update

        current = current_selection if isinstance(current_selection, dict) else {}
        current_anchor = _coerce_anchor(current.get("anchor"))
        original_selected = set(decode_removed_keys_payload(current.get("selected")))
        grid_order = decode_removed_keys_payload(grid_order_payload)

        shift = bool(delta.get("shift"))

        if shift and current_anchor is not None:
            try:
                slice_keys = expand_range(grid_order, current_anchor, delta_key)
            except ValueError:
                # Anchor or target slipped out of the current grid (e.g.
                # the filter or axis changed since the anchor was
                # captured). Fall back to a single-toggle on the delta
                # key, dropping the stale anchor.
                working = set(original_selected)
                if delta_key in working:
                    working.discard(delta_key)
                    new_anchor: tuple[str, int] | None = None
                else:
                    working.add(delta_key)
                    new_anchor = delta_key
                new_selected = working
            else:
                new_selected = original_selected | set(slice_keys)
                new_anchor = current_anchor
        else:
            working = set(original_selected)
            if delta_key in working:
                working.discard(delta_key)
                new_anchor = None
            else:
                working.add(delta_key)
                new_anchor = delta_key
            new_selected = working

        # Suppress same-value re-emissions so a click that doesn't
        # change the selection (e.g. shift-click within an already-fully-
        # covered range) doesn't re-fire the downstream visibility +
        # render callbacks.
        if new_anchor == current_anchor and new_selected == original_selected:
            return no_update
        return _selection_payload(new_anchor, sorted(new_selected))

    # ----------------------------------------------------------------------
    # 5. Bulk-bar visibility / count label
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.COLONY_BULK_BAR_ID, "style"),
        Output(ids.COLONY_BULK_COUNT_LABEL_ID, "children"),
        Input(ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _bulk_bar_visibility(selection_payload: Any) -> tuple[dict[str, str], str]:
        """Show the bulk-action bar iff at least one cell is selected."""
        selected: list[Any] = []
        if isinstance(selection_payload, dict):
            raw = selection_payload.get("selected")
            if isinstance(raw, list):
                selected = raw
        n = len(selected)
        if n == 0:
            return {"display": "none"}, "0 selected"
        return (
            {"display": "flex", "alignItems": "center", "gap": "0.5rem"},
            f"{n} selected",
        )

    # ----------------------------------------------------------------------
    # 6. Bulk Remove / Restore
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Output(ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Input(ids.COLONY_BULK_REMOVE_BTN_ID, "n_clicks"),
        Input(ids.COLONY_BULK_RESTORE_BTN_ID, "n_clicks"),
        State(ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _bulk_remove_or_restore(
        remove_clicks: int | None,
        restore_clicks: int | None,
        selection_payload: Any,
    ) -> tuple[Any, Any]:
        """Apply Remove/Restore to every key in the active selection.

        Reads ``callback_context.triggered_id`` to decide which action to
        take, then clears the selection so the bulk bar collapses.
        """
        del remove_clicks, restore_clicks  # button id resolves the action.
        triggered = callback_context.triggered_id
        if not triggered:
            return no_update, no_update

        selected: list[tuple[str, int]] = []
        if isinstance(selection_payload, dict):
            selected = decode_removed_keys_payload(selection_payload.get("selected"))
        if not selected:
            return no_update, no_update

        if triggered == ids.COLONY_BULK_REMOVE_BTN_ID:
            action = "remove"
        elif triggered == ids.COLONY_BULK_RESTORE_BTN_ID:
            action = "restore"
        else:
            return no_update, no_update

        # Mutate + emit under the same lock so a concurrent click can't
        # slip a divergent payload between the bulk save and the next
        # render.
        def _apply(state: FilteredMeasurements) -> None:
            if action == "remove":
                state.remove_many(selected)
            else:
                state.restore_many(selected)

        payload = filtered_state.mutate_and_payload(_apply)
        return payload, _EMPTY_SELECTION

    # ----------------------------------------------------------------------
    # 7. Bulk Clear
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Input(ids.COLONY_BULK_CLEAR_BTN_ID, "n_clicks"),
        prevent_initial_call=True,
    )
    def _bulk_clear(n_clicks: int | None) -> dict[str, Any]:
        """Reset the active selection without touching the curated set."""
        del n_clicks
        return _EMPTY_SELECTION

    # ----------------------------------------------------------------------
    # 8. Reset selection when the grid layout changes
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Input(ids.COLONY_X_AXIS_DROPDOWN_ID, "value"),
        Input(ids.COLONY_Y_AXIS_DROPDOWN_ID, "value"),
        Input(ids.STORE_FILTER_SPEC, "data"),
        State(ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _reset_selection_on_layout_change(
        x_axis: Any,
        y_axis: Any,
        filter_spec: Any,
        current_selection: Any,
    ) -> Any:
        """Clear the active selection on any change that re-keys the grid.

        Triggers on axis-dropdown swaps and filter-spec changes. Removed
        rows survive (they live on the parquet, not the selection store);
        only the transient multi-select state is reset. Short-circuits
        when the selection is already empty so a filter-row keystroke
        with nothing selected doesn't re-fire the downstream visibility
        and grid-render callbacks.
        """
        del x_axis, y_axis, filter_spec
        if isinstance(current_selection, dict):
            anchor = current_selection.get("anchor")
            selected = current_selection.get("selected") or []
            if anchor is None and not selected:
                return no_update
        return _EMPTY_SELECTION

    # ----------------------------------------------------------------------
    # 9. Clientside selection-styling
    # ----------------------------------------------------------------------
    #
    # ``STORE_COLONY_SELECTION`` changes can fire dozens of times per
    # second (every shift+click), but the only DOM effect is toggling
    # the ``is-selected`` class on cells whose ``data-key`` is in the
    # selection. Doing that in JS instead of through a Python callback
    # avoids re-running ``FilterSpec.apply_to`` and rebuilding the
    # Component tree on every click.
    app.clientside_callback(
        """
        function(selection) {
            const container = document.getElementById("colony-grid-container");
            if (!container) return window.dash_clientside.no_update;
            const selected = (selection && Array.isArray(selection.selected))
                ? selection.selected : [];
            const wanted = new Set();
            selected.forEach(function (entry) {
                if (Array.isArray(entry) && entry.length === 2) {
                    wanted.add(entry[0] + "::" + entry[1]);
                }
            });
            container.querySelectorAll(".colony-cell").forEach(function (cell) {
                const cb = cell.querySelector(".colony-cell-checkbox");
                const key = cb ? cb.dataset.key : null;
                const shouldBeSelected = !!key && wanted.has(key);
                cell.classList.toggle("is-selected", shouldBeSelected);
                if (cb) cb.classList.toggle("is-checked", shouldBeSelected);
            });
            return Date.now();
        }
        """,
        Output(ids.COLONY_SELECTION_EFFECT_ID, "data"),
        Input(ids.STORE_COLONY_SELECTION, "data"),
    )


__all__ = ["register_callbacks"]

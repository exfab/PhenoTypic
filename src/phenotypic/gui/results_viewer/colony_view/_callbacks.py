"""Dash callbacks driving the colony-view tab.

Owns every callback whose Output is confined to the colony-view subtree:

1. **Render** — rebuild the grid component when the filter spec, removal
   set, selection, axis dropdowns, refresh button, or active tab change.
2. **Axis dropdown population** — keep the X/Y axis dropdown options in
   sync with the filtered frame's selectable columns.
3. **Radial category mark / restore** — assign a colony's curation
   category (durable remove) when one of its radial wedges fires, or
   clear it when the center restore node fires.
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
from typing import Any, get_args

import dash
from dash import ALL, MATCH, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from phenotypic.gui._config import (
    COLONY_TILE_SIZE_DEFAULT,
    COLONY_TILE_SIZE_MAX,
    COLONY_TILE_SIZE_MIN,
    TILE_DIM_DEFAULT,
    stepped_colony_tile_size_from_trigger,
    stepped_alpha_from_trigger,
)
from phenotypic.gui._shared._radial import build_radial_body
from phenotypic.gui._shared.tiles import DEFAULT_LAYER, LayerName
from phenotypic.gui._shared._triage_callbacks import (
    apply_wedge_mark,
    bulk_mark,
    category_dropdown_options,
    decode_wedge_trigger,
    fold_selection_delta,
    register_custom_category_safe,
)
from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_state import FilterSpec
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import (
    decode_removed_keys_payload,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    output_mutations_disabled,
    require_output_mutation,
)
from phenotypic.gui.results_viewer.colony_view._grid import (
    build_grid,
    build_stack_popover_rows,
    compute_max_bbox_size,
    selectable_axis_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


# Empty selection payload — used by Clear, layout-change reset, and the
# post-bulk-action emission. Module-level so all three sites agree.
_EMPTY_SELECTION: dict[str, Any] = {"anchor": None, "selected": []}


def _normalize_layer_value(value: Any) -> LayerName:
    """Coerce a raw layer-toggle / store value to a valid :data:`LayerName`.

    The layer rides through Dash as a ``dcc.Store`` payload (a plain JSON
    string) so it can arrive as ``None`` (store not yet seeded, or the toggle
    hidden in a standalone bundle) or — defensively — as any other value. Any
    value outside ``{"rgb", "detect_mat", "objmap"}`` collapses to ``"rgb"`` so
    a stray value renders the finished RGB plate rather than 500-ing the crop
    route. Extracted to a module-level helper so the toggle→store→URL wiring is
    unit-testable without a live Dash app (the callback below is a thin adapter).

    Args:
        value: The raw store/toggle value (``str | None`` in practice).

    Returns:
        One of ``"rgb"`` / ``"detect_mat"`` / ``"objmap"``.
    """
    if value in get_args(LayerName):
        return value  # type: ignore[return-value]  # narrowed by the membership test
    return DEFAULT_LAYER


# These pure helpers now live in the shared triage module so the colony and
# QC surfaces single-source them; re-exported here for back-compat with
# existing imports (e.g. the colony helper unit test).
# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(
    app: dash.Dash,
    output_root: OutputRoot,
    filtered_state: CurationLabels,
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
    mutations_disabled = output_mutations_disabled(output_root)

    def _curation_authorized(action: str) -> bool:
        try:
            require_output_mutation(action)
        except OutputMutationBlocked as exc:
            logger.warning("%s", exc)
            return False
        return True

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
        Input(ids.STORE_COLONY_TILE_SIZE, "data"),
        Input(ids.STORE_TILE_DIM_ALPHA, "data"),
        Input(ids.STORE_ACTIVE_LAYER, "data"),
        Input(ids.TABS_ID, "active_tab"),
    )
    def _render_colony_grid(
        filter_payload: Any,
        removed_payload: Any,
        x_axis: str | None,
        y_axis: str | None,
        refresh_clicks: int | None,
        tile_size: int | None,
        dim_alpha: float | None,
        active_layer: Any,
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
            logger.exception(
                "FilterSpec.apply_to failed in colony-view render."
            )
            filtered_df = df

        max_size = compute_max_bbox_size(filtered_df)
        # Display size: clamp the stepper value into the stepper's own range
        # then cap at the server crop size so the browser never upscales
        # the PNG beyond its native resolution.
        requested_size = (
            int(tile_size) if tile_size else COLONY_TILE_SIZE_DEFAULT
        )
        requested_size = max(
            COLONY_TILE_SIZE_MIN,
            min(COLONY_TILE_SIZE_MAX, requested_size),
        )
        display_size = min(requested_size, max_size)
        removed_keys = set(decode_removed_keys_payload(removed_payload))
        alpha = TILE_DIM_DEFAULT if dim_alpha is None else float(dim_alpha)
        layer = _normalize_layer_value(active_layer)

        # Decision A: read the per-object category map straight off the
        # durable store under its lock (a server-side snapshot — there is
        # NO separate STORE_LABELS Dash store). STORE_REMOVED_KEYS firing is
        # the render trigger; the labels dict it implies is read here so the
        # radial trigger on each tile renders the right category badge.
        with filtered_state._lock:
            category_of = dict(filtered_state.labels)

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
            dim_alpha=alpha,
            category_of=category_of,
            layer=layer,
            mutations_disabled=mutations_disabled,
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
    ) -> tuple[
        list[dict[str, str]], list[dict[str, str]], str | None, str | None
    ]:
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
    # 3. Radial category mark / restore (pattern-matching ALL)
    # ----------------------------------------------------------------------
    #
    # MF4: the radial menu subsumes the old binary ✕ remove button. A wedge
    # click marks the colony with the wedge's category (durable remove); the
    # center node carries ``RADIAL_RESTORE_SENTINEL`` and restores it.

    @app.callback(
        Output(ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Input(
            {
                "type": "colony-cat-wedge",
                "image_file": ALL,
                "label": ALL,
                "category": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _mark_colony_category(_n: list[int | None]) -> Any:
        """Mark (or restore) a colony's category from a radial wedge click.

        Thin adapter over the shared, Dash-free
        :func:`~phenotypic.gui._shared._triage_callbacks.decode_wedge_trigger`
        (ALL-pattern decode + initial-empty-fire guard + custom-folder
        short-circuit) and :func:`apply_wedge_mark` (restore-sentinel → unmark
        else mark). ``None`` from the decode → ``PreventUpdate`` (the colony
        no-op).
        """
        decoded = decode_wedge_trigger(
            callback_context.triggered_id, callback_context.triggered
        )
        if decoded is None:
            raise PreventUpdate
        if not _curation_authorized("Colony curation"):
            raise PreventUpdate
        image_file, label, category = decoded
        payload = apply_wedge_mark(filtered_state, image_file, label, category)
        # ``STORE_REMOVED_KEYS`` is an ``allow_duplicate`` (multi-mode) output
        # whose value is itself a list. Restoring the LAST labeled object
        # yields an empty payload ``[]``; a bare ``[]`` makes Dash's multi-mode
        # response validator see *zero* output values and 500. Wrap in a
        # 1-tuple so Dash sees exactly one value (the list) regardless of its
        # length. (Dash multi-mode artifact — kept at the callback layer.)
        return (payload,)

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

        Thin adapter over the pure, Dash-free
        :func:`~phenotypic.gui._shared._triage_callbacks.fold_selection_delta`
        (shared with the QC-review surface). ``None`` from the helper — a
        malformed delta or a same-value re-emission — maps to ``no_update``
        so a no-op click never re-fires the downstream visibility + render
        callbacks.
        """
        payload = fold_selection_delta(
            delta, current_selection, grid_order_payload
        )
        return no_update if payload is None else payload

    # ----------------------------------------------------------------------
    # 5. Bulk-bar visibility / count label
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.COLONY_BULK_BAR_ID, "style"),
        Output(ids.COLONY_BULK_COUNT_LABEL_ID, "children"),
        Input(ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _bulk_bar_visibility(
        selection_payload: Any,
    ) -> tuple[dict[str, str], str]:
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
            selected = decode_removed_keys_payload(
                selection_payload.get("selected")
            )
        if not selected:
            return no_update, no_update
        if not _curation_authorized("Bulk colony curation"):
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
        def _apply(state: CurationLabels) -> None:
            if action == "remove":
                state.remove_many(selected)
            else:
                state.restore_many(selected)

        payload = filtered_state.mutate_and_payload(_apply)
        return payload, _EMPTY_SELECTION

    # ----------------------------------------------------------------------
    # 6b. Bulk "Mark N selected as ▾" — category dropdown
    # ----------------------------------------------------------------------
    #
    # Populate the dropdown options from the live vocabulary (core + custom),
    # refreshed when a custom category is registered (vocab-revision tick).

    @app.callback(
        Output(ids.COLONY_BULK_MARK_DROPDOWN_ID, "options"),
        Input(ids.STORE_CATEGORY_VOCAB_REVISION, "data"),
    )
    def _populate_bulk_mark_options(
        _revision: int | None,
    ) -> list[dict[str, str]]:
        """Refresh the bulk-mark dropdown options from the category vocabulary."""
        with filtered_state._lock:
            categories = filtered_state.categories()
        return category_dropdown_options(categories)

    @app.callback(
        Output(ids.STORE_REMOVED_KEYS, "data", allow_duplicate=True),
        Output(ids.STORE_COLONY_SELECTION, "data", allow_duplicate=True),
        Output(ids.COLONY_BULK_MARK_DROPDOWN_ID, "value"),
        Input(ids.COLONY_BULK_MARK_DROPDOWN_ID, "value"),
        State(ids.STORE_COLONY_SELECTION, "data"),
        prevent_initial_call=True,
    )
    def _bulk_mark_selected(
        category: str | None,
        selection_payload: Any,
    ) -> tuple[Any, Any, Any]:
        """Mark the active selection with the chosen category, then clear.

        Reset the dropdown ``value`` to ``None`` afterward so re-picking the
        same category on a fresh selection fires again (a dropdown that keeps
        its value would not re-trigger).
        """
        if not category:
            return no_update, no_update, no_update
        selected: list[tuple[str, int]] = []
        if isinstance(selection_payload, dict):
            selected = decode_removed_keys_payload(
                selection_payload.get("selected")
            )
        if not selected:
            return no_update, no_update, None
        if not _curation_authorized("Bulk colony category assignment"):
            return no_update, no_update, None
        try:
            payload = bulk_mark(filtered_state, selected, category)
        except ValueError:
            logger.warning("Bulk-mark rejected unknown category %r", category)
            return no_update, no_update, None
        return payload, _EMPTY_SELECTION, None

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
    # ``STORE_COLONY_SELECTION`` is SHARED by the colony grid and the QC
    # review gallery (M1: selection parity), so the styler sweeps BOTH
    # containers' tiles — a key selected on either surface lights up there.
    # (Within one tab the user selects on a single surface at a time, so a
    # shared selection store + a both-container sweep is correct.)
    app.clientside_callback(
        """
        function(selection) {
            const containers = [
                document.getElementById("colony-grid-container"),
                document.getElementById("qc-review-gallery"),
            ].filter(Boolean);
            if (containers.length === 0) {
                return window.dash_clientside.no_update;
            }
            const selected = (selection && Array.isArray(selection.selected))
                ? selection.selected : [];
            const wanted = new Set();
            selected.forEach(function (entry) {
                if (Array.isArray(entry) && entry.length === 2) {
                    wanted.add(entry[0] + "::" + entry[1]);
                }
            });
            containers.forEach(function (container) {
                container.querySelectorAll(".colony-cell").forEach(function (cell) {
                    const cb = cell.querySelector(".colony-cell-checkbox");
                    const key = cb ? cb.dataset.key : null;
                    const shouldBeSelected = !!key && wanted.has(key);
                    cell.classList.toggle("is-selected", shouldBeSelected);
                    if (cb) cb.classList.toggle("is-checked", shouldBeSelected);
                });
            });
            return Date.now();
        }
        """,
        Output(ids.COLONY_SELECTION_EFFECT_ID, "data"),
        Input(ids.STORE_COLONY_SELECTION, "data"),
    )

    # ----------------------------------------------------------------------
    # 8. Populate stack popover on first badge click
    # ----------------------------------------------------------------------

    @app.callback(
        Output(
            {
                "type": "colony-cell-popover-body",
                "image_file": MATCH,
                "label": MATCH,
            },
            "children",
        ),
        Input(
            {
                "type": "colony-cell-count-badge",
                "image_file": MATCH,
                "label": MATCH,
            },
            "n_clicks",
        ),
        State(
            {
                "type": "colony-cell-popover-data",
                "image_file": MATCH,
                "label": MATCH,
            },
            "data",
        ),
        State(ids.STORE_REMOVED_KEYS, "data"),
        prevent_initial_call=True,
    )
    def _populate_stack_popover(
        n_clicks: int | None,
        data: Any,
        removed_payload: Any,
    ) -> Any:
        """Render the popover body the first time its badge is clicked.

        The popover ships from :func:`build_grid` with an empty body and
        a co-located ``dcc.Store`` carrying the cell's members and per-
        grid sizes. This MATCH callback reads that store on the first
        click and emits the row children, so no ``<img>`` tags exist in
        the DOM until the user actually opens the stack.

        Subsequent clicks re-emit the same children (idempotent given
        identical state) so toggling the popover open/closed never
        re-fetches.
        """
        if not n_clicks or not isinstance(data, dict):
            return no_update
        members_payload = data.get("members") or []
        members: list[tuple[str, str, int]] = []
        for entry in members_payload:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            try:
                members.append((str(entry[0]), str(entry[1]), int(entry[2])))
            except (TypeError, ValueError):
                continue
        if not members:
            return no_update
        try:
            crop_size = int(data.get("crop_size") or 0)
            display_size = int(data.get("display_size") or 0)
        except (TypeError, ValueError):
            return no_update
        if crop_size <= 0 or display_size <= 0:
            return no_update
        try:
            dim_alpha = float(data.get("dim_alpha") or 0.0)
        except (TypeError, ValueError):
            dim_alpha = 0.0
        layer = _normalize_layer_value(data.get("layer"))
        removed_keys = set(decode_removed_keys_payload(removed_payload))
        return build_stack_popover_rows(
            members,
            crop_size=crop_size,
            display_size=display_size,
            removed_keys=removed_keys,
            dim_alpha=dim_alpha,
            layer=layer,
        )

    # ----------------------------------------------------------------------
    # 8b. Lazy-populate the radial popover body on trigger click (4c)
    # ----------------------------------------------------------------------
    #
    # Mirrors the stack-popover populate-on-click pattern: tiles ship with an
    # EMPTY radial popover body (build_radial_trigger) so a grid of many tiles
    # stays light. The first time a tile's ▾ trigger is clicked, this MATCH
    # callback fills the body with the wedge ring via build_radial_body,
    # reading the vocabulary + the colony's current category under the store
    # lock for a consistent snapshot.

    @app.callback(
        Output(
            {
                "type": "colony-radial-popover-body",
                "image_file": MATCH,
                "label": MATCH,
            },
            "children",
        ),
        Input(
            {
                "type": "colony-radial-trigger",
                "image_file": MATCH,
                "label": MATCH,
            },
            "n_clicks",
        ),
        State(
            {
                "type": "colony-radial-store",
                "image_file": MATCH,
                "label": MATCH,
            },
            "data",
        ),
        prevent_initial_call=True,
    )
    def _populate_radial_body(n_clicks: int | None, data: Any) -> Any:
        """Render the radial wedge ring the first time a tile's ▾ is clicked.

        The trigger's co-located ``dcc.Store`` carries ``{image_file, label,
        surface}``. This MATCH callback reads it on click and emits the wedge
        body via :func:`build_radial_body`, snapshotting the live category
        vocabulary and the colony's current category under
        ``filtered_state._lock`` so a concurrent ``mark`` can't tear the read.
        Subsequent clicks re-emit (idempotent given identical state) so
        toggling the popover open/closed never re-fetches.
        """
        if not n_clicks or not isinstance(data, dict):
            return no_update
        raw_image_file = data.get("image_file")
        raw_label = data.get("label")
        surface = str(data.get("surface") or "colony")
        if raw_image_file is None or raw_label is None:
            return no_update
        try:
            image_file = str(raw_image_file)
            label = int(raw_label)
        except (TypeError, ValueError):
            return no_update

        # Snapshot the custom vocabulary + this colony's current category under
        # the lock so the populated ring matches live store state (decision A /
        # concurrency note). The core ring is built from the fixed
        # ErrorCategory tokens inside build_radial_body — no active-category
        # list needed.
        with filtered_state._lock:
            custom_categories = list(filtered_state.custom_categories)
            current_category = filtered_state.labels.get((image_file, label))

        body = build_radial_body(
            surface,
            image_file,
            label,
            custom_categories,
            current_category=current_category,
        )
        # This is a wildcard (MATCH) output, which Dash treats as multi-mode:
        # a single ``Div`` return must be wrapped so the multi-return validator
        # sees exactly one output value (the body) rather than trying to flatten
        # the Div's children list. (The stack-popover callback dodges this only
        # because it happens to return a list of rows.)
        return (body,)

    # ----------------------------------------------------------------------
    # 8c. Add-custom-category from the radial folder (Task 7)
    # ----------------------------------------------------------------------
    #
    # The Custom folder section of the radial body carries a ＋ Add input +
    # confirm. On submit, register the sanitized name (catching collisions /
    # empties as an inline message), then re-render THIS tile's body so the
    # new custom chip appears and bump STORE_CATEGORY_VOCAB_REVISION so every
    # bulk-mark dropdown refreshes its options.
    #
    # NOTE (invariant): this callback and ``_populate_radial_body`` both write
    # the ``colony-radial-popover-body`` MATCH ``children`` output (the latter
    # plain, this one ``allow_duplicate``). They must NEVER be triggerable by
    # the same Input — ``_populate_radial_body`` fires on the trigger
    # ``n_clicks``; this fires on the custom-submit ``n_clicks`` / input
    # ``n_submit`` — or Dash raises a duplicate-output collision.

    @app.callback(
        Output(
            {
                "type": "colony-radial-popover-body",
                "image_file": MATCH,
                "label": MATCH,
            },
            "children",
            allow_duplicate=True,
        ),
        Output(
            {
                "type": "colony-radial-custom-msg",
                "image_file": MATCH,
                "label": MATCH,
            },
            "children",
        ),
        Output(
            ids.STORE_CATEGORY_VOCAB_REVISION, "data", allow_duplicate=True
        ),
        Input(
            {
                "type": "colony-radial-custom-submit",
                "image_file": MATCH,
                "label": MATCH,
            },
            "n_clicks",
        ),
        # Enter in the input submits too (debounce=True fires n_submit).
        Input(
            {
                "type": "colony-radial-custom-input",
                "image_file": MATCH,
                "label": MATCH,
            },
            "n_submit",
        ),
        State(
            {
                "type": "colony-radial-custom-input",
                "image_file": MATCH,
                "label": MATCH,
            },
            "value",
        ),
        State(ids.STORE_CATEGORY_VOCAB_REVISION, "data"),
        prevent_initial_call=True,
    )
    def _add_custom_category(
        n_clicks: int | None,
        n_submit: int | None,
        name: str | None,
        revision: int | None,
    ) -> Any:
        """Register a custom category from a tile's ＋ Add affordance.

        Fires on the ``＋ Add`` button click OR Enter in the input
        (``n_submit``). On success: re-render this tile's radial body (so the
        new chip shows) and bump the vocabulary revision (so bulk-mark
        dropdowns refresh). On failure (empty / collision): leave the body
        untouched and surface the reason in the inline message slot. ``MATCH``
        keys the input value, the message slot, and the body to the same tile
        automatically.
        """
        del n_clicks, n_submit  # either Input fires; gate on triggered below.
        if not callback_context.triggered:
            raise PreventUpdate
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        try:
            image_file = str(triggered["image_file"])
            label = int(triggered["label"])
        except (KeyError, TypeError, ValueError):
            raise PreventUpdate

        if not _curation_authorized("Custom curation category"):
            return (
                no_update,
                "Output is read-only; refresh before editing.",
                no_update,
            )
        token, message = register_custom_category_safe(filtered_state, name)
        if token is None:
            # Validation failure: only the message updates.
            return no_update, message, no_update

        # Re-render this tile's body with the new custom chip; bump the vocab
        # revision so every bulk-mark dropdown re-reads the vocabulary.
        with filtered_state._lock:
            custom_categories = list(filtered_state.custom_categories)
            current_category = filtered_state.labels.get((image_file, label))
        body = build_radial_body(
            "colony",
            image_file,
            label,
            custom_categories,
            current_category=current_category,
        )
        return body, message, int(revision or 0) + 1

    # ----------------------------------------------------------------------
    # 8. Colony rendered tile-size stepper → tile-size store
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_COLONY_TILE_SIZE, "data"),
        Input(ids.COLONY_TILE_SIZE_MINUS, "n_clicks"),
        Input(ids.COLONY_TILE_SIZE_PLUS, "n_clicks"),
        State(ids.STORE_COLONY_TILE_SIZE, "data"),
        prevent_initial_call=True,
    )
    def _step_colony_tile_size(
        _minus_clicks: int | None,
        _plus_clicks: int | None,
        current: int | None,
    ) -> int:
        """Step the rendered colony tile size on a ``−``/``+`` click."""
        return stepped_colony_tile_size_from_trigger(
            dash.ctx.triggered_id,
            current,
            plus_id=ids.COLONY_TILE_SIZE_PLUS,
            minus_id=ids.COLONY_TILE_SIZE_MINUS,
        )

    @app.callback(
        Output(ids.COLONY_TILE_SIZE_READOUT, "children"),
        Input(ids.STORE_COLONY_TILE_SIZE, "data"),
    )
    def _sync_colony_tile_size_readout(tile_size: int | None) -> str:
        """Render ``150 px`` into the tile-size readout from the store."""
        size = (
            COLONY_TILE_SIZE_DEFAULT if tile_size is None else int(tile_size)
        )
        size = max(COLONY_TILE_SIZE_MIN, min(COLONY_TILE_SIZE_MAX, size))
        return f"{size} px"

    # ----------------------------------------------------------------------
    # 9. Colony tile-spotlight ``dim`` stepper → shared store
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_TILE_DIM_ALPHA, "data", allow_duplicate=True),
        Input(ids.COLONY_DIM_MINUS, "n_clicks"),
        Input(ids.COLONY_DIM_PLUS, "n_clicks"),
        State(ids.STORE_TILE_DIM_ALPHA, "data"),
        prevent_initial_call=True,
    )
    def _step_colony_dim(
        _minus_clicks: int | None,
        _plus_clicks: int | None,
        current: float | None,
    ) -> float:
        """Step the shared spotlight strength on a colony ``−``/``+`` click.

        Thin adapter over the pure, Dash-free
        :func:`stepped_alpha_from_trigger` helper (direction from
        ``dash.ctx.triggered_id``; clamp/round inside the helper).
        """
        return stepped_alpha_from_trigger(
            dash.ctx.triggered_id,
            current,
            plus_id=ids.COLONY_DIM_PLUS,
            minus_id=ids.COLONY_DIM_MINUS,
        )

    # ----------------------------------------------------------------------
    # 10. Readout sync — keep the colony toolbar's ``dim`` readout in step
    #    with the shared store. Registered here because the colony surface
    #    always mounts with the viewer.
    #
    #    A second Output drove the QC Review toolbar's readout until the QC
    #    tab was unmounted (spec §3). That was not a harmless leftover:
    #    ``suppress_callback_exceptions`` relaxes SERVER-side validation
    #    only, so the renderer kept throwing on the absent id and discarding
    #    the whole response — freezing the colony readout this callback
    #    exists to drive. When QC returns it registers its own readout sync
    #    rather than being written to from here.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.COLONY_DIM_READOUT, "children"),
        Input(ids.STORE_TILE_DIM_ALPHA, "data"),
    )
    def _sync_dim_readouts(dim_alpha: float | None) -> str:
        """Render ``dim 0.60`` into the colony gallery's readout from the store."""
        alpha = TILE_DIM_DEFAULT if dim_alpha is None else float(dim_alpha)
        return f"dim {alpha:.2f}"

    # ----------------------------------------------------------------------
    # 11. Pixel-layer toggle → active-layer store
    # ----------------------------------------------------------------------
    #
    # Thin adapter over the pure, Dash-free :func:`_normalize_layer_value`
    # helper. The store value is threaded into every crop URL as ``&layer=``
    # by the grid-render callback above. Only mounted/fired in a full run —
    # in a standalone bundle the toggle is hidden and (with
    # ``suppress_callback_exceptions``) this callback simply never fires, so
    # the store keeps its default ``rgb``.

    @app.callback(
        Output(ids.STORE_ACTIVE_LAYER, "data"),
        Input(ids.LAYER_TOGGLE, "value"),
        prevent_initial_call=True,
    )
    def _sync_active_layer(layer_value: Any) -> LayerName:
        """Mirror the segmented layer toggle into the active-layer store."""
        return _normalize_layer_value(layer_value)


__all__ = ["register_callbacks"]

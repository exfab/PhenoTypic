"""Component IDs owned by the QC Review sub-view.

The QC tab is a single tab with a **Configure | Review** segmented
toggle. Configure is the existing per-check card editor; Review is the
master–detail walkthrough built under this ``review/`` subtree. All IDs
the Review layout mounts and the Review callbacks bind live here so the
two modules share one grep-able contract (mirrors
:mod:`phenotypic.gui.results_viewer._qc_tab._ids`).

Group-key values reach the client as the JSON-string encoding from
:func:`._review_state.encode_group_key`, so pattern-matching worklist-row
ids carry that encoded string in their ``key`` field — round-trippable
and safe as a Dash id component.
"""

from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------------------------
# Configure | Review toggle (mounted in the QC tab body)
# ---------------------------------------------------------------------------

#: Segmented control switching the QC tab between its Configure and Review
#: sub-views. A ``dbc.RadioItems`` (button-group style) whose value is one
#: of :data:`QC_SUBVIEW_CONFIGURE` / :data:`QC_SUBVIEW_REVIEW`.
QC_SUBVIEW_TOGGLE_ID: str = "qc-subview-toggle"

#: ``dcc.Store`` mirroring the active sub-view so callbacks that need the
#: mode without depending on the toggle widget can read it.
STORE_QC_SUBVIEW: str = "store-qc-subview"

#: Wrapper ``<div>`` for the Configure sub-view (the existing card editor).
#: Visibility toggled via ``style.display`` by the sub-view switch callback.
QC_CONFIGURE_VIEW_ID: str = "qc-configure-view"

#: Wrapper ``<div>`` for the Review sub-view (this subtree's layout).
QC_REVIEW_VIEW_ID: str = "qc-review-view"

#: Sub-view value constants (the toggle's option values).
QC_SUBVIEW_CONFIGURE: str = "configure"
QC_SUBVIEW_REVIEW: str = "review"


# ---------------------------------------------------------------------------
# Review toolbar
# ---------------------------------------------------------------------------

#: Module picker dropdown — options are the enabled QC entries (one per
#: ``instance_id``); value is the selected ``instance_id``.
QC_REVIEW_MODULE_PICKER_ID: str = "qc-review-module-picker"

#: ``on`` / ``groupby`` chip row beside the module picker (read-only).
QC_REVIEW_MODULE_CHIPS_ID: str = "qc-review-module-chips"

#: "Show: unreviewed / all / fail+warn" worklist filter (``dbc.RadioItems``).
QC_REVIEW_SHOW_FILTER_ID: str = "qc-review-show-filter"

#: ↻ Re-sort queue button — re-applies worst-first order to the worklist
#: (the only thing that reorders it; recompute updates rows in place).
QC_REVIEW_RESORT_BTN_ID: str = "qc-review-resort-btn"

#: Show-filter option values.
QC_SHOW_UNREVIEWED: str = "unreviewed"
QC_SHOW_ALL: str = "all"
QC_SHOW_FAIL_WARN: str = "fail_warn"

#: ``−`` button of the Review tile-spotlight ``dim`` stepper. Each click
#: steps the shared ``STORE_TILE_DIM_ALPHA`` strength (owned by the
#: results-viewer layer) down by
#: :data:`phenotypic.gui._config.TILE_DIM_STEP`.
QC_REVIEW_DIM_MINUS: str = "qc-review-dim-minus"

#: ``+`` button of the Review tile-spotlight ``dim`` stepper. Steps the
#: shared strength up by ``TILE_DIM_STEP``.
QC_REVIEW_DIM_PLUS: str = "qc-review-dim-plus"

#: Read-only ``dim 0.60`` readout between the Review stepper's buttons.
#: Synced from ``STORE_TILE_DIM_ALPHA`` by the shared readout callback.
QC_REVIEW_DIM_READOUT: str = "qc-review-dim-readout"


# ---------------------------------------------------------------------------
# Summary header (stat tiles)
# ---------------------------------------------------------------------------

#: Container ``<div>`` holding the per-module stat tiles (total / fail /
#: warn / pass / insufficient / reviewed / colonies removed / median
#: metric). Rebuilt by the summary-render callback on module switch +
#: after each recompute.
QC_REVIEW_SUMMARY_HEADER_ID: str = "qc-review-summary-header"


# ---------------------------------------------------------------------------
# Worklist sidebar
# ---------------------------------------------------------------------------

#: Outer sidebar wrapper (chevron toggle + worklist). Its width/display is
#: flipped by the collapse callback; the detail/gallery pane (``flex: 1 1
#: auto``) reclaims the freed width automatically.
QC_REVIEW_SIDEBAR_ID: str = "qc-review-sidebar"

#: Left worklist container — rebuilt (frozen order) on module switch and
#: on ↻ Re-sort; individual rows update in place after recompute. Its
#: width is driven by :data:`STORE_QC_SIDEBAR_WIDTH` (set by the JS
#: drag-splitter), clamped to [140, 380] px.
QC_REVIEW_WORKLIST_ID: str = "qc-review-worklist"

#: Thin draggable splitter handle between the worklist sidebar and the
#: detail/gallery pane. A clientside drag (see ``results_viewer.js``)
#: updates the sidebar width live and persists the final px to
#: :data:`STORE_QC_SIDEBAR_WIDTH` on mouse-up.
QC_REVIEW_SPLITTER_ID: str = "qc-review-splitter"

#: Chevron button collapsing/expanding the sidebar to/from a thin rail.
#: Glyph flips ◀ (collapse) / ▶ (expand) with state.
QC_REVIEW_SIDEBAR_TOGGLE_ID: str = "qc-review-sidebar-toggle"

#: ``dcc.Store`` (memory) holding the sidebar collapsed flag (bool). A
#: callback toggles the sidebar wrapper style + chevron glyph from it.
STORE_QC_SIDEBAR_COLLAPSED: str = "store-qc-sidebar-collapsed"

#: ``dcc.Store`` (memory) holding the user's dragged sidebar width in px.
#: Written by the JS drag-splitter on mouse-up; a Dash callback applies it
#: to the worklist's ``style.width`` so the width survives re-renders and
#: collapse/expand. Defaults to :data:`SIDEBAR_DEFAULT_WIDTH_PX`.
STORE_QC_SIDEBAR_WIDTH: str = "store-qc-sidebar-width"

#: ``dcc.Store`` holding the frozen worklist order for the active module
#: as a list of encoded group keys (so recompute can update a row in place
#: without reordering, and "next" can advance through the frozen order).
STORE_QC_WORKLIST_ORDER: str = "store-qc-worklist-order"

#: ``dcc.Store`` holding the encoded key of the currently-open group.
STORE_QC_SELECTED_GROUP: str = "store-qc-selected-group"

#: ``dcc.Store`` mapping encoded group key -> {"before": float|None,
#: "after": float|None, "moved": bool} accumulated across in-session
#: recomputes, so the worklist + detail header can render the
#: before→after delta and a "moved/changed" hint.
STORE_QC_RECOMPUTE_DELTAS: str = "store-qc-recompute-deltas"


def worklist_row_id(instance_id: str, key: str) -> Dict[str, str]:
    """Pattern-matching id for a worklist row (one QC group).

    Args:
        instance_id: The owning module's ``instance_id``.
        key: The group key encoded via
            :func:`._review_state.encode_group_key`.

    Returns:
        ``{"type": "qc-worklist-row", "instance": instance_id, "key": key}``.
    """
    return {"type": "qc-worklist-row", "instance": instance_id, "key": key}


def worklist_row_metric_id(instance_id: str, key: str) -> Dict[str, str]:
    """Pattern-matching id for a worklist row's metric/badge cell.

    Updated in place after recompute (no row reorder).
    """
    return {
        "type": "qc-worklist-row-metric",
        "instance": instance_id,
        "key": key,
    }


# ---------------------------------------------------------------------------
# Detail pane
# ---------------------------------------------------------------------------

#: Container for the detail pane (group header + faceted tile gallery +
#: action bar). Rebuilt whenever the selected group changes.
QC_REVIEW_DETAIL_ID: str = "qc-review-detail"

#: Group-header sub-region (key, metric before→after delta, status, n,
#: removed). Rebuilt on selection + after recompute.
QC_REVIEW_DETAIL_HEADER_ID: str = "qc-review-detail-header"

#: Faceted tile-gallery sub-region (one row per timepoint for time-course
#: checks, else a single flat gallery via
#: :func:`gui._shared.tiles.build_tile_grid`). The JS shift-click bridge
#: attaches to this container so QC tile checkbox clicks emit a selection
#: delta exactly like the colony grid (selection parity, M1).
QC_REVIEW_GALLERY_ID: str = "qc-review-gallery"

#: ``dcc.Store`` written by the QC gallery render carrying the row-major
#: order of the currently-rendered QC tiles as a list of ``[image_file,
#: label]`` pairs. The QC selection-delta consumer resolves shift-range
#: selections against this order (the QC analogue of
#: :data:`...._ids.STORE_COLONY_GRID_ORDER`), since the QC gallery's order
#: differs from the colony grid's.
STORE_QC_GALLERY_ORDER: str = "store-qc-gallery-order"

#: ``dcc.Store`` written by the JS shift-click bridge carrying the most
#: recent QC-tile selection delta (``{"key": [image_file, label], "shift":
#: bool, "ts": int}``). A QC-specific consumer folds it into the SHARED
#: :data:`...._ids.STORE_COLONY_SELECTION` (within one tab the user selects
#: on a single surface at a time), resolving ranges against
#: :data:`STORE_QC_GALLERY_ORDER`. The QC analogue of
#: :data:`...._ids.STORE_COLONY_SELECTION_DELTA`.
STORE_QC_GALLERY_SELECTION_DELTA: str = "store-qc-gallery-selection-delta"

#: "Mark reviewed" button in the detail action bar — marks the open group
#: reviewed and (if changes were made) triggers the per-group recompute.
QC_REVIEW_MARK_REVIEWED_BTN_ID: str = "qc-review-mark-reviewed-btn"

#: Icon-only previous-group button — moves backward through the frozen
#: visible worklist order without marking reviewed or recomputing.
QC_REVIEW_PREV_BTN_ID: str = "qc-review-prev-btn"

#: Icon-only next-group button — advances to the next unreviewed group in the
#: frozen order, auto-marking the current one reviewed if it was curated.
QC_REVIEW_NEXT_BTN_ID: str = "qc-review-next-btn"

#: Bulk-remove button for the multi-selected tiles in the open group.
QC_REVIEW_BULK_REMOVE_BTN_ID: str = "qc-review-bulk-remove-btn"

#: Bulk-restore button for the multi-selected tiles in the open group.
QC_REVIEW_BULK_RESTORE_BTN_ID: str = "qc-review-bulk-restore-btn"

#: Category dropdown in the Review detail action bar — "Mark selected as ▾".
#: Options are ``filtered_state.categories()`` (core + custom); selecting one
#: marks the active selection via ``mark_many(selected, category)``. The
#: explicit Remove(=other)/Restore buttons stay.
QC_REVIEW_BULK_MARK_DROPDOWN_ID: str = "qc-review-bulk-mark-dropdown"

#: Empty-state placeholder shown when no QC artifact exists / no module
#: selected.
QC_REVIEW_EMPTY_STATE_ID: str = "qc-review-empty-state"


__all__ = [
    # Toggle
    "QC_SUBVIEW_TOGGLE_ID",
    "STORE_QC_SUBVIEW",
    "QC_CONFIGURE_VIEW_ID",
    "QC_REVIEW_VIEW_ID",
    "QC_SUBVIEW_CONFIGURE",
    "QC_SUBVIEW_REVIEW",
    # Toolbar
    "QC_REVIEW_MODULE_PICKER_ID",
    "QC_REVIEW_MODULE_CHIPS_ID",
    "QC_REVIEW_SHOW_FILTER_ID",
    "QC_REVIEW_RESORT_BTN_ID",
    "QC_SHOW_UNREVIEWED",
    "QC_SHOW_ALL",
    "QC_SHOW_FAIL_WARN",
    "QC_REVIEW_DIM_MINUS",
    "QC_REVIEW_DIM_PLUS",
    "QC_REVIEW_DIM_READOUT",
    # Summary header
    "QC_REVIEW_SUMMARY_HEADER_ID",
    # Worklist
    "QC_REVIEW_SIDEBAR_ID",
    "QC_REVIEW_WORKLIST_ID",
    "QC_REVIEW_SPLITTER_ID",
    "QC_REVIEW_SIDEBAR_TOGGLE_ID",
    "STORE_QC_SIDEBAR_COLLAPSED",
    "STORE_QC_SIDEBAR_WIDTH",
    "STORE_QC_WORKLIST_ORDER",
    "STORE_QC_SELECTED_GROUP",
    "STORE_QC_RECOMPUTE_DELTAS",
    "worklist_row_id",
    "worklist_row_metric_id",
    # Detail
    "QC_REVIEW_DETAIL_ID",
    "QC_REVIEW_DETAIL_HEADER_ID",
    "QC_REVIEW_GALLERY_ID",
    "STORE_QC_GALLERY_ORDER",
    "STORE_QC_GALLERY_SELECTION_DELTA",
    "QC_REVIEW_MARK_REVIEWED_BTN_ID",
    "QC_REVIEW_PREV_BTN_ID",
    "QC_REVIEW_NEXT_BTN_ID",
    "QC_REVIEW_BULK_REMOVE_BTN_ID",
    "QC_REVIEW_BULK_RESTORE_BTN_ID",
    "QC_REVIEW_BULK_MARK_DROPDOWN_ID",
    "QC_REVIEW_EMPTY_STATE_ID",
]

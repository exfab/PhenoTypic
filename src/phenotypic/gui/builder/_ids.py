"""Single source of truth for component IDs in the Dash pipeline builder.

This module exposes plain string constants for static (non-pattern-matching)
component ids, plus small helpers for the pattern-matching ids used by the
palette and breadcrumb. Both layout (Phase 2) and callbacks (Phase 3) import
from here so the contract between them is stable and grep-able.

Notes:
    - Pattern-matching ids returned by helpers are plain dicts; Dash hashes
      them at registration time, so Phase 3 can use ``MATCH`` / ``ALL`` against
      the ``type`` key documented on each helper.
    - Constants intentionally use kebab-case strings to match Dash convention
      and the existing ``DIR_PICKER_*`` ids in ``_directory_browser.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

#: Closed set of page tokens for the Load Picker modal.
LoadPickerPage = Literal["chooser", "json", "prefab"]

#: Closed set of pipeline stage names.
StageName = Literal["ops", "meas", "post"]


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

#: Holds the JSON dump of :class:`BuilderState` (see ``state_to_json``).
STORE_BUILDER_STATE = "store-builder-state"

#: Per-tab uuid (storage_type='session') used as the IntermediatesCache key.
STORE_SESSION_ID = "store-session-id"

#: List of ``node_id`` values that have a cached intermediate this session.
STORE_INTERMEDIATE_KEYS = "store-intermediate-keys"

# ---------------------------------------------------------------------------
# DAG-redesign ids (spec §6)
# ---------------------------------------------------------------------------
#
# The popover-era stores (``PORT_CLICK_STORE`` / ``POPOVER_*``) were
# retired in Phase 7. The DAG path is the only renderer since Phase 8
# (the ``PHENOTYPIC_GUI_DAG`` feature flag and the legacy linear-list
# canvas were retired together; only the legacy migration tests still
# reach those code paths).

#: ``dcc.Store`` written by the clientside ``viewport_ops.js`` glue when the
#: user (or a server-side callback) requests a viewport-level operation such
#: as ``scroll_to``, ``relayout``, ``reanchor``, ``drill_to_scope``, or
#: ``block_collapsed_toggle``.  Server-side callbacks subscribe to it to
#: route through ``_dispatch_state_update`` (or, for purely visual ops, to
#: forward the payload back to the clientside).  See spec §5.5 "Clientside
#: event contract" for the exact payload schema.
STORE_VIEWPORT_OP = "store-viewport-op"

#: ``dcc.Store`` holding the most recent ``List[Issue]`` produced by
#: :func:`phenotypic.gui.builder._validation.validate`.  Drives the toolbar
#: issue badge count + tooltip rows + per-block red/yellow border decoration.
STORE_ISSUES = "store-issues"

#: ``dbc.Badge`` rendered in the canvas toolbar showing the live issue
#: count (e.g. ``"3 issues, 1 hint"``).  Click acts as a tooltip target —
#: hovering pops :data:`ISSUE_BADGE_TOOLTIP` listing one row per issue.
#: ``update_issue_badge`` wires the label + tooltip rows from :data:`STORE_ISSUES`.
ISSUE_BADGE = "issue-badge"

#: ``dbc.Popover`` (anchored at :data:`ISSUE_BADGE`) listing one
#: :func:`issue_row_id` row per :class:`Issue` in :data:`STORE_ISSUES`.
#: Issues sort first (by ``kind`` alphabetically), then hints.  Each
#: row click writes a ``scroll_to`` payload to :data:`STORE_VIEWPORT_OP`
#: so 6B's ``phenotypicScrollTo`` chain consumes it.
ISSUE_BADGE_TOOLTIP = "issue-badge-tooltip"

#: ``dcc.Store`` written by the asset-readiness polling loop in
#: ``assets/builder.js``.  Data shape:
#: ``{"wire_drawing": bool, "palette_dnd": bool, "viewport_ops": bool,
#: "dagre_missing": bool}`` — ``True`` means the asset's IIFE registered
#: the ``window.phenotypic_<name>_ready`` sentinel within the polling
#: window (``False`` means missing / failed to load).  Consumed by
#: :func:`phenotypic.gui.builder._layout.build_asset_status_banner` and by
#: the ``asset_status_disables`` callback to gate the ``Re-layout`` button
#: and the palette ``pointer-events`` style.
STORE_ASSET_STATUS = "store-asset-status"

#: ``dcc.Store`` written by the clientside ``palette_dnd.js`` glue when
#: a user drops a palette button onto the canvas (or fires the
#: keyboard fallback). Server-side callbacks subscribe to it to route
#: through ``_dispatch_state_update`` with the ``block_create`` kind.
#: Data shape (per spec §5.5):
#: ``{"kind": "block_create", "class_name": str, "x": float, "y": float,
#: "container_block_id": str | None, "ts": int}`` — ``None`` between
#: drops.  ``ts`` is a monotonic timestamp so repeat drops of the same
#: class still trigger change detection.
STORE_PALETTE_DROP = "store-palette-drop"

#: ``dcc.Store`` written by the clientside ``wire_drawing.js`` glue when
#: the user completes (or cancels) a wire-drag gesture, **or** when an
#: inspector wire-card / aux-card action triggers an edge mutation.
#: Server-side callbacks subscribe to it to route through
#: ``_dispatch_state_update`` with the appropriate ``edge_*`` /
#: ``list_aux_*`` / ``wire_select`` / ``block_select`` kind.  See spec
#: §5.5 (clientside event contract) and §5.6 (dispatch table).  Data
#: shape varies by ``kind``:
#:
#: * ``{"kind": "edge_create", "source_block_id": str,
#:   "target_block_id": str, "target_port": str, "edge_kind":
#:   "image" | "aux", "ts": int}`` — note ``edge_kind`` is the wire
#:   kind (image/aux); ``kind`` is reserved for the dispatch
#:   discriminator at the top level.
#: * ``{"kind": "edge_delete", "edge_id": str, "ts": int}``.
#: * ``{"kind": "list_aux_reorder", "block_id": str, "param": str,
#:   "new_order": List[str | None], "ts": int}``.
#: * ``{"kind": "list_aux_add_empty_slot", "block_id": str,
#:   "param": str, "ts": int}``.
#: * ``{"kind": "wire_select", "edge_id": str | None, "ts": int}``.
#: * ``{"kind": "block_select", "block_id": str | None, "ts": int}``.
#:
#: ``None`` between events.  The fan-in callback routes on
#: ``payload["kind"]`` so the single store carries every wire-related
#: mutation; this keeps the JS surface tiny and lets the inspector
#: emit dispatches through the same channel rather than needing a
#: parallel store.
STORE_EDGE_EVENT = "store-edge-event"

#: Toolbar button that re-runs the dagre layout pass + ``cy.fit()``.  Wired
#: to the ``relayout`` payload in ``STORE_VIEWPORT_OP``; disabled by the
#: ``asset_status_disables`` callback when ``viewport_ops.js`` or the
#: ``cytoscape-dagre`` extension is missing.
BTN_RELAYOUT = "btn-relayout"

#: Button inside the Input Image inspector card (spec §4.5) that
#: dispatches a ``reanchor`` payload to :data:`STORE_VIEWPORT_OP` so the
#: clientside viewport-ops chain pans / zooms the cytoscape view to
#: centre on the auto-seeded Input Image block.  Distinct from
#: :data:`BTN_RELAYOUT` (which re-runs dagre on the full graph) so the
#: dispatcher / clientside can subscribe to each independently.
BTN_REANCHOR = "btn-reanchor"

#: Container wrapping the inspector "empty-state" placeholder card
#: shown when neither a block nor a wire is selected (spec §4.5).
#: Carries the "Drag an operation from the palette to begin." prompt
#: plus a one-line hint pointing at the toolbar validation badge.
#: Used as a stable handle for integration tests that assert the
#: empty-state branch is rendering.
INSPECTOR_EMPTY_STATE = "inspector-empty-state"

#: Container wrapping the inspector "Input Image — pipeline source"
#: card (spec §4.5).  Rendered when ``selected_block_id`` resolves to
#: a block with ``class_name == INPUT_IMAGE_CLASS_NAME``.  The card
#: carries the Re-layout + Re-anchor buttons (no param form, no
#: delete button) and is a stable handle for integration tests.
INSPECTOR_INPUT_IMAGE_CARD = "inspector-input-image-card"

#: ``html.Div`` row sitting above the canvas that surfaces missing-asset
#: messages (one row per missing JS file).  Subscribes to
#: :data:`STORE_ASSET_STATUS`; hidden when all assets ready.  Rendered by
#: :func:`phenotypic.gui.builder._layout.build_asset_status_banner`.
BANNER_ASSET_STATUS = "banner-asset-status"

#: ``dbc.Modal`` housing the confirm-delete prompt shown when the user
#: requests deletion of a non-empty :class:`~phenotypic.ImagePipeline`
#: container block.  Mounted once at app boot; ``is_open`` driven by
#: ``STORE_BUILDER_STATE.pending_delete_block_id``.  See spec §6
#: "Confirm-delete modal" row.
CONFIRM_DELETE_MODAL_ID = "confirm-delete-modal"

#: Primary action button inside :data:`CONFIRM_DELETE_MODAL_ID`.  Dispatches
#: ``block_delete_confirm``.
BTN_CONFIRM_DELETE = "btn-confirm-delete"

#: Cancel button inside :data:`CONFIRM_DELETE_MODAL_ID`.  Clears
#: ``state.pending_delete_block_id`` so the modal closes.
BTN_CANCEL_DELETE = "btn-cancel-delete"

#: Visible "Drill in →" button on the inspector container card (spec §4.5).
#: Rendered by :func:`phenotypic.gui.builder._layout._build_dag_inspector`
#: when the selected block is a :data:`PIPELINE_CLASS_NAME` container.
#: Distinct from :data:`BTN_DRILL_IN` (which dispatches the legacy
#: ``drill_in`` for nested-pipeline ``StepNode`` selections) so the
#: container dispatcher can subscribe without spurious triggers
#: from the legacy path.  Dispatches ``drill_into_container``.
BTN_DRILL_IN_CONTAINER = "btn-drill-in-container"

#: Container-name text input inside the inspector container card.  Bound
#: to ``BuilderScope.name`` of the selected container's nested scope
#: (spec §4.5).  The dispatcher reads this to update
#: ``block.nested.name`` on debounce.
INPUT_CONTAINER_NAME = "input-container-name"

#: Container-description text input inside the inspector container card.
#: Bound to ``BuilderScope.desc`` of the selected container's nested
#: scope (spec §4.5).
INPUT_CONTAINER_DESC = "input-container-desc"

# NOTE: ``STORE_EDGE_EVENT`` is declared higher up in this module.
# The inspector pane writes into the same store via the kinds
# ``edge_delete``, ``list_aux_add_empty_slot``, and ``list_aux_reorder``
# so the clientside ``wire_drawing.js`` glue and the server-side
# inspector callbacks share a single mutation channel.

#: ``html.Div`` wrapping the inspector wire-card. Mounted by
#: :func:`phenotypic.gui.builder._layout.build_inspector` (DAG branch)
#: only when ``state.selected_edge_id`` resolves to an :class:`Edge`
#: in the active scope.  Carries the wire's source/target labels,
#: kind tag, and a ``Disconnect`` button.
INSPECTOR_WIRE_CARD = "inspector-wire-card"

#: ``html.Div`` wrapping the inspector aux-ports section.  Mounted by
#: :func:`phenotypic.gui.builder._layout.build_inspector` (DAG branch)
#: only when ``state.selected_block_id`` resolves to a :class:`BlockNode`
#: that exposes one or more op-typed parameters.  Enumerates each aux
#: port (scalar or list-typed) with the wired edges + the ``+ Add
#: empty slot`` affordance.
INSPECTOR_AUX_SECTION = "inspector-aux-section"

#: Pattern-match ``type`` key for the wire-card / list-row ``Disconnect``
#: buttons.  Both surfaces dispatch ``edge_delete`` against a specific
#: :class:`Edge.edge_id`; the inspector callback matches
#: ``Input({"type": BTN_INSPECTOR_DISCONNECT, "edge_id": ALL}, "n_clicks")``.
BTN_INSPECTOR_DISCONNECT = "btn-inspector-disconnect"

#: Pattern-match ``type`` key for the list-aux per-row remove buttons
#: (``✕``).  Distinguished from :data:`BTN_INSPECTOR_DISCONNECT` so
#: both surfaces can co-exist in the same DOM without callback id
#: collisions.
BTN_INSPECTOR_LIST_REMOVE = "btn-inspector-list-remove"

#: Pattern-match ``type`` key for the ``+ Add empty slot`` buttons inside
#: the aux ports section.  Keyed by ``(block_id, param)`` so the
#: dispatcher knows which list-aux port to extend.
BTN_INSPECTOR_ADD_EMPTY_SLOT = "btn-inspector-add-empty-slot"

#: Pattern-match ``type`` key for the per-row ``▲``/``▼`` move
#: buttons (drag-handle fallback).  Each id carries ``edge_id`` +
#: ``direction`` so a single callback can dispatch the right reorder.
BTN_INSPECTOR_LIST_MOVE = "btn-inspector-list-move"

#: Pattern-match ``type`` key for the hidden ``dcc.Store`` rendered once
#: per list-typed op-param on the selected block.  Future HTML5 drag
#: glue can write the new permutation as a ``List[str]`` of edge_ids
#: here without churn to the inspector callback; the arrow-button
#: fallback uses :data:`BTN_INSPECTOR_LIST_MOVE` instead.
STORE_INSPECTOR_LIST_REORDER = "store-inspector-list-reorder"


def issue_row_id(block_id: Optional[str], kind: str, idx: int) -> Dict[str, Any]:
    """Build the pattern-matching id for a single issue-tooltip row.

    Each row inside :data:`ISSUE_BADGE_TOOLTIP` carries a structured id
    so the click-dispatch callback (spec §4.6, §5.6) can recover the
    issue identity without parsing the DOM.  The row id pattern is
    matched in :mod:`_callbacks` to write a ``scroll_to`` payload to
    :data:`STORE_VIEWPORT_OP`.

    Args:
        block_id: ``BlockNode.block_id`` of the offender, or ``None``
            for scope-level findings (e.g. ``missing_input``).  ``None``
            is mangled to the literal string ``"__scope__"`` because
            Dash pattern-matched ids must be JSON-serialisable and Dash
            rejects ``None`` as a key value in some store-write paths.
        kind: The :class:`Issue.kind` string (e.g. ``"fork"``,
            ``"stub"``, ``"missing_input"``).  Matches one entry in
            :data:`~phenotypic.gui.builder._validation.IssueKind`.
        idx: Position in the rendered tooltip list (0-based).  Required
            because multiple issues can share ``(block_id, kind)`` —
            e.g. two ``fork`` issues on the same source block (one for
            image-out, one for image-in).  Each row needs a unique id
            so the pattern-match callback can disambiguate.

    Returns:
        Dict of shape ``{"type": "issue-row", "block_id": str,
        "kind": str, "idx": int}``.  :func:`issue_row_click_dispatch`
        matches ``Input({"type": "issue-row", "block_id": ALL,
        "kind": ALL, "idx": ALL}, "n_clicks")``.
    """

    return {
        "type": "issue-row",
        "block_id": block_id if block_id is not None else "__scope__",
        "kind": kind,
        "idx": idx,
    }


def inspector_disconnect_id(edge_id: str) -> Dict[str, Any]:
    """Build the pattern-matching id for the inspector ``Disconnect`` button.

    The wire-card and the per-row ``✕`` remove button inside the
    aux ports section both dispatch ``edge_delete`` for a specific
    :class:`Edge`.  Both call into this helper so callbacks can match
    against ``Input({"type": BTN_INSPECTOR_DISCONNECT, "edge_id": ALL},
    "n_clicks")`` regardless of which surface emitted the click.

    Args:
        edge_id: The :class:`Edge.edge_id` value of the wire to delete.

    Returns:
        Dict of shape ``{"type": BTN_INSPECTOR_DISCONNECT, "edge_id":
        edge_id}``.
    """

    return {"type": BTN_INSPECTOR_DISCONNECT, "edge_id": edge_id}


def inspector_list_remove_id(edge_id: str) -> Dict[str, Any]:
    """Build the pattern-matching id for the list-aux row ``✕`` button.

    Rendered once per edge inside the aux ports section's list-aux row
    enumeration.  Distinguished from :func:`inspector_disconnect_id` by
    the ``type`` key so the inspector can render *both* in the same
    DOM (wire-card Disconnect + per-row remove) without callback
    pattern-match collisions.

    Args:
        edge_id: The :class:`Edge.edge_id` value of the list-aux row.

    Returns:
        Dict of shape ``{"type": BTN_INSPECTOR_LIST_REMOVE, "edge_id":
        edge_id}``.
    """

    return {"type": BTN_INSPECTOR_LIST_REMOVE, "edge_id": edge_id}


def inspector_add_empty_slot_id(block_id: str, param: str) -> Dict[str, Any]:
    """Build the pattern-matching id for the ``+ Add empty slot`` button.

    Rendered once per list-typed op-param on the selected block.
    Carries enough context (block_id + param name) for the dispatcher
    to know which port to extend without consulting
    ``selected_block_id`` at click-time.

    Args:
        block_id: ``BlockNode.block_id`` the slot belongs to.
        param: Op-typed parameter name on that block.

    Returns:
        Dict of shape ``{"type": BTN_INSPECTOR_ADD_EMPTY_SLOT,
        "block_id": block_id, "param": param}``.
    """

    return {
        "type": BTN_INSPECTOR_ADD_EMPTY_SLOT,
        "block_id": block_id,
        "param": param,
    }


def inspector_list_move_id(
    edge_id: str, direction: str
) -> Dict[str, Any]:
    """Build the pattern-matching id for a list-aux row up/down arrow.

    The ordered-list section ships with arrow-button reorder as a
    fallback for the drag-handles called out in spec §4.5.  Each row
    carries an ``▲`` (up) and ``▼`` (down) button keyed by
    ``edge_id`` + direction so a single pattern-match callback can
    dispatch the right reorder.

    Args:
        edge_id: The :class:`Edge.edge_id` value of the wire to move.
        direction: ``"up"`` or ``"down"``.

    Returns:
        Dict of shape ``{"type": BTN_INSPECTOR_LIST_MOVE, "edge_id":
        edge_id, "direction": direction}``.
    """

    return {
        "type": BTN_INSPECTOR_LIST_MOVE,
        "edge_id": edge_id,
        "direction": direction,
    }


def inspector_list_reorder_store_id(
    block_id: str, param: str
) -> Dict[str, Any]:
    """Build the pattern-matching id for the list-aux reorder ``dcc.Store``.

    A hidden ``dcc.Store`` rendered once per list-typed op-param on the
    selected block.  Future HTML5 drag glue (or the existing arrow-
    button fallback) can write the new permutation here; the server-
    side callback dispatches ``list_aux_reorder`` against the same
    ``(block_id, param)`` without re-walking the selected-block layout.

    Args:
        block_id: ``BlockNode.block_id`` the slot list belongs to.
        param: Op-typed parameter name on that block.

    Returns:
        Dict of shape ``{"type": STORE_INSPECTOR_LIST_REORDER,
        "block_id": block_id, "param": param}``.
    """

    return {
        "type": STORE_INSPECTOR_LIST_REORDER,
        "block_id": block_id,
        "param": param,
    }


def block_port_id(block_id: str, port: str) -> str:
    """Return the cytoscape node id for a DAG block's port sub-node.

    The DAG redesign renders every port (image-in, image-out, aux) as a
    cytoscape compound child of its parent block.  Each port carries a
    deterministic id derived from the parent ``BlockNode.block_id`` and
    the port name so callbacks reading ``tapNodeData`` can recover the
    structured pair via ``id.split("__")``.

    Args:
        block_id: 32-character ``BlockNode.block_id`` of the parent block.
        port: Logical port name — ``"in"`` for image-input, ``"out"`` for
            image-output, the parameter name for aux ports (with a
            ``"[<i>]"`` suffix for list-aux slots).

    Returns:
        Flat string ``"port__<block_id>__<port>"`` suitable as a
        cytoscape element id.
    """

    return f"port__{block_id}__{port}"


def edge_id(eid: str) -> str:
    """Return the cytoscape edge id for a DAG :class:`Edge`.

    Cytoscape requires every edge to carry a stable string id; the DAG
    schema generates random 32-character UUID hex strings for
    ``Edge.edge_id`` and the canvas wraps them in a short prefix so the
    cytoscape id namespace doesn't collide with block / port ids.

    Args:
        eid: ``Edge.edge_id`` value.

    Returns:
        Flat string ``"edge__<eid>"``.
    """

    return f"edge__{eid}"


# ---------------------------------------------------------------------------
# Top-level layout regions
# ---------------------------------------------------------------------------

#: Container for the breadcrumb nav at the top of the page.
BREADCRUMB_CONTAINER = "breadcrumb"

#: Container for the operation palette accordion (left column).
PALETTE_CONTAINER = "palette"

#: Cytoscape canvas in the centre column. Phase 3 will read tap/select events
#: and write nodes/edges/elements as state changes.
CANVAS_CYTOSCAPE = "canvas-cytoscape"

#: Right-column inspector wrapper — Phase 3 swaps its children when the
#: selected node changes.
INSPECTOR_CONTAINER = "inspector"

#: Mount point for ``param_form`` output. Lives inside ``INSPECTOR_CONTAINER``.
INSPECTOR_PARAM_FORM = "inspector-param-form"

#: Mount point for the per-step preview (image thumbnail or DataTable).
INSPECTOR_PREVIEW = "inspector-preview"

#: Toggle button that opens/closes the Inspector "Documentation" collapse.
#: Rendered visibly in the non-pipeline inspector branch when the selected
#: operation has a docstring; rendered hidden in every other branch via
#: ``_hidden_inspector_widgets`` so the toggle callback's ``Input`` always
#: resolves.
INSPECTOR_DOC_TOGGLE = "inspector-doc-toggle"

#: Collapse holding the operation class docstring. ``is_open=False`` on
#: every fresh inspector render (selection change rebuilds the inspector
#: from scratch, so the section naturally re-collapses on node switch).
INSPECTOR_DOC_COLLAPSE = "inspector-doc-collapse"

#: Footer row holding image-source + run/save/load buttons.
FOOTER_CONTAINER = "footer"


# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------

#: Triggers ``apply_with_intermediates`` against the current pipeline.
BTN_RUN_PREVIEW = "btn-run-preview"

#: Writes ``pipeline.to_json()`` to the path in INPUT_SAVE_PATH.
BTN_SAVE = "btn-save"

#: Reads JSON from INPUT_LOAD_PATH and replaces the current state.
BTN_LOAD = "btn-load"

#: Adds a fresh ``ImagePipeline`` step to the current scope.
BTN_NEW_PIPELINE_NODE = "btn-new-pipeline-node"

#: Removes the currently selected node from the visible scope.
BTN_DELETE_NODE = "btn-delete-node"

#: Toolbar button next to "Delete selected" for deleting the selected aux wire.
BTN_DELETE_WIRE = "btn-delete-wire"

#: Pops the breadcrumb (returns to the parent scope).
BTN_DRILL_OUT = "btn-drill-out"

#: Drills into the currently selected pipeline-typed node. The inspector's
#: visible "Drill in ▸" button uses this id when a pipeline node is selected;
#: a hidden placeholder with the same id is rendered in every other inspector
#: branch so the fan-in callback's ``Input`` always resolves.
BTN_DRILL_IN = "btn-drill-in"

#: Recenters / refits the cytoscape canvas on the current pipeline. Useful
#: after a drag has scrolled nodes off-screen.
BTN_CANVAS_FIT = "btn-canvas-fit"

#: Zoom the cytoscape canvas in by a fixed factor.
BTN_CANVAS_ZOOM_IN = "btn-canvas-zoom-in"

#: Zoom the cytoscape canvas out by a fixed factor.
BTN_CANVAS_ZOOM_OUT = "btn-canvas-zoom-out"

#: Hidden ``dcc.Store`` whose data is bumped by clientside callbacks that
#: drive the cytoscape canvas (zoom / fit). Only exists to satisfy Dash's
#: "every callback needs an Output" rule for action-style buttons.
STORE_CANVAS_CONTROL = "store-canvas-control"

#: Active image path. The directory browser writes it on "Use this path" /
#: "Use synthetic plate"; Run preview reads it as State.
STORE_IMAGE_PATH = "store-image-path"

#: Plain ``html.Div`` id whose ``children`` get re-rendered to show the
#: currently-loaded image basename below the "Load image" / "Use synthetic
#: plate" buttons.
ACTIVE_IMAGE_LABEL = "active-image-label"

#: Opens the Load Image modal. Replaces the inline directory picker that used
#: to live in the footer.
BTN_LOAD_IMAGE = "btn-load-image"

#: Top-level "Use synthetic plate" shortcut surfaced alongside the new
#: ``BTN_LOAD_IMAGE`` so muscle memory from the inline picker carries over.
BTN_USE_SYNTHETIC = "btn-use-synthetic"

#: Synthetic-plate shortcut rendered inside the Load Image modal footer. Must
#: be a distinct id from ``BTN_USE_SYNTHETIC`` because both buttons live in
#: the DOM at the same time.
BTN_USE_SYNTHETIC_MODAL = "btn-use-synthetic-modal"

#: ``dbc.Modal`` housing the JSON-vs-Prefab chooser triggered by ``BTN_LOAD``.
#: Body content swaps between the chooser, the JSON browser, and the prefab
#: list within this single modal.
MODAL_LOAD_PICKER = "modal-load-picker"

#: ``dcc.Store`` holding the active "page" of ``MODAL_LOAD_PICKER``: one of
#: ``"chooser"``, ``"json"``, ``"prefab"``. Drives the body swap.
STORE_LOAD_PICKER_PAGE = "store-load-picker-page"

#: Container inside ``MODAL_LOAD_PICKER`` whose children are re-rendered when
#: ``STORE_LOAD_PICKER_PAGE`` changes.
MODAL_LOAD_PICKER_BODY = "modal-load-picker-body"

#: Buttons inside the chooser page that switch ``STORE_LOAD_PICKER_PAGE``.
BTN_LOAD_JSON_CHOICE = "btn-load-json-choice"
BTN_LOAD_PREFAB_CHOICE = "btn-load-prefab-choice"

#: "Back" button shown on the JSON / Prefab pages of the load chooser.
BTN_LOAD_PICKER_BACK = "btn-load-picker-back"

#: ``dcc.Store`` of the directory currently being viewed in the JSON
#: browser (string path or ``None``). Re-renders the tree on update.
STORE_BROWSE_DIR_JSON = "store-browse-dir-json"

#: ``dbc.Modal`` housing the Save Pipeline browser triggered by ``BTN_SAVE``.
MODAL_SAVE = "modal-save"

#: Container inside ``MODAL_SAVE`` whose children are the folder-only tree.
MODAL_SAVE_BODY = "modal-save-body"

#: ``dcc.Store`` of the directory currently being viewed in the Save modal.
STORE_BROWSE_DIR_SAVE = "store-browse-dir-save"

#: Filename input inside ``MODAL_SAVE``. Defaults to ``"pipeline.json"``.
INPUT_SAVE_FILENAME = "input-save-filename"

#: Save button inside ``MODAL_SAVE``. Distinct from ``BTN_SAVE`` (which now
#: only opens the modal).
BTN_SAVE_CONFIRM = "btn-save-confirm"

#: Cancel button on ``MODAL_SAVE``.
BTN_SAVE_CANCEL = "btn-save-cancel"

#: ``dbc.Modal`` housing the Load Image browser triggered by
#: ``BTN_LOAD_IMAGE``.
MODAL_LOAD_IMAGE = "modal-load-image"

#: Container inside ``MODAL_LOAD_IMAGE`` whose children are the image tree.
MODAL_LOAD_IMAGE_BODY = "modal-load-image-body"

#: ``dcc.Store`` of the directory currently being viewed in the Load Image
#: modal.
STORE_BROWSE_DIR_IMAGE = "store-browse-dir-image"

#: ``id_type`` constants passed to :func:`directory_tree` so each modal's
#: pattern-matching callback can subscribe without conflicting with the
#: others.
DIR_ENTRY_TYPE_IMAGE = "dir-entry-image"
DIR_ENTRY_TYPE_JSON = "dir-entry-json"
DIR_ENTRY_TYPE_SAVE = "dir-entry-save"


# ---------------------------------------------------------------------------
# Point picker modal
# ---------------------------------------------------------------------------

#: Container id for the point-picker dbc.Modal.
MODAL_POINT_PICKER = "modal-point-picker"

#: Mount point for the OpenSeadragon viewer inside the modal body.
PICKER_OSD_DIV = "picker-osd"

#: Inline help line under the channel radio.
PICKER_CHANNEL_HELP = "picker-channel-help"

#: Channel toggle (RGB vs predecessor intermediate).
PICKER_CHANNEL_RADIO = "picker-channel-radio"

#: Stores used by the modal. PICKER_STAGED_STORE holds the in-flight list of
#: ``[y, x]`` picks; PICKER_TARGET_STORE remembers which node.param the modal
#: writes back to on Confirm; PICKER_DZI_URL_STORE drives the OSD mount;
#: PICKER_CHANNEL_AVAIL_STORE flips the intermediate radio off when the
#: predecessor's preview hasn't been cached yet.
PICKER_STAGED_STORE = "picker-staged-store"
PICKER_TARGET_STORE = "picker-target-store"
PICKER_DZI_URL_STORE = "picker-dzi-url-store"
PICKER_CHANNEL_AVAIL_STORE = "picker-channel-avail-store"

#: Write-only sink for the clientside mount/redraw/dispose callbacks. Holds
#: a monotonic timestamp; nothing downstream reads it.
PICKER_OSD_MOUNT_TRIGGER = "picker-osd-mount-trigger"

#: Count label inside the modal body ("3 points").
PICKER_COUNT_LABEL = "picker-count-label"

#: Modal action buttons.
BTN_PICKER_CLEAR = "btn-picker-clear"
BTN_PICKER_UNDO = "btn-picker-undo"
BTN_PICKER_CANCEL = "btn-picker-cancel"
BTN_PICKER_CONFIRM = "btn-picker-confirm"

#: Pattern-match ``type`` keys for the per-node picker components emitted
#: by :func:`phenotypic.gui.builder._param_form._picker_widget`. Both layout
#: (``_param_form``) and callbacks (``_callbacks``, ``_point_picker``) match
#: against these strings; one source of truth avoids typo drift.
PICKER_PARAM_STORE_TYPE = "param-point-picker-store"
PICKER_PARAM_BTN_TYPE = "param-point-picker-btn"
PICKER_PARAM_COUNT_TYPE = "param-point-picker-count"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: Server-side path to write the current pipeline JSON to.
#:
#: .. deprecated:: superseded by ``MODAL_SAVE`` + ``INPUT_SAVE_FILENAME``;
#:    retained only because external callers may still reference the symbol.
INPUT_SAVE_PATH = "input-save-path"

#: Server-side path to read pipeline JSON from.
#:
#: .. deprecated:: superseded by ``MODAL_LOAD_PICKER`` (JSON page); retained
#:    only because external callers may still reference the symbol.
INPUT_LOAD_PATH = "input-load-path"

#: Optional ``nrows`` override for grid pipelines (root scope only).
INPUT_NROWS = "input-nrows"

#: Optional ``ncols`` override for grid pipelines (root scope only).
INPUT_NCOLS = "input-ncols"

#: Inspector text input for editing the selected node's display label.
INPUT_NODE_LABEL = "input-node-label"


# ---------------------------------------------------------------------------
# Toast / status
# ---------------------------------------------------------------------------

#: Floating toast used for save/load/preview success and error messages.
TOAST_NOTIFICATION = "toast-notification"

#: ``dcc.Loading`` wrapper around the preview button + canvas / inspector
#: previews so the spinner appears while ``apply_with_intermediates`` runs.
PREVIEW_LOADING = "preview-loading"


# ---------------------------------------------------------------------------
# Pattern-matching id-builders
# ---------------------------------------------------------------------------


def palette_button_id(class_name: str) -> Dict[str, Any]:
    """Build the pattern-matching id for an operation-palette button.

    Args:
        class_name: Registry key (e.g. ``"GaussianBlur"``) the button adds.

    Returns:
        Dict of shape ``{"type": "palette-add", "class_name": class_name}``.
        Phase 3 callbacks should match ``Input({"type": "palette-add",
        "class_name": ALL}, "n_clicks")``.
    """

    return {"type": "palette-add", "class_name": class_name}


def breadcrumb_link_id(depth: int) -> Dict[str, Any]:
    """Build the pattern-matching id for a clickable breadcrumb segment.

    Args:
        depth: Position in the breadcrumb (0 = root, 1 = first child, ...).

    Returns:
        Dict of shape ``{"type": "breadcrumb-link", "depth": depth}``.
    """

    return {"type": "breadcrumb-link", "depth": depth}


def prefab_card_id(class_name: str) -> Dict[str, Any]:
    """Build the pattern-matching id for a prefab-pipeline picker card.

    Args:
        class_name: ``phenotypic.prefab`` registry key shown on the card
            (e.g. ``"HeavyOtsuPipeline"``).

    Returns:
        Dict of shape ``{"type": "prefab-card", "class_name": class_name}``.
        The Load Picker callback subscribes via ``Input({"type":
        "prefab-card", "class_name": ALL}, "n_clicks")``.
    """

    return {"type": "prefab-card", "class_name": class_name}


# ---------------------------------------------------------------------------
# Main-port pattern-matching helpers
# ---------------------------------------------------------------------------
#
# Cytoscape elements require flat string ids (cytoscape rejects dict ids
# outside the Dash pattern-matching layer). We mangle the structured
# components into delimited strings so callbacks reading ``tapNodeData`` can
# recover the structured form via the decoder helpers without a lookup
# table. Choice of ``__`` as separator avoids collision with single
# underscores in realistic class/param names.


#: Prefix for cytoscape main-input port node ids.
_MAIN_INPUT_PREFIX: str = "main-input"

#: Prefix for cytoscape main-output port node ids.
_MAIN_OUTPUT_PREFIX: str = "main-output"

#: Separator used to delimit fields inside the encoded flat string ids.
#: Mirrors the convention previously used for ``port-handle__...`` ids.
_PORT_ID_SEP: str = "__"


def main_input_port_id(node_id: str) -> str:
    """Build the cytoscape element id for a node's main-input port.

    Each ribbon operation renders a small blue circle on its LEFT edge
    representing the image-flow input. Image-flow edges connect the
    upstream node's main-output port to this main-input port.

    Args:
        node_id: ``node_id`` of the operation the port attaches to.

    Returns:
        Flat string ``"main-input__<node_id>"`` suitable as a cytoscape
        element id. Use :func:`_decode_main_port_id` in callbacks reading
        ``tapNodeData`` to recover the ``node_id``.
    """

    return _encode_main_port_id(_MAIN_INPUT_PREFIX, node_id)


def main_output_port_id(node_id: str) -> str:
    """Build the cytoscape element id for a node's main-output port.

    Each ribbon operation renders a small blue circle on its RIGHT edge
    representing the image-flow output. Image-flow edges connect this
    main-output port to the downstream node's main-input port.

    Args:
        node_id: ``node_id`` of the operation the port attaches to.

    Returns:
        Flat string ``"main-output__<node_id>"`` suitable as a cytoscape
        element id. Use :func:`_decode_main_port_id` in callbacks reading
        ``tapNodeData`` to recover the ``node_id``.
    """

    return _encode_main_port_id(_MAIN_OUTPUT_PREFIX, node_id)


def _encode_main_port_id(prefix: str, node_id: str) -> str:
    """Mangle (prefix, node_id) into a flat cytoscape id for a main port.

    Used by :func:`main_input_port_id` and :func:`main_output_port_id`;
    callbacks decode via :func:`_decode_main_port_id`.

    Args:
        prefix: Either :data:`_MAIN_INPUT_PREFIX` or
            :data:`_MAIN_OUTPUT_PREFIX`.
        node_id: Ribbon node identifier the port attaches to.

    Returns:
        Flat string ``"<prefix>__<node_id>"``.
    """

    return _PORT_ID_SEP.join([prefix, node_id])


def _decode_main_port_id(encoded: str) -> Optional[tuple[str, str]]:
    """Reverse of :func:`_encode_main_port_id`.

    Returns ``None`` for any string that doesn't match the main-input or
    main-output encoding (e.g. a ribbon node).

    Args:
        encoded: Cytoscape element id string.

    Returns:
        ``(side, node_id)`` tuple where ``side`` is either
        ``"main-input"`` or ``"main-output"``, otherwise ``None``.
    """

    for prefix in (_MAIN_INPUT_PREFIX, _MAIN_OUTPUT_PREFIX):
        head = prefix + _PORT_ID_SEP
        if encoded.startswith(head):
            return prefix, encoded[len(head):]
    return None


__all__ = [
    "LoadPickerPage",
    "StageName",
    "STORE_BUILDER_STATE",
    "STORE_SESSION_ID",
    "STORE_INTERMEDIATE_KEYS",
    "BREADCRUMB_CONTAINER",
    "PALETTE_CONTAINER",
    "CANVAS_CYTOSCAPE",
    "INSPECTOR_CONTAINER",
    "INSPECTOR_PARAM_FORM",
    "INSPECTOR_PREVIEW",
    "INSPECTOR_DOC_TOGGLE",
    "INSPECTOR_DOC_COLLAPSE",
    "FOOTER_CONTAINER",
    "BTN_RUN_PREVIEW",
    "BTN_SAVE",
    "BTN_LOAD",
    "BTN_NEW_PIPELINE_NODE",
    "BTN_DELETE_NODE",
    "BTN_DELETE_WIRE",
    "BTN_DRILL_OUT",
    "BTN_DRILL_IN",
    "BTN_CANVAS_FIT",
    "BTN_CANVAS_ZOOM_IN",
    "BTN_CANVAS_ZOOM_OUT",
    "STORE_CANVAS_CONTROL",
    "STORE_IMAGE_PATH",
    "ACTIVE_IMAGE_LABEL",
    "BTN_LOAD_IMAGE",
    "BTN_USE_SYNTHETIC",
    "BTN_USE_SYNTHETIC_MODAL",
    "MODAL_LOAD_PICKER",
    "STORE_LOAD_PICKER_PAGE",
    "MODAL_LOAD_PICKER_BODY",
    "BTN_LOAD_JSON_CHOICE",
    "BTN_LOAD_PREFAB_CHOICE",
    "BTN_LOAD_PICKER_BACK",
    "STORE_BROWSE_DIR_JSON",
    "MODAL_SAVE",
    "MODAL_SAVE_BODY",
    "STORE_BROWSE_DIR_SAVE",
    "INPUT_SAVE_FILENAME",
    "BTN_SAVE_CONFIRM",
    "BTN_SAVE_CANCEL",
    "MODAL_LOAD_IMAGE",
    "MODAL_LOAD_IMAGE_BODY",
    "STORE_BROWSE_DIR_IMAGE",
    "DIR_ENTRY_TYPE_IMAGE",
    "DIR_ENTRY_TYPE_JSON",
    "DIR_ENTRY_TYPE_SAVE",
    "MODAL_POINT_PICKER",
    "PICKER_OSD_DIV",
    "PICKER_CHANNEL_HELP",
    "PICKER_CHANNEL_RADIO",
    "PICKER_STAGED_STORE",
    "PICKER_TARGET_STORE",
    "PICKER_DZI_URL_STORE",
    "PICKER_CHANNEL_AVAIL_STORE",
    "PICKER_OSD_MOUNT_TRIGGER",
    "PICKER_COUNT_LABEL",
    "BTN_PICKER_CLEAR",
    "BTN_PICKER_UNDO",
    "BTN_PICKER_CANCEL",
    "BTN_PICKER_CONFIRM",
    "PICKER_PARAM_STORE_TYPE",
    "PICKER_PARAM_BTN_TYPE",
    "PICKER_PARAM_COUNT_TYPE",
    "INPUT_SAVE_PATH",
    "INPUT_LOAD_PATH",
    "INPUT_NROWS",
    "INPUT_NCOLS",
    "INPUT_NODE_LABEL",
    "TOAST_NOTIFICATION",
    "PREVIEW_LOADING",
    "palette_button_id",
    "breadcrumb_link_id",
    "prefab_card_id",
    "main_input_port_id",
    "main_output_port_id",
    "_encode_main_port_id",
    "_decode_main_port_id",
    # Phase 2 DAG redesign additions
    "STORE_VIEWPORT_OP",
    "STORE_ISSUES",
    "STORE_ASSET_STATUS",
    "STORE_PALETTE_DROP",
    "BTN_RELAYOUT",
    "BTN_REANCHOR",
    "INSPECTOR_EMPTY_STATE",
    "INSPECTOR_INPUT_IMAGE_CARD",
    "BANNER_ASSET_STATUS",
    "CONFIRM_DELETE_MODAL_ID",
    "BTN_CONFIRM_DELETE",
    "BTN_CANCEL_DELETE",
    "BTN_DRILL_IN_CONTAINER",
    "INPUT_CONTAINER_NAME",
    "INPUT_CONTAINER_DESC",
    "ISSUE_BADGE",
    "ISSUE_BADGE_TOOLTIP",
    "issue_row_id",
    "block_port_id",
    "edge_id",
    # DAG canvas wire + inspector additions
    "STORE_EDGE_EVENT",
    "INSPECTOR_WIRE_CARD",
    "INSPECTOR_AUX_SECTION",
    "BTN_INSPECTOR_DISCONNECT",
    "BTN_INSPECTOR_LIST_REMOVE",
    "BTN_INSPECTOR_ADD_EMPTY_SLOT",
    "BTN_INSPECTOR_LIST_MOVE",
    "STORE_INSPECTOR_LIST_REORDER",
    "inspector_disconnect_id",
    "inspector_list_remove_id",
    "inspector_add_empty_slot_id",
    "inspector_list_move_id",
    "inspector_list_reorder_store_id",
]

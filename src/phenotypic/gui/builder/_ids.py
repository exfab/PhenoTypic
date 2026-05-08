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

from typing import Any, Dict, Literal

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
    "FOOTER_CONTAINER",
    "BTN_RUN_PREVIEW",
    "BTN_SAVE",
    "BTN_LOAD",
    "BTN_NEW_PIPELINE_NODE",
    "BTN_DELETE_NODE",
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
]

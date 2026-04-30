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

from typing import Any, Dict


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


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

#: Server-side path to write the current pipeline JSON to.
INPUT_SAVE_PATH = "input-save-path"

#: Server-side path to read pipeline JSON from.
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


__all__ = [
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
    "INPUT_SAVE_PATH",
    "INPUT_LOAD_PATH",
    "INPUT_NROWS",
    "INPUT_NCOLS",
    "INPUT_NODE_LABEL",
    "TOAST_NOTIFICATION",
    "PREVIEW_LOADING",
    "palette_button_id",
    "breadcrumb_link_id",
]

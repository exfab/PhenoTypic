"""Run console component IDs.

All Run console IDs are prefixed with ``rc-`` (and pattern-matching dict
``type`` values with ``rc-``) so they cannot collide with builder /
results-viewer IDs once the unified hub mounts the apps side-by-side.

Notes:
    - Static (non-pattern-matching) IDs are plain ``str`` constants.
    - Pattern-matching IDs are returned by helper functions; ``ALL`` /
      ``MATCH`` callbacks subscribe via the dict literal.
"""
from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Top-level layout regions
# ---------------------------------------------------------------------------

#: Top-level run-console container.
RC_ROOT = "rc-root"

#: Form column wrapping pickers + mode + advanced + slurm.
RC_FORM_COL = "rc-form-col"

#: Right-hand iframe panel container.
RC_IFRAME_PANEL = "rc-iframe-panel"

#: The dashboard iframe itself; ``src`` is rewritten by callbacks.
RC_IFRAME = "rc-iframe"

#: Empty-state placeholder div shown before any run is started. Its
#: ``style.display`` is toggled when a real run lights up the iframe.
RC_IFRAME_PLACEHOLDER = "rc-iframe-placeholder"

#: ``<pre>`` element for the live log tail (monospace, scroll-bottom).
RC_LOG_TAIL = "rc-log-tail"

#: Recent Runs panel container.
RC_RECENTS = "rc-recents"

#: ``html.Tbody`` whose rows are pattern-matched to recent runs.
RC_RECENTS_BODY = "rc-recents-body"

#: Status banner below the iframe (mode + run-id + status).
RC_STATUS_BANNER = "rc-status-banner"


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

#: Holds the JSON form state (see ``RunConsoleState``).
RC_STORE_FORM_STATE = "rc-store-form-state"

#: Holds the active run id (string) so the log-tail interval can target it.
RC_STORE_ACTIVE_RUN_ID = "rc-store-active-run-id"

#: Holds the active run's ``rel_path`` so the iframe ``src`` can be set.
RC_STORE_ACTIVE_REL_PATH = "rc-store-active-rel-path"

#: Holds the selected pipeline JSON path (or ``None``).
RC_STORE_PIPELINE_PATH = "rc-store-pipeline-path"

#: Holds the selected input directory path (or ``None``).
RC_STORE_INPUT_DIR = "rc-store-input-dir"

#: Holds the selected output directory path (or ``None``).
RC_STORE_OUTPUT_DIR = "rc-store-output-dir"

#: Holds the directory currently being browsed in each modal.
RC_STORE_BROWSE_DIR_PIPELINE = "rc-store-browse-dir-pipeline"
RC_STORE_BROWSE_DIR_INPUT = "rc-store-browse-dir-input"
RC_STORE_BROWSE_DIR_OUTPUT = "rc-store-browse-dir-output"

#: Counter bumped when the recent-runs panel needs a refresh (e.g. after
#: a new run starts).
RC_STORE_RECENTS_REFRESH = "rc-store-recents-refresh"


# ---------------------------------------------------------------------------
# Form fields (mode + checkboxes + advanced + slurm)
# ---------------------------------------------------------------------------

#: Local / SLURM radio.
RC_RADIO_MODE = "rc-radio-mode"

#: Inline checklist with ``dry-run`` and ``resume`` options.
RC_CHECKS_FLAGS = "rc-checks-flags"

#: Advanced collapse wrapper.
RC_COLLAPSE_ADVANCED = "rc-collapse-advanced"
RC_BTN_TOGGLE_ADVANCED = "rc-btn-toggle-advanced"

#: SLURM collapse wrapper.
RC_COLLAPSE_SLURM = "rc-collapse-slurm"
RC_BTN_TOGGLE_SLURM = "rc-btn-toggle-slurm"

#: Advanced fields.
RC_INPUT_SAMPLE = "rc-input-sample"
RC_INPUT_NROWS = "rc-input-nrows"
RC_INPUT_NCOLS = "rc-input-ncols"
RC_INPUT_IMAGE_TYPE = "rc-input-image-type"
RC_INPUT_WORKERS = "rc-input-workers"
RC_INPUT_LOG_LEVEL = "rc-input-log-level"

#: SLURM typed common fields.
RC_INPUT_SLURM_PARTITION = "rc-input-slurm-partition"
RC_INPUT_SLURM_TIME = "rc-input-slurm-time"
RC_INPUT_SLURM_MEM = "rc-input-slurm-mem"
RC_INPUT_SLURM_CPUS = "rc-input-slurm-cpus"
RC_INPUT_SLURM_GPUS = "rc-input-slurm-gpus"

#: Free-form ``k=v`` SLURM textarea (one entry per line).
RC_INPUT_SLURM_EXTRA = "rc-input-slurm-extra"

#: Staged-GPU controls. The section is mounted but hidden until callbacks
#: determine that the selected pipeline contains a ``GpuDetector``.
RC_STAGED_GPU_SECTION = "rc-staged-gpu-section"
RC_INPUT_GPU_SLURM = "rc-input-gpu-slurm"
RC_INPUT_GPU_SHARDS = "rc-input-gpu-shards"


# ---------------------------------------------------------------------------
# Picker buttons and modal IDs
# ---------------------------------------------------------------------------

RC_BTN_PICK_PIPELINE = "rc-btn-pick-pipeline"
RC_BTN_PICK_INPUT = "rc-btn-pick-input"
RC_BTN_PICK_OUTPUT = "rc-btn-pick-output"

#: Selected-path display labels (next to each picker button).
RC_LABEL_PIPELINE = "rc-label-pipeline"
RC_LABEL_INPUT = "rc-label-input"
RC_LABEL_OUTPUT = "rc-label-output"

#: Pipeline-JSON picker modal.
RC_MODAL_PIPELINE = "rc-modal-pipeline"
RC_MODAL_PIPELINE_BODY = "rc-modal-pipeline-body"
RC_BTN_PIPELINE_CANCEL = "rc-btn-pipeline-cancel"
RC_BTN_PIPELINE_CONFIRM = "rc-btn-pipeline-confirm"

#: Input-directory picker modal.
RC_MODAL_INPUT = "rc-modal-input"
RC_MODAL_INPUT_BODY = "rc-modal-input-body"
RC_BTN_INPUT_CANCEL = "rc-btn-input-cancel"
RC_BTN_INPUT_CONFIRM = "rc-btn-input-confirm"

#: Output-directory picker modal.
RC_MODAL_OUTPUT = "rc-modal-output"
RC_MODAL_OUTPUT_BODY = "rc-modal-output-body"
RC_INPUT_OUTPUT_PATH = "rc-input-output-path"
RC_BTN_OUTPUT_CANCEL = "rc-btn-output-cancel"
RC_BTN_OUTPUT_CONFIRM = "rc-btn-output-confirm"


# ---------------------------------------------------------------------------
# Action buttons (validate / run / cancel / save preset / load preset)
# ---------------------------------------------------------------------------

RC_BTN_VALIDATE = "rc-btn-validate"
RC_BTN_RUN = "rc-btn-run"
RC_BTN_CANCEL = "rc-btn-cancel"
RC_BTN_REFRESH_DASHBOARD = "rc-btn-refresh-dashboard"
RC_BTN_SAVE_PRESET = "rc-btn-save-preset"
RC_INPUT_PRESET_NAME = "rc-input-preset-name"
RC_DROPDOWN_LOAD_PRESET = "rc-dropdown-load-preset"


# ---------------------------------------------------------------------------
# Sidebar hand-off banner — consumes ``SHELL_SIDEBAR_SELECTION_STORE``.
# ---------------------------------------------------------------------------

#: Banner container; hidden when no sidebar selection is active.
RC_HANDOFF_BANNER = "rc-handoff-banner"

#: Label inside the banner — shows the selected sandbox-relative path.
RC_HANDOFF_LABEL = "rc-handoff-label"

#: "Set as pipeline" button (enabled when selection looks like a JSON).
RC_HANDOFF_USE_PIPELINE = "rc-handoff-use-pipeline"

#: "Set as input dir" button (enabled when selection is a directory).
RC_HANDOFF_USE_INPUT = "rc-handoff-use-input"

#: "Set as output dir" button (enabled when selection is a directory).
RC_HANDOFF_USE_OUTPUT = "rc-handoff-use-output"

#: Dismiss button — clears ``SHELL_SIDEBAR_SELECTION_STORE``.
RC_HANDOFF_DISMISS = "rc-handoff-dismiss"


# ---------------------------------------------------------------------------
# Toast + intervals
# ---------------------------------------------------------------------------

RC_TOAST = "rc-toast"
RC_INTERVAL_LOG = "rc-interval-log"
RC_INTERVAL_DASHBOARD_POLL = "rc-interval-dashboard-poll"


# ---------------------------------------------------------------------------
# ``id_type`` constants for directory-tree pattern-matching
# ---------------------------------------------------------------------------

#: Pipeline-JSON tree dir-entry ``type`` value.
RC_DIR_ENTRY_TYPE_PIPELINE_JSON = "rc-dir-entry-pipeline-json"

#: Input-directory tree dir-entry ``type`` value.
RC_DIR_ENTRY_TYPE_INPUT_DIR = "rc-dir-entry-input-dir"

#: Output-directory tree dir-entry ``type`` value.
RC_DIR_ENTRY_TYPE_OUTPUT_DIR = "rc-dir-entry-output-dir"


# ---------------------------------------------------------------------------
# Pattern-matching id helpers
# ---------------------------------------------------------------------------


def recents_row_id(rel_path: str) -> Dict[str, Any]:
    """Build the pattern-matching id for one Recent Runs row.

    Args:
        rel_path: Sandbox-relative path of the run's output directory.

    Returns:
        Dict ``{"type": "rc-recents-row", "rel_path": rel_path}``.
    """
    return {"type": "rc-recents-row", "rel_path": rel_path}


__all__ = [
    "RC_ROOT",
    "RC_FORM_COL",
    "RC_IFRAME_PANEL",
    "RC_IFRAME",
    "RC_IFRAME_PLACEHOLDER",
    "RC_LOG_TAIL",
    "RC_RECENTS",
    "RC_RECENTS_BODY",
    "RC_STATUS_BANNER",
    "RC_STORE_FORM_STATE",
    "RC_STORE_ACTIVE_RUN_ID",
    "RC_STORE_ACTIVE_REL_PATH",
    "RC_STORE_PIPELINE_PATH",
    "RC_STORE_INPUT_DIR",
    "RC_STORE_OUTPUT_DIR",
    "RC_STORE_BROWSE_DIR_PIPELINE",
    "RC_STORE_BROWSE_DIR_INPUT",
    "RC_STORE_BROWSE_DIR_OUTPUT",
    "RC_STORE_RECENTS_REFRESH",
    "RC_RADIO_MODE",
    "RC_CHECKS_FLAGS",
    "RC_COLLAPSE_ADVANCED",
    "RC_BTN_TOGGLE_ADVANCED",
    "RC_COLLAPSE_SLURM",
    "RC_BTN_TOGGLE_SLURM",
    "RC_INPUT_SAMPLE",
    "RC_INPUT_NROWS",
    "RC_INPUT_NCOLS",
    "RC_INPUT_IMAGE_TYPE",
    "RC_INPUT_WORKERS",
    "RC_INPUT_LOG_LEVEL",
    "RC_INPUT_SLURM_PARTITION",
    "RC_INPUT_SLURM_TIME",
    "RC_INPUT_SLURM_MEM",
    "RC_INPUT_SLURM_CPUS",
    "RC_INPUT_SLURM_GPUS",
    "RC_INPUT_SLURM_EXTRA",
    "RC_STAGED_GPU_SECTION",
    "RC_INPUT_GPU_SLURM",
    "RC_INPUT_GPU_SHARDS",
    "RC_BTN_PICK_PIPELINE",
    "RC_BTN_PICK_INPUT",
    "RC_BTN_PICK_OUTPUT",
    "RC_LABEL_PIPELINE",
    "RC_LABEL_INPUT",
    "RC_LABEL_OUTPUT",
    "RC_MODAL_PIPELINE",
    "RC_MODAL_PIPELINE_BODY",
    "RC_BTN_PIPELINE_CANCEL",
    "RC_BTN_PIPELINE_CONFIRM",
    "RC_MODAL_INPUT",
    "RC_MODAL_INPUT_BODY",
    "RC_BTN_INPUT_CANCEL",
    "RC_BTN_INPUT_CONFIRM",
    "RC_MODAL_OUTPUT",
    "RC_MODAL_OUTPUT_BODY",
    "RC_INPUT_OUTPUT_PATH",
    "RC_BTN_OUTPUT_CANCEL",
    "RC_BTN_OUTPUT_CONFIRM",
    "RC_BTN_VALIDATE",
    "RC_BTN_RUN",
    "RC_BTN_CANCEL",
    "RC_BTN_REFRESH_DASHBOARD",
    "RC_BTN_SAVE_PRESET",
    "RC_INPUT_PRESET_NAME",
    "RC_HANDOFF_BANNER",
    "RC_HANDOFF_LABEL",
    "RC_HANDOFF_USE_PIPELINE",
    "RC_HANDOFF_USE_INPUT",
    "RC_HANDOFF_USE_OUTPUT",
    "RC_HANDOFF_DISMISS",
    "RC_DROPDOWN_LOAD_PRESET",
    "RC_TOAST",
    "RC_INTERVAL_LOG",
    "RC_INTERVAL_DASHBOARD_POLL",
    "RC_DIR_ENTRY_TYPE_PIPELINE_JSON",
    "RC_DIR_ENTRY_TYPE_INPUT_DIR",
    "RC_DIR_ENTRY_TYPE_OUTPUT_DIR",
    "recents_row_id",
]

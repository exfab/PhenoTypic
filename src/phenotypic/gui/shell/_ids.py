"""Component IDs for the PhenoTypic GUI shell.

Single source of truth for all chrome / sidebar / per-tool-store IDs. Layout
builders and callback registrars import from here so a rename in one place
flows everywhere.

Naming convention
    Every shell-owned ID is prefixed ``"shell-"`` (kebab-case). Both the
    builder and the results viewer use unprefixed kebab-case (``"palette"``,
    ``"results-viewer-tabs"``); the ``"shell-"`` prefix is collision-proof
    against both. Per-tool path stores keep the tool's prefix
    (``"builder-image-root"``, ``"viewer-output-root"``, ``"run-*"``) because
    the chrome wrapper owns their wiring even though the tools own their
    semantics.

Pattern-matching IDs use ``str``-typed component IDs for static elements;
dict-typed pattern IDs are reserved for sidebar entries (one per directory
node) so the same callback can fire for every clicked entry.
"""
from __future__ import annotations

from typing import Literal

#: Closed set of tool names used in release-button / release-status IDs.
ToolName = Literal["viewer", "analysis", "builder", "run", "tune"]

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

#: Container holding the top-bar tab nav, root display, RSS readout, help btn.
SHELL_TOP_BAR = "shell-top-bar"

#: Read-only label echoing the resolved sandbox root.
SHELL_ROOT_LABEL = "shell-root-label"

#: Top-bar settings button and popover.
SHELL_SETTINGS_BUTTON = "shell-settings-button"
SHELL_SETTINGS_POPOVER = "shell-settings-popover"
SHELL_SETTINGS_INPUT_FOLDER_PICK = "shell-settings-input-folder-pick"
SHELL_SETTINGS_INPUT_FOLDER_CLEAR = "shell-settings-input-folder-clear"
SHELL_SETTINGS_METADATA_CSV_LABEL = "shell-settings-metadata-csv-label"
SHELL_SETTINGS_METADATA_CSV_PICK = "shell-settings-metadata-csv-pick"
SHELL_SETTINGS_METADATA_CSV_CLEAR = "shell-settings-metadata-csv-clear"

#: Browser-local store holding the shared source-image-root payload.
SHELL_SOURCE_IMAGE_ROOT_STORE = "shell-source-image-root-store"

#: Top-bar label summarising the selected source image directory.
SHELL_SOURCE_IMAGE_ROOT_LABEL = "shell-source-image-root-label"

#: Top-bar action clearing :data:`SHELL_SOURCE_IMAGE_ROOT_STORE`.
SHELL_SOURCE_IMAGE_ROOT_CLEAR = "shell-source-image-root-clear"

#: Source-image-root picker modal opened from the top-bar source label.
SHELL_SOURCE_IMAGE_ROOT_MODAL = "shell-source-image-root-modal"

#: Body region for the source-image-root picker directory tree.
SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY = "shell-source-image-root-modal-body"

#: Store holding the source picker directory currently being browsed.
SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE = "shell-source-image-root-browse-store"

#: Cancel action inside :data:`SHELL_SOURCE_IMAGE_ROOT_MODAL`.
SHELL_SOURCE_IMAGE_ROOT_CANCEL = "shell-source-image-root-cancel"

#: Confirm action inside :data:`SHELL_SOURCE_IMAGE_ROOT_MODAL`.
SHELL_SOURCE_IMAGE_ROOT_CONFIRM = "shell-source-image-root-confirm"

#: Pattern-matching ``type`` for source picker tree entries.
SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE = "shell-source-image-root-entry"

#: Browser-local store holding the shared metadata CSV payload.
SHELL_METADATA_CSV_STORE = "shell-metadata-csv-store"

#: Metadata CSV picker modal opened from the settings popover.
SHELL_METADATA_CSV_MODAL = "shell-metadata-csv-modal"
SHELL_METADATA_CSV_MODAL_BODY = "shell-metadata-csv-modal-body"
SHELL_METADATA_CSV_BROWSE_STORE = "shell-metadata-csv-browse-store"
SHELL_METADATA_CSV_CANCEL = "shell-metadata-csv-cancel"
SHELL_METADATA_CSV_CONFIRM = "shell-metadata-csv-confirm"
SHELL_METADATA_CSV_ENTRY_TYPE = "shell-metadata-csv-entry"

#: Memory readout (``psutil.Process().memory_info().rss``); refreshed by
#: ``SHELL_RSS_INTERVAL``.
SHELL_RSS_LABEL = "shell-rss-label"

#: 5-second interval driving the RSS readout.
SHELL_RSS_INTERVAL = "shell-rss-interval"

#: "?" help button.
SHELL_HELP_BUTTON = "shell-help-button"

#: Help modal opened by ``SHELL_HELP_BUTTON``.
SHELL_HELP_MODAL = "shell-help-modal"

#: Tab-nav anchors. Active tab is server-rendered (Phase 5 will make these
#: real cross-app navigations; Phase 3 keeps them as ``html.A`` elements).
SHELL_TAB_HOME = "shell-tab-home"
SHELL_TAB_BUILDER = "shell-tab-builder"
SHELL_TAB_VIEWER = "shell-tab-viewer"
SHELL_TAB_RUN = "shell-tab-run"
SHELL_TAB_ANALYSIS = "shell-tab-analysis"
SHELL_TAB_TUNE = "shell-tab-tune"
SHELL_TAB_BROWSE = "shell-tab-browse"

#: Dropdown-group toggle ids. The flat tab strip is consolidated into two
#: grouped dropdowns: **Pipeline** (Builder / Run) and **Results**
#: (Viewer / Analysis). Home stays a standalone leaf tab. The group toggle
#: carries the gold ``shell-tab-group-active`` treatment whenever one of its
#: member mounts is the active tab; the member items keep the ``SHELL_TAB_*``
#: ids above so existing selectors/handoff wiring are untouched.
SHELL_TAB_GROUP_PIPELINE = "shell-tab-group-pipeline"
SHELL_TAB_GROUP_RESULTS = "shell-tab-group-results"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

#: Sidebar container (left rail).
SHELL_SIDEBAR = "shell-sidebar"

#: Tree body (lazy expanded children render here).
SHELL_SIDEBAR_TREE = "shell-sidebar-tree"

#: Sidebar refresh button (busts classifier cache + re-renders).
SHELL_SIDEBAR_REFRESH = "shell-sidebar-refresh"

#: Sidebar "Hidden files" toggle.
SHELL_SIDEBAR_HIDDEN_TOGGLE = "shell-sidebar-hidden-toggle"

#: Sidebar "External symlinks" toggle.
SHELL_SIDEBAR_SYMLINK_TOGGLE = "shell-sidebar-symlink-toggle"

#: Top-bar button that toggles sidebar visibility (the file explorer).
SHELL_SIDEBAR_COLLAPSE_BUTTON = "shell-sidebar-collapse-button"

#: Persistent store (``localStorage``) holding the sidebar collapsed flag
#: so the state survives navigation between mounts (``/``, ``/builder/``,
#: ``/results/``, ``/run/``).
SHELL_SIDEBAR_COLLAPSE_STORE = "shell-sidebar-collapse-store"

#: Memory store: relative path of the currently-selected sidebar entry, or
#: ``None`` if nothing is selected. Per-tool ``[↩ from sidebar]`` buttons
#: read this; the sidebar writes it on click.
SHELL_SIDEBAR_SELECTION_STORE = "shell-sidebar-selection-store"

#: Memory store: list of currently-expanded directory rel-paths. Mutated
#: by the entry-click callback; consumed by the tree-render callback.
SHELL_SIDEBAR_EXPANDED_STORE = "shell-sidebar-expanded-store"

#: Memory store: shared filesystem refresh revision. Bumped by Refresh; the
#: classifier, sidebar, open pickers, and source/metadata labels consume it.
SHELL_CLASSIFIER_CACHE_STORE = "shell-classifier-cache-store"

#: Semantic name for the shared refresh revision. The existing component ID is
#: retained so older callback clients and browser tests remain compatible.
SHELL_REFRESH_REVISION_STORE = SHELL_CLASSIFIER_CACHE_STORE

#: Browser-session record for the active or most recent Results/Analysis
#: binding job. Shared by every chrome-wrapped mount so progress survives a
#: full navigation between Results and Analysis.
SHELL_RESULTS_BINDING_JOB_STORE = "shell-results-binding-job-store"

#: Short client-side polling interval enabled only while the shared binding
#: store contains a queued/running job.
SHELL_RESULTS_BINDING_POLL_INTERVAL = "shell-results-binding-poll-interval"

#: Sidebar status card for the shared Results/Analysis hand-off.
SHELL_RESULTS_BINDING_PANEL = "shell-results-binding-panel"
SHELL_RESULTS_BINDING_STATUS = "shell-results-binding-status"
SHELL_RESULTS_BINDING_PHASE = "shell-results-binding-phase"
SHELL_RESULTS_BINDING_DETAIL = "shell-results-binding-detail"
SHELL_RESULTS_BINDING_PROGRESS = "shell-results-binding-progress"
SHELL_RESULTS_BINDING_PROGRESS_LABEL = "shell-results-binding-progress-label"
SHELL_RESULTS_BINDING_DIAGNOSTIC = "shell-results-binding-diagnostic"
SHELL_RESULTS_BINDING_CANCEL = "shell-results-binding-cancel"


def sidebar_entry_id(rel_path: str) -> dict[str, str]:
    """Return a pattern-matching ID for one sidebar tree row.

    Pattern-matching IDs let one callback handle clicks on every row without
    enumerating routes. Returns ``{"type": "shell-sidebar-entry", "path":
    rel_path}``.
    """
    return {"type": "shell-sidebar-entry", "path": rel_path}


def sidebar_expand_id(rel_path: str) -> dict[str, str]:
    """Pattern-matching ID for the expand-arrow button on a sidebar row."""
    return {"type": "shell-sidebar-expand", "path": rel_path}


# ---------------------------------------------------------------------------
# Main pane (where the active tool's body renders)
# ---------------------------------------------------------------------------

SHELL_MAIN_PANE = "shell-main-pane"

# ---------------------------------------------------------------------------
# Per-tool Release button (one instance per wrapped tool; the chrome layer
# tags each with the tool name so a single callback can dispatch).
# ---------------------------------------------------------------------------

def release_button_id(tool: ToolName) -> dict[str, str]:
    """Pattern-matching ID for a per-tool Release button."""
    return {"type": "shell-release-button", "tool": tool}


def release_status_id(tool: ToolName) -> dict[str, str]:
    """Pattern-matching ID for the per-tool Release status text."""
    return {"type": "shell-release-status", "tool": tool}


# ---------------------------------------------------------------------------
# Per-tool path stores (chrome owns the [↩ from sidebar] wiring; tool owns
# the semantics).
# ---------------------------------------------------------------------------

BUILDER_IMAGE_ROOT_STORE = "builder-image-root"
VIEWER_OUTPUT_ROOT_STORE = "viewer-output-root"
RUN_PIPELINE_PATH_STORE = "run-pipeline-path"
RUN_INPUT_DIR_STORE = "run-input-dir"
RUN_OUTPUT_DIR_STORE = "run-output-dir"
TUNE_PIPELINE_PATH_STORE = "tune-pipeline-path"


__all__ = [
    "ToolName",
    "SHELL_TOP_BAR",
    "SHELL_ROOT_LABEL",
    "SHELL_SETTINGS_BUTTON",
    "SHELL_SETTINGS_POPOVER",
    "SHELL_SETTINGS_INPUT_FOLDER_PICK",
    "SHELL_SETTINGS_INPUT_FOLDER_CLEAR",
    "SHELL_SETTINGS_METADATA_CSV_LABEL",
    "SHELL_SETTINGS_METADATA_CSV_PICK",
    "SHELL_SETTINGS_METADATA_CSV_CLEAR",
    "SHELL_SOURCE_IMAGE_ROOT_STORE",
    "SHELL_SOURCE_IMAGE_ROOT_LABEL",
    "SHELL_SOURCE_IMAGE_ROOT_CLEAR",
    "SHELL_SOURCE_IMAGE_ROOT_MODAL",
    "SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY",
    "SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE",
    "SHELL_SOURCE_IMAGE_ROOT_CANCEL",
    "SHELL_SOURCE_IMAGE_ROOT_CONFIRM",
    "SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE",
    "SHELL_METADATA_CSV_STORE",
    "SHELL_METADATA_CSV_MODAL",
    "SHELL_METADATA_CSV_MODAL_BODY",
    "SHELL_METADATA_CSV_BROWSE_STORE",
    "SHELL_METADATA_CSV_CANCEL",
    "SHELL_METADATA_CSV_CONFIRM",
    "SHELL_METADATA_CSV_ENTRY_TYPE",
    "SHELL_RSS_LABEL",
    "SHELL_RSS_INTERVAL",
    "SHELL_HELP_BUTTON",
    "SHELL_HELP_MODAL",
    "SHELL_TAB_HOME",
    "SHELL_TAB_BUILDER",
    "SHELL_TAB_VIEWER",
    "SHELL_TAB_RUN",
    "SHELL_TAB_ANALYSIS",
    "SHELL_TAB_TUNE",
    "SHELL_TAB_BROWSE",
    "SHELL_TAB_GROUP_PIPELINE",
    "SHELL_TAB_GROUP_RESULTS",
    "SHELL_SIDEBAR",
    "SHELL_SIDEBAR_TREE",
    "SHELL_SIDEBAR_REFRESH",
    "SHELL_SIDEBAR_HIDDEN_TOGGLE",
    "SHELL_SIDEBAR_SYMLINK_TOGGLE",
    "SHELL_SIDEBAR_COLLAPSE_BUTTON",
    "SHELL_SIDEBAR_COLLAPSE_STORE",
    "SHELL_SIDEBAR_SELECTION_STORE",
    "SHELL_SIDEBAR_EXPANDED_STORE",
    "SHELL_CLASSIFIER_CACHE_STORE",
    "SHELL_REFRESH_REVISION_STORE",
    "SHELL_RESULTS_BINDING_JOB_STORE",
    "SHELL_RESULTS_BINDING_POLL_INTERVAL",
    "SHELL_RESULTS_BINDING_PANEL",
    "SHELL_RESULTS_BINDING_STATUS",
    "SHELL_RESULTS_BINDING_PHASE",
    "SHELL_RESULTS_BINDING_DETAIL",
    "SHELL_RESULTS_BINDING_PROGRESS",
    "SHELL_RESULTS_BINDING_PROGRESS_LABEL",
    "SHELL_RESULTS_BINDING_DIAGNOSTIC",
    "SHELL_RESULTS_BINDING_CANCEL",
    "SHELL_MAIN_PANE",
    "BUILDER_IMAGE_ROOT_STORE",
    "VIEWER_OUTPUT_ROOT_STORE",
    "RUN_PIPELINE_PATH_STORE",
    "RUN_INPUT_DIR_STORE",
    "RUN_OUTPUT_DIR_STORE",
    "TUNE_PIPELINE_PATH_STORE",
    "sidebar_entry_id",
    "sidebar_expand_id",
    "release_button_id",
    "release_status_id",
]

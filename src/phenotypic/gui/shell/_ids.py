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
ToolName = Literal["viewer", "analysis", "builder", "run"]

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

#: Container holding the top-bar tab nav, root display, RSS readout, help btn.
SHELL_TOP_BAR = "shell-top-bar"

#: Read-only label echoing the resolved sandbox root.
SHELL_ROOT_LABEL = "shell-root-label"

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

#: Memory store: classifier cache version key. Bumped by Refresh; chrome
#: callbacks watch it to know when to re-render.
SHELL_CLASSIFIER_CACHE_STORE = "shell-classifier-cache-store"


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


__all__ = [
    "ToolName",
    "SHELL_TOP_BAR",
    "SHELL_ROOT_LABEL",
    "SHELL_RSS_LABEL",
    "SHELL_RSS_INTERVAL",
    "SHELL_HELP_BUTTON",
    "SHELL_HELP_MODAL",
    "SHELL_TAB_HOME",
    "SHELL_TAB_BUILDER",
    "SHELL_TAB_VIEWER",
    "SHELL_TAB_RUN",
    "SHELL_TAB_ANALYSIS",
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
    "SHELL_MAIN_PANE",
    "BUILDER_IMAGE_ROOT_STORE",
    "VIEWER_OUTPUT_ROOT_STORE",
    "RUN_PIPELINE_PATH_STORE",
    "RUN_INPUT_DIR_STORE",
    "RUN_OUTPUT_DIR_STORE",
    "sidebar_entry_id",
    "sidebar_expand_id",
    "release_button_id",
    "release_status_id",
]

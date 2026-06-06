"""Component IDs for the PhenoTypic ``/tune/`` co-pilot sub-app.

Single source of truth for every Dash component ID the tune sub-app uses —
layout builders, callbacks, and tests import from here so a rename in one place
flows everywhere. IDs are ``tune-``-prefixed kebab-case; the shell uses
``"shell-"`` and the DispatcherMiddleware mount keeps the namespaces
independent (a tune component never collides with a builder / viewer one).

The sub-tab surface is a four-view stack — Monitor (live read), Curate
(shortlist + overlays), Space (search-space view), Launch (apply the winner).
Chunk A ships Monitor; the other three are placeholders the later chunks fill.
"""
from __future__ import annotations

from typing import Literal

#: The closed set of tune sub-tab view names. The button IDs are
#: ``tune-subtab-<name>`` and the view-container IDs are ``tune-view-<name>``;
#: the switch callback maps a clicked button to its view name (see
#: :func:`phenotypic.gui.tune._callbacks.active_view`).
SubTabName = Literal["monitor", "curate", "space", "launch"]

#: Ordered sub-tab names — drives the button row + view-container render order.
SUBTAB_ORDER: tuple[SubTabName, ...] = ("monitor", "curate", "space", "launch")

#: Human-readable labels for each sub-tab button.
SUBTAB_LABELS: dict[SubTabName, str] = {
    "monitor": "Monitor",
    "curate": "Curate",
    "space": "Space",
    "launch": "Launch",
}


def subtab_button_id(name: SubTabName) -> str:
    """Return the static ID for the sub-tab button named ``name``."""
    return f"tune-subtab-{name}"


def view_container_id(name: SubTabName) -> str:
    """Return the static ID for the view container named ``name``."""
    return f"tune-view-{name}"


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

#: Top-level container for the tune page body.
TUNE_PAGE = "tune-page"

#: Run-picker header (run path display / picker placeholder).
TUNE_RUN_HEADER = "tune-run-header"

#: The active-view store: which sub-tab's container is currently shown.
TUNE_ACTIVE_VIEW_STORE = "tune-active-view-store"

#: The run-root store: a JSON-serialisable descriptor of the bound
#: :class:`~phenotypic.gui.tune.TuneRunRoot` the poll callback re-reads from.
TUNE_RUN_ROOT_STORE = "tune-run-root-store"

# ---------------------------------------------------------------------------
# Monitor view (Chunk A)
# ---------------------------------------------------------------------------

#: 3-second poll interval driving the Monitor live re-read.
TUNE_STUDY_POLL = "tune-study-poll"

#: The objective figure (running-best line + raw scatter).
TUNE_OBJECTIVE_FIGURE = "tune-objective-figure"

#: The param-importance bar figure.
TUNE_IMPORTANCE_FIGURE = "tune-importance-figure"

#: The winner-stability gap badge.
TUNE_GAP_BADGE = "tune-gap-badge"

#: The trials table (one row per trial).
TUNE_TRIALS_TABLE = "tune-trials-table"

#: A degrade / status note surfaced when the live read falls back to the
#: finished parquet, or when the tune extra is missing.
TUNE_MONITOR_NOTE = "tune-monitor-note"

#: The Pareto card — rendered only for a multi-objective run.
TUNE_PARETO_CARD = "tune-pareto-card"

# ---------------------------------------------------------------------------
# Curate view (Chunk B-ii)
# ---------------------------------------------------------------------------

#: The Curate Image Source store — the absolute directory of plate images the
#: overlay loads ``<Image Source>/<plate_name>`` from. Pre-filled from the bound
#: run's ``run.json`` ``images_dir``; ``None`` until the user picks one.
TUNE_IMAGE_SOURCE_STORE = "tune-image-source-store"

#: The "point me at the plate images" prompt shown when the Image Source is
#: unset (the run dir holds no input images).
TUNE_CURATE_PROMPT = "tune-curate-prompt"

#: The sandbox-bounded Image Source directory-picker modal (reuses the builder
#: ``directory_tree`` folder-only listing).
TUNE_IMAGE_SOURCE_MODAL = "tune-image-source-modal"

#: The modal body that re-renders the directory tree on navigation.
TUNE_IMAGE_SOURCE_MODAL_BODY = "tune-image-source-modal-body"

#: The browse-dir store the modal's tree re-renders from on navigation.
TUNE_IMAGE_SOURCE_BROWSE_DIR = "tune-image-source-browse-dir"

#: The "Image Source" button that opens the picker modal.
TUNE_BTN_PICK_IMAGE_SOURCE = "tune-btn-pick-image-source"

#: The modal's Cancel button.
TUNE_BTN_IMAGE_SOURCE_CANCEL = "tune-btn-image-source-cancel"

#: The modal's "Use this directory" confirm button.
TUNE_BTN_IMAGE_SOURCE_CONFIRM = "tune-btn-image-source-confirm"

#: The selected-Image-Source label shown next to the picker button.
TUNE_IMAGE_SOURCE_LABEL = "tune-image-source-label"

#: The pattern-matching ``type`` for a directory entry in the Image Source tree.
TUNE_DIR_ENTRY_IMAGE_SOURCE = "tune-dir-entry-image-source"

#: A toast surfaced for Curate-view errors (out-of-sandbox refusal, winner-write
#: permission failure).
TUNE_CURATE_TOAST = "tune-curate-toast"

#: The shortlist card container — one clickable card per shortlisted trial.
TUNE_SHORTLIST = "tune-shortlist"

#: The pattern-matching ``type`` for a shortlist card (one per trial).
TUNE_SHORTLIST_CARD = "tune-shortlist-card"

#: The A/B pin store: ``{"a": <trial number | None>, "b": <trial number | None>}``.
TUNE_AB_STORE = "tune-ab-store"

#: The Side-by-side ↔ Difference mode toggle.
TUNE_CURATE_MODE_TOGGLE = "tune-curate-mode-toggle"

#: The Curate mode store (``"side"`` or ``"difference"``).
TUNE_CURATE_MODE_STORE = "tune-curate-mode-store"

#: The plate picker dropdown (which plate to render overlays on).
TUNE_PLATE_PICKER = "tune-plate-picker"

#: The side-by-side A graph (``go.Image``).
TUNE_GRAPH_A = "tune-graph-a"

#: The side-by-side B graph (``go.Image``).
TUNE_GRAPH_B = "tune-graph-b"

#: The difference graph (``go.Image``).
TUNE_GRAPH_DIFF = "tune-graph-diff"

#: The side-by-side container (shown in ``"side"`` mode).
TUNE_SIDE_BY_SIDE = "tune-side-by-side"

#: The difference container (shown in ``"difference"`` mode).
TUNE_DIFFERENCE = "tune-difference"

#: The overlay-readiness poll — swaps the spinner figure for the real overlay
#: once the background render future resolves.
TUNE_OVERLAY_POLL = "tune-overlay-poll"

#: Per-tab session id (``storage_type="session"``) namespacing the pending
#: overlay futures so two browser tabs never share a render.
TUNE_SESSION_ID = "tune-session-id"

#: The winner bar's "Set as winner" button.
TUNE_BTN_SET_WINNER = "tune-btn-set-winner"

#: The winner-status note (which trial is pinned / written).
TUNE_WINNER_NOTE = "tune-winner-note"

# ---------------------------------------------------------------------------
# Launch view (Chunk C-i, Task C1)
# ---------------------------------------------------------------------------

#: The strategy dropdown (grid / random / tpe / cmaes / gp / nsga2).
TUNE_LAUNCH_STRATEGY = "tune-launch-strategy"

#: The trial-budget numeric input (``--n-trials``; blank → omit the flag).
TUNE_LAUNCH_N_TRIALS = "tune-launch-n-trials"

#: The Optuna storage-URL text input (``--storage-url``; blank → omit the flag).
TUNE_LAUNCH_STORAGE_URL = "tune-launch-storage-url"

#: The ``--screen`` two-round-freeze toggle (a checklist with one option).
TUNE_LAUNCH_SCREEN = "tune-launch-screen"

#: The ``--slurm`` distributed-fleet toggle (a checklist with one option).
TUNE_LAUNCH_SLURM = "tune-launch-slurm"

#: A hidden store carrying the bound run's spec / input / output paths, so the
#: clientside command mirror reads them without re-deriving from the layout.
TUNE_LAUNCH_PATHS_STORE = "tune-launch-paths-store"

#: The live command card — a ``<code>`` block showing the rendered invocation
#: the clientside callback keeps in sync with the form.
TUNE_LAUNCH_COMMAND = "tune-launch-command"


__all__ = [
    "SubTabName",
    "SUBTAB_ORDER",
    "SUBTAB_LABELS",
    "subtab_button_id",
    "view_container_id",
    "TUNE_PAGE",
    "TUNE_RUN_HEADER",
    "TUNE_ACTIVE_VIEW_STORE",
    "TUNE_RUN_ROOT_STORE",
    "TUNE_STUDY_POLL",
    "TUNE_OBJECTIVE_FIGURE",
    "TUNE_IMPORTANCE_FIGURE",
    "TUNE_GAP_BADGE",
    "TUNE_TRIALS_TABLE",
    "TUNE_MONITOR_NOTE",
    "TUNE_PARETO_CARD",
    "TUNE_IMAGE_SOURCE_STORE",
    "TUNE_CURATE_PROMPT",
    "TUNE_IMAGE_SOURCE_MODAL",
    "TUNE_IMAGE_SOURCE_MODAL_BODY",
    "TUNE_IMAGE_SOURCE_BROWSE_DIR",
    "TUNE_BTN_PICK_IMAGE_SOURCE",
    "TUNE_BTN_IMAGE_SOURCE_CANCEL",
    "TUNE_BTN_IMAGE_SOURCE_CONFIRM",
    "TUNE_IMAGE_SOURCE_LABEL",
    "TUNE_DIR_ENTRY_IMAGE_SOURCE",
    "TUNE_CURATE_TOAST",
    "TUNE_SHORTLIST",
    "TUNE_SHORTLIST_CARD",
    "TUNE_AB_STORE",
    "TUNE_CURATE_MODE_TOGGLE",
    "TUNE_CURATE_MODE_STORE",
    "TUNE_PLATE_PICKER",
    "TUNE_GRAPH_A",
    "TUNE_GRAPH_B",
    "TUNE_GRAPH_DIFF",
    "TUNE_SIDE_BY_SIDE",
    "TUNE_DIFFERENCE",
    "TUNE_OVERLAY_POLL",
    "TUNE_SESSION_ID",
    "TUNE_BTN_SET_WINNER",
    "TUNE_WINNER_NOTE",
    "TUNE_LAUNCH_STRATEGY",
    "TUNE_LAUNCH_N_TRIALS",
    "TUNE_LAUNCH_STORAGE_URL",
    "TUNE_LAUNCH_SCREEN",
    "TUNE_LAUNCH_SLURM",
    "TUNE_LAUNCH_PATHS_STORE",
    "TUNE_LAUNCH_COMMAND",
]

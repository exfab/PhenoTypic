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


def subtab_button_class(name: SubTabName, active: "SubTabName | None") -> str:
    """The CSS class string for a sub-tab button (active gets the highlight).

    Single source of truth for both the initial render (``_layout``, where
    ``active`` may be ``None``) and the switch callback (``_callbacks``), so the
    two can never disagree on which class marks the active sub-tab.
    """
    classes = ["tune-subtab"]
    if name == active:
        classes.append("tune-subtab-active")
    return " ".join(classes)


def view_container_class(name: SubTabName, active: "SubTabName | None") -> str:
    """The CSS class string for a view container (non-active is hidden).

    Single source of truth for both the initial render (``_layout``) and the
    switch callback (``_callbacks``); a container that is not the active view
    carries the ``tune-view-hidden`` class.
    """
    classes = ["tune-view"]
    if name != active:
        classes.append("tune-view-hidden")
    return " ".join(classes)


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

#: Top-level container for the tune page body.
TUNE_PAGE = "tune-page"

#: Run-picker header (run path display / picker placeholder).
TUNE_RUN_HEADER = "tune-run-header"

#: The active-view store: which sub-tab's container is currently shown.
TUNE_ACTIVE_VIEW_STORE = "tune-active-view-store"

#: The active top-level tune destination: Setup / Run / Monitor.
TUNE_ACTIVE_DESTINATION_STORE = "tune-active-destination-store"

#: Hamburger/destination row container.
TUNE_DESTINATION_DRAWER = "tune-destination-drawer"

#: The run-root store: a JSON-serialisable descriptor of the bound
#: :class:`~phenotypic.gui.tune.TuneRunRoot` the poll callback re-reads from.
TUNE_RUN_ROOT_STORE = "tune-run-root-store"

# ---------------------------------------------------------------------------
# Run picker (Chunk C — bind a tune output directory at runtime)
# ---------------------------------------------------------------------------

#: The page-body container the bind callback swaps from the pick-a-run prompt to
#: the loaded four-view layout. Held at the page root so the store-write that
#: binds a run can re-render the whole body without disturbing the picker chrome
#: or the run-root store above it.
TUNE_PAGE_BODY = "tune-page-body"

# ---------------------------------------------------------------------------
# Setup destination
# ---------------------------------------------------------------------------

TUNE_SETUP_PIPELINE_STORE = "tune-setup-pipeline-store"
TUNE_SETUP_METADATA_STORE = "tune-setup-metadata-store"
TUNE_SETUP_PIPELINE_PICKER_STORE = "tune-setup-pipeline-picker-store"
TUNE_SETUP_METADATA_PICKER_STORE = "tune-setup-metadata-picker-store"
TUNE_SETUP_AUTHORED_SPEC_STORE = "tune-setup-authored-spec-store"
TUNE_SETUP_SIGNATURE_STORE = "tune-setup-signature-store"
TUNE_SETUP_PIPELINE_INPUT = "tune-setup-pipeline-input"
TUNE_SETUP_METADATA_INPUT = "tune-setup-metadata-input"
TUNE_SETUP_PIPELINE_SOURCE = "tune-setup-pipeline-source"
TUNE_SETUP_METADATA_SOURCE = "tune-setup-metadata-source"
TUNE_SETUP_PICK_PIPELINE = "tune-setup-pick-pipeline"
TUNE_SETUP_PICK_METADATA = "tune-setup-pick-metadata"
TUNE_SETUP_PIPELINE_MODAL = "tune-setup-pipeline-modal"
TUNE_SETUP_METADATA_MODAL = "tune-setup-metadata-modal"
TUNE_SETUP_PIPELINE_MODAL_BODY = "tune-setup-pipeline-modal-body"
TUNE_SETUP_METADATA_MODAL_BODY = "tune-setup-metadata-modal-body"
TUNE_SETUP_PIPELINE_BROWSE_DIR = "tune-setup-pipeline-browse-dir"
TUNE_SETUP_METADATA_BROWSE_DIR = "tune-setup-metadata-browse-dir"
TUNE_SETUP_PIPELINE_CANCEL = "tune-setup-pipeline-cancel"
TUNE_SETUP_METADATA_CANCEL = "tune-setup-metadata-cancel"
TUNE_SETUP_PIPELINE_ENTRY = "tune-setup-pipeline-entry"
TUNE_SETUP_METADATA_ENTRY = "tune-setup-metadata-entry"
TUNE_SETUP_GATE = "tune-setup-gate"
TUNE_SETUP_SEARCH_SPACE = "tune-setup-search-space"
TUNE_SETUP_SCORER = "tune-setup-scorer"
TUNE_SETUP_REPLACE_SCORER = "tune-setup-replace-scorer"
TUNE_SETUP_SPACE_KNOB_ROW = "tune-setup-space-knob-row"
TUNE_SETUP_SPACE_LOW = "tune-setup-space-low"
TUNE_SETUP_SPACE_HIGH = "tune-setup-space-high"
TUNE_SETUP_SPACE_LOG = "tune-setup-space-log"
TUNE_SETUP_SPACE_CHOICES = "tune-setup-space-choices"
TUNE_SETUP_SPACE_TUNABLE = "tune-setup-space-tunable"
TUNE_SETUP_FOOTER = "tune-setup-footer"
TUNE_SETUP_CONTINUE = "tune-setup-continue"

# ---------------------------------------------------------------------------
# Run destination
# ---------------------------------------------------------------------------

TUNE_RUN_IMAGES_OVERRIDE = "tune-run-images-override"
TUNE_RUN_OUTPUT_DIR = "tune-run-output-dir"
TUNE_RUN_STRATEGY = "tune-run-strategy"
TUNE_RUN_N_TRIALS = "tune-run-n-trials"
TUNE_RUN_STORAGE_URL = "tune-run-storage-url"
TUNE_RUN_STORAGE_MODE = "tune-run-storage-mode"
TUNE_RUN_STORAGE_ENV = "tune-run-storage-env"
TUNE_RUN_N_WORKERS = "tune-run-n-workers"
TUNE_RUN_SLURM_PARTITION = "tune-run-slurm-partition"
TUNE_RUN_SLURM_MEM = "tune-run-slurm-mem"
TUNE_RUN_SLURM_TIME = "tune-run-slurm-time"
TUNE_RUN_HELD_OUT_FRACTION = "tune-run-held-out-fraction"
TUNE_RUN_CV_GROUP = "tune-run-cv-group"
TUNE_RUN_MODE = "tune-run-mode"
TUNE_RUN_SCREEN = "tune-run-screen"
TUNE_RUN_COMMAND = "tune-run-command"
TUNE_RUN_PORTABLE_COMMAND = "tune-run-portable-command"
TUNE_RUN_COPY = "tune-run-copy"
TUNE_RUN_PREFLIGHT = "tune-run-preflight"
TUNE_RUN_DEPLOY = "tune-run-deploy"
TUNE_RUN_STATUS = "tune-run-status"
TUNE_RUN_ACTIVE_RECORD_STORE = "tune-run-active-record-store"

# ---------------------------------------------------------------------------
# Monitor destination extensions
# ---------------------------------------------------------------------------

TUNE_MONITOR_ACTIVE_RUN_STORE = "tune-monitor-active-run-store"
TUNE_MONITOR_SWITCHER = "tune-monitor-switcher"
TUNE_MONITOR_RUN_SWITCH = "tune-monitor-run-switch"
TUNE_MONITOR_LOCAL_LOG = "tune-monitor-local-log"
TUNE_MONITOR_SLURM_FLEET = "tune-monitor-slurm-fleet"
TUNE_MONITOR_CANCEL_CONFIRM = "tune-monitor-cancel-confirm"
TUNE_MONITOR_CANCEL = "tune-monitor-cancel"
TUNE_MONITOR_CANCEL_NOTE = "tune-monitor-cancel-note"
TUNE_MONITOR_EXPORT = "tune-monitor-export"
TUNE_MONITOR_EXPORT_NOTE = "tune-monitor-export-note"
TUNE_MONITOR_RESULT_ZONE = "tune-monitor-result-zone"

#: The "Bind run" / "Browse..." run-picker button that opens the sandbox-bounded
#: run-directory picker modal.
TUNE_BTN_PICK_RUN = "tune-btn-pick-run"

#: The selected-run-directory label shown next to the picker button (the bound
#: run path, or a "no run bound" placeholder).
TUNE_RUN_PICKER_LABEL = "tune-run-picker-label"

#: A status / error note next to the run picker: surfaces a "not a tune output"
#: rejection (a clear message, never a 500) when discovery fails.
TUNE_RUN_PICKER_NOTE = "tune-run-picker-note"

#: The sandbox-bounded run-directory picker modal (reuses the builder
#: ``directory_tree`` folder-only listing, like the Curate Image Source picker).
TUNE_RUN_PICKER_MODAL = "tune-run-picker-modal"

#: The run-picker modal body that re-renders the directory tree on navigation.
TUNE_RUN_PICKER_MODAL_BODY = "tune-run-picker-modal-body"

#: The browse-dir store the run-picker modal's tree re-renders from on navigation.
TUNE_RUN_PICKER_BROWSE_DIR = "tune-run-picker-browse-dir"

#: The run-picker modal's Cancel button.
TUNE_BTN_RUN_PICKER_CANCEL = "tune-btn-run-picker-cancel"

#: The run-picker modal's "Bind this run" confirm button.
TUNE_BTN_RUN_PICKER_CONFIRM = "tune-btn-run-picker-confirm"

#: The pattern-matching ``type`` for a directory entry in the run-picker tree.
TUNE_DIR_ENTRY_RUN = "tune-dir-entry-run"

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

# ---------------------------------------------------------------------------
# Space view (Chunk C-i, Task C2)
# ---------------------------------------------------------------------------

#: A per-knob editor row (pattern-matching ``{"type": …, "key": "0.sigma"}``).
TUNE_SPACE_KNOB_ROW = "tune-space-knob-row"

#: A range knob's low-bound numeric input (pattern-matching, keyed by knob key).
TUNE_SPACE_LOW = "tune-space-low"

#: A range knob's high-bound numeric input (pattern-matching, keyed by knob key).
TUNE_SPACE_HIGH = "tune-space-high"

#: A range knob's log-scale switch (pattern-matching, keyed by knob key).
TUNE_SPACE_LOG = "tune-space-log"

#: A categorical knob's choice checklist (pattern-matching, keyed by knob key).
TUNE_SPACE_CHOICES = "tune-space-choices"

#: A per-knob ``tunable`` on/off switch (pattern-matching, keyed by knob key).
TUNE_SPACE_TUNABLE = "tune-space-tunable"

#: The "Export tuning_spec.json" button.
TUNE_BTN_SPACE_EXPORT = "tune-btn-space-export"

#: The Space-view status note (export result / "review in Launch" hint).
TUNE_SPACE_NOTE = "tune-space-note"


__all__ = [
    "SubTabName",
    "SUBTAB_ORDER",
    "SUBTAB_LABELS",
    "subtab_button_id",
    "view_container_id",
    "subtab_button_class",
    "view_container_class",
    "TUNE_PAGE",
    "TUNE_RUN_HEADER",
    "TUNE_ACTIVE_VIEW_STORE",
    "TUNE_ACTIVE_DESTINATION_STORE",
    "TUNE_DESTINATION_DRAWER",
    "TUNE_RUN_ROOT_STORE",
    "TUNE_PAGE_BODY",
    "TUNE_SETUP_PIPELINE_STORE",
    "TUNE_SETUP_METADATA_STORE",
    "TUNE_SETUP_PIPELINE_PICKER_STORE",
    "TUNE_SETUP_METADATA_PICKER_STORE",
    "TUNE_SETUP_AUTHORED_SPEC_STORE",
    "TUNE_SETUP_SIGNATURE_STORE",
    "TUNE_SETUP_PIPELINE_INPUT",
    "TUNE_SETUP_METADATA_INPUT",
    "TUNE_SETUP_PIPELINE_SOURCE",
    "TUNE_SETUP_METADATA_SOURCE",
    "TUNE_SETUP_PICK_PIPELINE",
    "TUNE_SETUP_PICK_METADATA",
    "TUNE_SETUP_PIPELINE_MODAL",
    "TUNE_SETUP_METADATA_MODAL",
    "TUNE_SETUP_PIPELINE_MODAL_BODY",
    "TUNE_SETUP_METADATA_MODAL_BODY",
    "TUNE_SETUP_PIPELINE_BROWSE_DIR",
    "TUNE_SETUP_METADATA_BROWSE_DIR",
    "TUNE_SETUP_PIPELINE_CANCEL",
    "TUNE_SETUP_METADATA_CANCEL",
    "TUNE_SETUP_PIPELINE_ENTRY",
    "TUNE_SETUP_METADATA_ENTRY",
    "TUNE_SETUP_GATE",
    "TUNE_SETUP_SEARCH_SPACE",
    "TUNE_SETUP_SCORER",
    "TUNE_SETUP_REPLACE_SCORER",
    "TUNE_SETUP_SPACE_KNOB_ROW",
    "TUNE_SETUP_SPACE_LOW",
    "TUNE_SETUP_SPACE_HIGH",
    "TUNE_SETUP_SPACE_LOG",
    "TUNE_SETUP_SPACE_CHOICES",
    "TUNE_SETUP_SPACE_TUNABLE",
    "TUNE_SETUP_FOOTER",
    "TUNE_SETUP_CONTINUE",
    "TUNE_RUN_IMAGES_OVERRIDE",
    "TUNE_RUN_OUTPUT_DIR",
    "TUNE_RUN_STRATEGY",
    "TUNE_RUN_N_TRIALS",
    "TUNE_RUN_STORAGE_URL",
    "TUNE_RUN_STORAGE_MODE",
    "TUNE_RUN_STORAGE_ENV",
    "TUNE_RUN_N_WORKERS",
    "TUNE_RUN_SLURM_PARTITION",
    "TUNE_RUN_SLURM_MEM",
    "TUNE_RUN_SLURM_TIME",
    "TUNE_RUN_HELD_OUT_FRACTION",
    "TUNE_RUN_CV_GROUP",
    "TUNE_RUN_MODE",
    "TUNE_RUN_SCREEN",
    "TUNE_RUN_COMMAND",
    "TUNE_RUN_PORTABLE_COMMAND",
    "TUNE_RUN_COPY",
    "TUNE_RUN_PREFLIGHT",
    "TUNE_RUN_DEPLOY",
    "TUNE_RUN_STATUS",
    "TUNE_RUN_ACTIVE_RECORD_STORE",
    "TUNE_MONITOR_ACTIVE_RUN_STORE",
    "TUNE_MONITOR_SWITCHER",
    "TUNE_MONITOR_RUN_SWITCH",
    "TUNE_MONITOR_LOCAL_LOG",
    "TUNE_MONITOR_SLURM_FLEET",
    "TUNE_MONITOR_CANCEL_CONFIRM",
    "TUNE_MONITOR_CANCEL",
    "TUNE_MONITOR_CANCEL_NOTE",
    "TUNE_MONITOR_EXPORT",
    "TUNE_MONITOR_EXPORT_NOTE",
    "TUNE_MONITOR_RESULT_ZONE",
    "TUNE_BTN_PICK_RUN",
    "TUNE_RUN_PICKER_LABEL",
    "TUNE_RUN_PICKER_NOTE",
    "TUNE_RUN_PICKER_MODAL",
    "TUNE_RUN_PICKER_MODAL_BODY",
    "TUNE_RUN_PICKER_BROWSE_DIR",
    "TUNE_BTN_RUN_PICKER_CANCEL",
    "TUNE_BTN_RUN_PICKER_CONFIRM",
    "TUNE_DIR_ENTRY_RUN",
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
    "TUNE_SPACE_KNOB_ROW",
    "TUNE_SPACE_LOW",
    "TUNE_SPACE_HIGH",
    "TUNE_SPACE_LOG",
    "TUNE_SPACE_CHOICES",
    "TUNE_SPACE_TUNABLE",
    "TUNE_BTN_SPACE_EXPORT",
    "TUNE_SPACE_NOTE",
]

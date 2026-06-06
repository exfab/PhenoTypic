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
]

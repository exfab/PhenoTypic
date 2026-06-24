"""Component IDs for the PhenoTypic analysis sub-app.

Single source of truth for all Dash component IDs used by the analysis
sub-app — layout, callbacks, and Playwright tests import from here so
renames flow everywhere. IDs are unprefixed kebab-case in line with the
builder/results-viewer convention; the shell uses ``"shell-"`` and
mount middleware keeps the namespaces independent.
"""
from __future__ import annotations

from typing import Literal

#: Section-stack kinds used by :func:`section_remove_button_id` and the
#: layout/callback dispatch. Aliased so layout, callbacks, and pattern-
#: matching ID builders all share the same string-union type — typos like
#: ``"filters"`` get caught by mypy at every call site.
SectionKind = Literal["post", "filter", "edge"]

#: Wider set of analyzer-creation kinds used by ``_instantiate``. Includes
#: ``"model"`` because the model section is exclusive (not a stack) but
#: still constructed from the same dropdown-driven flow.
InstantiationKind = Literal["post", "filter", "model", "edge"]

#: Section kinds that carry a plotting-preview affordance. Post sections
#: use the table-preview path instead, so they are intentionally absent;
#: ``"model"`` is included because the model card hosts a plot preview
#: even though it is not a stack (it always uses ``index`` 0).
PlotSectionKind = Literal["filter", "model", "edge"]

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

#: Top-level container for the analysis page body.
ANALYSIS_PAGE = "analysis-page"

#: Output-root header (path display + reload button).
ANALYSIS_OUTPUT_HEADER = "analysis-output-header"

#: Pipeline summary chip (e.g. "3 ops · 1 post · 2 filters · LogGrowthModel").
ANALYSIS_PIPELINE_HEADER = "analysis-pipeline-header"

#: Banner shown above the post stack, reminding users that post edits
#: require a CLI re-run to land in ``measurements.parquet`` (the
#: post-applied mirror; ``master_measurements.parquet`` stays clean).
ANALYSIS_RECOMPILE_BANNER = "analysis-recompile-banner"

#: Banner shown when the on-disk ``pipeline.json`` mtime no longer
#: matches what the session loaded — a CLI re-run happened under us.
ANALYSIS_STALE_BANNER = "analysis-stale-banner"

#: Banner shown when one or more analyzer entries in ``pipeline.json``
#: reference a class that can no longer be resolved (rename or removal).
#: Renders the list of missing classes plus the path to ``pipeline.json``
#: so the user can manually re-add a replacement. Hidden when no
#: warnings were collected during load.
ANALYSIS_LOAD_WARNINGS_BANNER = "analysis-load-warnings-banner"

# ---------------------------------------------------------------------------
# Section stacks
# ---------------------------------------------------------------------------

#: Container holding the post-section accordion stack.
ANALYSIS_POST_STACK = "analysis-post-stack"

#: Container holding the filter-section accordion stack.
ANALYSIS_FILTER_STACK = "analysis-filter-stack"

#: Container holding the (single) model section.
ANALYSIS_MODEL_SECTION = "analysis-model-section"

#: Dropdown to add a new post operation to the chain.
ANALYSIS_POST_ADD_DROPDOWN = "analysis-post-add-dropdown"

#: Dropdown to add a new filter to the chain.
ANALYSIS_FILTER_ADD_DROPDOWN = "analysis-filter-add-dropdown"

#: Container holding the edge-correction section accordion stack.
ANALYSIS_EDGE_STACK = "analysis-edge-stack"

#: Dropdown to add a new edge corrector to the chain.
ANALYSIS_EDGE_ADD_DROPDOWN = "analysis-edge-add-dropdown"

#: Dropdown to choose / replace the endpoint model.
ANALYSIS_MODEL_DROPDOWN = "analysis-model-dropdown"

# ---------------------------------------------------------------------------
# Run console (sticky footer)
# ---------------------------------------------------------------------------

#: "Run analysis" button — disabled when no model is configured.
ANALYSIS_RUN_BUTTON = "analysis-run-button"

#: Status line beneath the run button (e.g. "Wrote 96 rows · 1.4s").
ANALYSIS_RUN_STATUS = "analysis-run-status"

#: Spinner shown while ``pipeline.analyze`` is running.
ANALYSIS_RUN_SPINNER = "analysis-run-spinner"

# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

#: ``dcc.Store`` echoing the current pipeline JSON so callbacks can avoid
#: round-tripping disk on every dropdown change.
ANALYSIS_PIPELINE_STORE = "analysis-pipeline-store"

#: ``dcc.Store(storage_type="session")`` holding per-section plotting
#: preferences (``figsize`` / ``collapsed`` / ``cmap`` / ...). Session
#: scoped on purpose — these display tweaks are for live pipeline tuning
#: and must never serialize into ``pipeline.json``. Keyed by
#: ``f"{kind}-{index}-{name}"``; survives browser refresh, dies on tab
#: close.
ANALYSIS_PLOT_PREFS_STORE = "analysis-plot-prefs-store"

# ---------------------------------------------------------------------------
# Empty-state hand-off (mirrors the results-viewer pattern). The bind
# endpoint is shared with the viewer; clicking either tool's "Open"
# button releases both ToolSessions and rebuilds them against the new
# ``viewer_state["output_root"]``.
# ---------------------------------------------------------------------------

EMPTY_HANDOFF_BANNER = "analysis-empty-handoff-banner"
EMPTY_HANDOFF_LABEL = "analysis-empty-handoff-label"
EMPTY_HANDOFF_OPEN_BUTTON = "analysis-empty-handoff-open-button"
EMPTY_HANDOFF_ERROR = "analysis-empty-handoff-error"


def post_section_id(index: int) -> dict[str, str | int]:
    """Pattern-matching ID for one post section in the stack."""
    return {"type": "analysis-post-section", "index": index}


def filter_section_id(index: int) -> dict[str, str | int]:
    """Pattern-matching ID for one filter section in the stack."""
    return {"type": "analysis-filter-section", "index": index}


def edge_section_id(index: int) -> dict[str, str | int]:
    """Pattern-matching ID for one edge-correction section in the stack."""
    return {"type": "analysis-edge-section", "index": index}


def section_remove_button_id(
    kind: SectionKind, index: int
) -> dict[str, str | int]:
    """Pattern-matching ID for the ``×`` remove button on a section."""
    return {"type": "analysis-section-remove", "kind": kind, "index": index}


def preview_button_id(
    kind: PlotSectionKind, index: int
) -> dict[str, str | int]:
    """Pattern-matching ID for a section's ``Preview`` button."""
    return {"type": "analysis-preview-btn", "kind": kind, "index": index}


def plot_slot_id(kind: PlotSectionKind, index: int) -> dict[str, str | int]:
    """Pattern-matching ID for the (initially empty) plot-output slot."""
    return {"type": "analysis-plot-slot", "kind": kind, "index": index}


def plot_param_id(
    kind: PlotSectionKind, index: int, name: str
) -> dict[str, str | int]:
    """Pattern-matching ID for one plotting-preference widget.

    All plotting widgets share this single id schema so a single
    ``ALL`` pattern matches every one of them. ``tuple``-typed params
    (e.g. ``figsize``) render as two widgets whose ``name`` carries a
    ``"__0"`` / ``"__1"`` axis suffix; see :mod:`._plot_controls`.
    """
    return {
        "type": "analysis-plot-param",
        "kind": kind,
        "index": index,
        "name": name,
    }


__all__ = [
    "SectionKind",
    "InstantiationKind",
    "PlotSectionKind",
    "ANALYSIS_PAGE",
    "ANALYSIS_OUTPUT_HEADER",
    "ANALYSIS_PIPELINE_HEADER",
    "ANALYSIS_RECOMPILE_BANNER",
    "ANALYSIS_STALE_BANNER",
    "ANALYSIS_LOAD_WARNINGS_BANNER",
    "ANALYSIS_POST_STACK",
    "ANALYSIS_FILTER_STACK",
    "ANALYSIS_MODEL_SECTION",
    "ANALYSIS_POST_ADD_DROPDOWN",
    "ANALYSIS_FILTER_ADD_DROPDOWN",
    "ANALYSIS_EDGE_STACK",
    "ANALYSIS_EDGE_ADD_DROPDOWN",
    "ANALYSIS_MODEL_DROPDOWN",
    "ANALYSIS_RUN_BUTTON",
    "ANALYSIS_RUN_STATUS",
    "ANALYSIS_RUN_SPINNER",
    "ANALYSIS_PIPELINE_STORE",
    "ANALYSIS_PLOT_PREFS_STORE",
    "EMPTY_HANDOFF_BANNER",
    "EMPTY_HANDOFF_LABEL",
    "EMPTY_HANDOFF_OPEN_BUTTON",
    "EMPTY_HANDOFF_ERROR",
    "post_section_id",
    "filter_section_id",
    "edge_section_id",
    "section_remove_button_id",
    "preview_button_id",
    "plot_slot_id",
    "plot_param_id",
]

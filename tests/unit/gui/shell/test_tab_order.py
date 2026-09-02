"""Unit-level guard for the top-bar nav model.

The shell's :data:`NAV_MODEL` tuple is the single source of truth for the
unified hub's top-bar navigation. It consolidates the six reachable mounts
into a Home leaf, a Browse leaf, plus two dropdown tab groups:

* **Browse** (leaf) -> source-image viewer
* **Pipeline** -> Builder, Run
* **Results**  -> Viewer, Analysis

Tune is unmounted (docs/superpowers/plans/2026-08-26-gui-simplification-removals,
phase-4-tune-unmount.md) and is therefore not a Pipeline-group member here;
its package stays on disk and importable, just unreachable from the UI.

The order follows the user workflow (browse the source images, then
compose -> run, then inspect -> analyse) and downstream
documentation (FEATURES.md) advertises it. A future refactor that
rotates the tuple, drops a member, or duplicates a tab id would
otherwise only surface in the e2e suite.
"""
from __future__ import annotations

from phenotypic.gui.shell._ids import (
    SHELL_TAB_ANALYSIS,
    SHELL_TAB_BROWSE,
    SHELL_TAB_BUILDER,
    SHELL_TAB_GROUP_PIPELINE,
    SHELL_TAB_GROUP_RESULTS,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_VIEWER,
)
from phenotypic.gui.shell._layout import (
    NAV_MODEL,
    _NavGroup,
    _TAB_HREFS,
    _TAB_LABELS,
)


def _flatten(nav: tuple) -> list[str]:
    """Flatten NAV_MODEL into the ordered list of every member tab id."""
    out: list[str] = []
    for entry in nav:
        if isinstance(entry, _NavGroup):
            out.extend(entry.members)
        else:
            out.append(entry)
    return out


def test_nav_model_structure_matches_workflow() -> None:
    """Home leaf, Browse leaf, then Pipeline (Builder/Run), then
    Results (Viewer/Analysis)."""
    assert NAV_MODEL[0] == SHELL_TAB_HOME

    # Browse is a leaf immediately after Home.
    assert NAV_MODEL[1] == SHELL_TAB_BROWSE

    pipeline = NAV_MODEL[2]
    assert isinstance(pipeline, _NavGroup)
    assert pipeline.label == "Pipeline"
    assert pipeline.group_id == SHELL_TAB_GROUP_PIPELINE
    assert pipeline.members == (
        SHELL_TAB_BUILDER,
        SHELL_TAB_RUN,
    )

    results = NAV_MODEL[3]
    assert isinstance(results, _NavGroup)
    assert results.label == "Results"
    assert results.group_id == SHELL_TAB_GROUP_RESULTS
    assert results.members == (SHELL_TAB_VIEWER, SHELL_TAB_ANALYSIS)

    # Exactly the Home + Browse leaves + two groups, nothing else.
    assert len(NAV_MODEL) == 4


def test_nav_model_covers_every_known_tab() -> None:
    """Flattened nav lists every tab id known to the href / label maps,
    with no extras and no gaps."""
    flat = _flatten(NAV_MODEL)
    assert set(flat) == set(_TAB_HREFS.keys())
    assert set(flat) == set(_TAB_LABELS.keys())


def test_nav_model_has_no_duplicate_tabs() -> None:
    """A duplicated id would render the same mount under two tabs/groups."""
    flat = _flatten(NAV_MODEL)
    assert len(flat) == len(set(flat))


def test_builder_label_is_builder_not_pipelines() -> None:
    """Under the 'Pipeline' group, the Builder item drops the old
    'Pipelines' wording to avoid 'Pipeline > Pipelines' redundancy."""
    assert _TAB_LABELS[SHELL_TAB_BUILDER] == "Builder"

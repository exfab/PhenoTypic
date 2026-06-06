"""Unit-level guard for the top-bar tab display order.

The shell's :data:`TAB_DISPLAY_ORDER` tuple is the single source of truth
for which tabs the unified hub shows and in what order. The order is
chosen to follow the user workflow (compose -> run -> inspect ->
analyse), and downstream documentation (FEATURES.md) advertises that
ordering. A future refactor that accidentally rotates the tuple would
otherwise only surface in the e2e suite.
"""
from __future__ import annotations

from phenotypic.gui.shell._ids import (
    SHELL_TAB_ANALYSIS,
    SHELL_TAB_BUILDER,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_TUNE,
    SHELL_TAB_VIEWER,
)
from phenotypic.gui.shell._layout import TAB_DISPLAY_ORDER, _TAB_HREFS, _TAB_LABELS


def test_tab_display_order_matches_workflow() -> None:
    """Tabs render in workflow order: Home, Builder, Run, Tune, Viewer, Analysis."""
    assert TAB_DISPLAY_ORDER == (
        SHELL_TAB_HOME,
        SHELL_TAB_BUILDER,
        SHELL_TAB_RUN,
        SHELL_TAB_TUNE,
        SHELL_TAB_VIEWER,
        SHELL_TAB_ANALYSIS,
    )


def test_tab_display_order_covers_every_known_tab() -> None:
    """The display tuple lists every tab id known to the href / label maps,
    with no extras and no gaps."""
    assert set(TAB_DISPLAY_ORDER) == set(_TAB_HREFS.keys())
    assert set(TAB_DISPLAY_ORDER) == set(_TAB_LABELS.keys())


def test_tab_display_order_has_no_duplicates() -> None:
    """A duplicated id would render the same tab twice in the nav."""
    assert len(TAB_DISPLAY_ORDER) == len(set(TAB_DISPLAY_ORDER))

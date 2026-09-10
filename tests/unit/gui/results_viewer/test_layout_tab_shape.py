"""Pin the set of tabs the results viewer actually mounts.

This test is edited deliberately as surfaces are removed. Each edit is the
executable statement that a tab came off; a surprise failure here means a tab
moved without a spec change behind it.
"""

from __future__ import annotations

from phenotypic.gui.results_viewer import _ids as ids


def _tab_ids(layout) -> list[str]:
    """Collect ``tab_id`` from the single ``dbc.Tabs`` in a built layout."""
    found: list[str] = []

    def walk(node) -> None:
        children = getattr(node, "children", None)
        if type(node).__name__ == "Tabs":
            for tab in children or []:
                found.append(tab.tab_id)
            return
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(layout)
    return found


def test_results_tabs_expose_exactly_the_mounted_surfaces(built_results_layout):
    assert _tab_ids(built_results_layout) == [
        ids.TAB_PLATE_ID,
        ids.TAB_COLONY_ID,
        ids.TAB_SCATTER_ID,
    ]


def test_scatter_is_mounted_and_the_deprecated_tabs_are_not(
    built_results_layout,
) -> None:
    """Scatter joins Plate and Colony; Heatmap and QC stay unmounted."""
    tab_ids = _tab_ids(built_results_layout)
    assert ids.TAB_SCATTER_ID in tab_ids
    assert ids.TAB_HEATMAP_ID not in tab_ids
    assert ids.TAB_QC_ID not in tab_ids
    assert ids.TAB_ERROR_ID not in tab_ids

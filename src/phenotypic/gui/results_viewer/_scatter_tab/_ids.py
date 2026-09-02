"""Component ids for the Scatter tab.

Kept in the sub-package rather than in ``results_viewer/_ids.py`` for the
same reason the colony view keeps its own: only this tab's layout and
callbacks bind them. The one id that does live in the parent module is
:data:`~phenotypic.gui.results_viewer._ids.TAB_SCATTER_ID`, because the
top-level ``dbc.Tabs`` mounts it.

Naming follows the viewer's kebab-case convention, and every id is
prefixed ``scatter-`` (or ``store-scatter-``) so a Dash duplicate-id error
names the surface that raised it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Figure surface
# ---------------------------------------------------------------------------

#: The ``dcc.Graph`` the section's faceted figure renders into. Its
#: ``clickData`` is the click path's entry point: each point carries a
#: single int32 index into ``OutputRoot.master_df``.
SCATTER_GRAPH = "scatter-graph"

#: Read-only chip naming the section on screen and its position in the
#: section list (``"BY4741  (3 / 23)"``), plus any facet-cap notice.
SCATTER_PAGER_LABEL = "scatter-pager-label"

#: Step to the previous / next section group.
SCATTER_PREV_BTN = "scatter-prev-btn"
SCATTER_NEXT_BTN = "scatter-next-btn"

#: Render every section to a multi-page PDF.
SCATTER_EXPORT_BTN = "scatter-export-btn"

#: ``dcc.Download`` target the export writes through.
SCATTER_DOWNLOAD = "scatter-download"


# ---------------------------------------------------------------------------
# Configuration popover
# ---------------------------------------------------------------------------

#: Toggle in the tab-bar actions strip that opens the config popover.
SCATTER_CONFIG_TOGGLE = "scatter-config-toggle"

#: The popover itself.
SCATTER_CONFIG_POPOVER = "scatter-config-popover"

#: Column whose values become sections -- one on-screen page, one PDF page.
SCATTER_SECTION_COL = "scatter-section-col"

#: Columns whose values become facet rows / columns within a section.
SCATTER_ROW_COL = "scatter-row-col"
SCATTER_COL_COL = "scatter-col-col"

#: Point position. ``SCATTER_X_COL`` additionally offers the derived
#: capture-order frame index (``_facets.COMPUTED_FRAME_INDEX``).
SCATTER_X_COL = "scatter-x-col"
SCATTER_Y_COL = "scatter-y-col"

#: Columns mapped onto marker colour / marker symbol.
SCATTER_HUE_COL = "scatter-hue-col"
SCATTER_SHAPE_COL = "scatter-shape-col"

#: Checklist toggle: draw curation-removed colonies as a grey x series.
SCATTER_SHOW_REMOVED = "scatter-show-removed"


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

#: Zero-based position in the current section list, moved by the pager.
STORE_SCATTER_SECTION_INDEX = "store-scatter-section-index"

#: The ``OutputSnapshotDescriptor`` fingerprint the rendered figure was
#: built from. Carried beside the point index so a click that arrives
#: after a curation write plus re-discover is refused rather than
#: resolved against a frame it no longer indexes.
STORE_SCATTER_FINGERPRINT = "store-scatter-fingerprint"


__all__ = [
    "SCATTER_COL_COL",
    "SCATTER_CONFIG_POPOVER",
    "SCATTER_CONFIG_TOGGLE",
    "SCATTER_DOWNLOAD",
    "SCATTER_EXPORT_BTN",
    "SCATTER_GRAPH",
    "SCATTER_HUE_COL",
    "SCATTER_NEXT_BTN",
    "SCATTER_PAGER_LABEL",
    "SCATTER_PREV_BTN",
    "SCATTER_ROW_COL",
    "SCATTER_SECTION_COL",
    "SCATTER_SHAPE_COL",
    "SCATTER_SHOW_REMOVED",
    "SCATTER_X_COL",
    "SCATTER_Y_COL",
    "STORE_SCATTER_FINGERPRINT",
    "STORE_SCATTER_SECTION_INDEX",
]

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
#: section list, plus any facet-cap notice. Shape: ``"<name>  (i / n)"``.
#: The example is deliberately not drawn from the verification fixture --
#: ``BY4741`` is not one of its strains, and its section count is 22, not
#: the 23 an earlier draft of this line used. A worked example here would
#: be a fixture claim maintained in a docstring, which is how a number
#: outlives the thing it measured.
SCATTER_PAGER_LABEL = "scatter-pager-label"

#: Step to the previous / next section group.
SCATTER_PREV_BTN = "scatter-prev-btn"
SCATTER_NEXT_BTN = "scatter-next-btn"

#: Render every section to a multi-page PDF.
SCATTER_EXPORT_BTN = "scatter-export-btn"

#: ``dcc.Download`` target the export writes through.
SCATTER_DOWNLOAD = "scatter-download"

#: Where the export reports a refusal. kaleido needs Chrome, which is an
#: undeclared prerequisite (spec section 11.2), so the one failure a user
#: will actually hit needs somewhere to say so -- a button that silently
#: does nothing reads as a broken export.
SCATTER_EXPORT_STATUS = "scatter-export-status"


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
# Style steppers (spec section 9's "Sizing" row)
# ---------------------------------------------------------------------------

#: ``type`` of the pattern-matching id every Style stepper button carries:
#: ``{"type": SCATTER_STYLE_STEP, "field": <field>, "dir": -1 | 1}``. One
#: ``ALL``-keyed callback serves all eight, so a ninth field is a row in
#: ``SCATTER_STYLE_FIELDS`` rather than a ninth callback.
#:
#: Note a pattern-matching id is a JSON object, which is not a valid CSS
#: selector -- browser automation that walks the accessibility tree
#: chokes on it. A test drives these by ``field``/``dir`` through the DOM,
#: not by role.
SCATTER_STYLE_STEP = "scatter-style-step"

#: ``type`` of the readout beside each stepper pair:
#: ``{"type": SCATTER_STYLE_READOUT, "field": <field>}``.
SCATTER_STYLE_READOUT = "scatter-style-readout"


# ---------------------------------------------------------------------------
# Export settings (spec section 11)
# ---------------------------------------------------------------------------

#: Page-size preset. ``_layout.PAGE_SIZE_CUSTOM`` reveals the two inch
#: inputs; every other value drives them and disables them.
SCATTER_PAGE_PRESET = "scatter-page-preset"

#: Page width / height in inches. Only editable under the Custom preset.
SCATTER_PAGE_WIDTH = "scatter-page-width"
SCATTER_PAGE_HEIGHT = "scatter-page-height"

#: The row holding the two inch inputs, hidden unless Custom is chosen.
SCATTER_PAGE_CUSTOM_ROW = "scatter-page-custom-row"

#: Which corner the figure's legend sits in, and whether it is collapsed
#: away entirely. Spec section 9's Legend row; both feed one store the
#: figure callback reads.
SCATTER_LEGEND_CORNER = "scatter-legend-corner"
SCATTER_LEGEND_COLLAPSE = "scatter-legend-collapse"


# ---------------------------------------------------------------------------
# Click inspector
# ---------------------------------------------------------------------------

#: Right-docked ``dbc.Offcanvas`` a resolved click opens. Also the pane
#: the splitter resizes, so its id is what the handle's
#: ``data-splitter-target`` names.
SCATTER_INSPECTOR = "scatter-inspector"

#: Line naming the colony on show (``"ds / stem / label 12"``), or the
#: reason a click was refused.
SCATTER_INSPECTOR_TITLE = "scatter-inspector-title"

#: ``html.Img`` fed by the ``scatter-crops`` route. Its ``src`` is
#: rebuilt when the Contours/Raw control moves, so a toggle re-requests
#: rather than re-resolving the click.
SCATTER_INSPECTOR_CROP = "scatter-inspector-crop"

#: Container the clicked colony's measurements render into, grouped by
#: the ``MeasureFeatures`` class that emitted each column.
SCATTER_INSPECTOR_MEASUREMENTS = "scatter-inspector-measurements"

#: Segmented control choosing whether the crop is composited with object
#: boundaries (``?contours=1``) or served raw (``?contours=0``).
SCATTER_CONTOUR_TOGGLE = "scatter-contour-toggle"

#: Drag handle for the inspector's width. Carrying
#: ``data-splitter-target`` is what makes an element a handle to the
#: shared splitter in ``results_viewer.js`` section H; alongside it the
#: handle declares ``-store``, and -- because this pane is right-docked
#: and the controller cannot infer it -- ``-edge``/``-min``/``-max``.
#: This module names no JS behaviour of its own.
SCATTER_INSPECTOR_SPLITTER = "scatter-inspector-splitter"


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

#: The colony a resolved click landed on, as
#: ``{"dataset": str, "stem": str, "label": int}``. Written once per
#: click and read by the crop-URL callback, so moving the Contours/Raw
#: control re-requests the image without re-resolving the click.
STORE_SCATTER_COLONY = "store-scatter-colony"

#: Legend placement: ``{"corner": str, "collapsed": bool}``. One payload
#: rather than two Inputs so the figure callback reads legend state the
#: same way whether it came from the corner control or the collapse
#: switch.
STORE_SCATTER_LEGEND = "store-scatter-legend"

#: Inspector width in px, written by the shared drag-splitter and
#: re-applied to the offcanvas by a Python callback so it survives a
#: re-render.
STORE_SCATTER_INSPECTOR_WIDTH = "store-scatter-inspector-width"

#: Every Style field as ``{field: value}`` -- the five type sizes, marker
#: size, marker opacity and facet height. One payload rather than eight
#: Inputs, for the reason ``STORE_SCATTER_LEGEND`` gives: the figure
#: callback reads styling one way regardless of which stepper moved, and
#: its Input list does not grow a row per control.
STORE_SCATTER_STYLE = "store-scatter-style"

#: Page size as ``{"preset": str, "width_in": float, "height_in": float}``.
#: Kept **out** of :data:`STORE_SCATTER_STYLE` because it changes nothing
#: on screen: it is a ``State`` on the export callback and an ``Input``
#: nowhere, so choosing a page size does not re-render a figure it cannot
#: affect.
STORE_SCATTER_PAGE = "store-scatter-page"


__all__ = [
    "SCATTER_COL_COL",
    "SCATTER_CONFIG_POPOVER",
    "SCATTER_CONFIG_TOGGLE",
    "SCATTER_CONTOUR_TOGGLE",
    "SCATTER_DOWNLOAD",
    "SCATTER_EXPORT_BTN",
    "SCATTER_EXPORT_STATUS",
    "SCATTER_GRAPH",
    "SCATTER_HUE_COL",
    "SCATTER_INSPECTOR",
    "SCATTER_INSPECTOR_CROP",
    "SCATTER_INSPECTOR_MEASUREMENTS",
    "SCATTER_INSPECTOR_SPLITTER",
    "SCATTER_INSPECTOR_TITLE",
    "SCATTER_LEGEND_COLLAPSE",
    "SCATTER_LEGEND_CORNER",
    "SCATTER_NEXT_BTN",
    "SCATTER_PAGER_LABEL",
    "SCATTER_PREV_BTN",
    "SCATTER_ROW_COL",
    "SCATTER_SECTION_COL",
    "SCATTER_SHAPE_COL",
    "SCATTER_PAGE_CUSTOM_ROW",
    "SCATTER_PAGE_HEIGHT",
    "SCATTER_PAGE_PRESET",
    "SCATTER_PAGE_WIDTH",
    "SCATTER_SHOW_REMOVED",
    "SCATTER_STYLE_READOUT",
    "SCATTER_STYLE_STEP",
    "SCATTER_X_COL",
    "SCATTER_Y_COL",
    "STORE_SCATTER_COLONY",
    "STORE_SCATTER_FINGERPRINT",
    "STORE_SCATTER_INSPECTOR_WIDTH",
    "STORE_SCATTER_LEGEND",
    "STORE_SCATTER_PAGE",
    "STORE_SCATTER_SECTION_INDEX",
    "STORE_SCATTER_STYLE",
]

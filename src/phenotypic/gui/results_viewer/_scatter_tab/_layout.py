"""The Scatter tab's component tree.

Builds the tab body: a toolbar (config popover toggle, section pager,
PDF export), the ``dcc.Graph`` one section renders into, and the stores
the callbacks read. The eight controls in the popover are spec section 9's
configuration surface.

Option lists and defaults are resolved **here, at build time**, and this
is a deliberate departure from the Colony view, which recomputes its axis
options in a callback keyed on the filter spec
(``colony_view/_callbacks.py:309``). The two surfaces want different
things from the same helper.

Colony's axes lay out a **grid**: a column narrowed to one value by the
filter makes a degenerate grid, so dropping it from the list is right, and
its callback duly falls back to ``columns[0]`` when the current selection
stops qualifying. Scatter's roles are not layout — a section group, a
facet axis, a hue or a shape with one value is an ordinary, correct
figure. :func:`._facets.plan_facets` treats it as such on purpose: its
``[""]`` fallback is documented there as load-bearing rather than
defensive.

So the Colony fallback, transplanted here, would be a defect rather than
a nicety. Filter down to one strain and the section-group dropdown would
silently jump to some other column and re-facet the whole figure under
the user. Holding the selection still and letting the pager report
``(1 / 1)`` says the same thing without moving anything.

The consequence, stated plainly: **these option lists describe the run,
not the current filter**, so they can offer a column the Colony tab
currently does not. That is the intended reading — "what this run can be
grouped by" — and the roles are labelled differently enough ("Section
group", "Facet rows" against "X axis", "Y axis") that the two lists are
not presented as the same question. What a filter *can* do is empty a
selected column, which is why
``test_scatter_layout.py`` pins that a one-value column still plans and
still renders.

Note the option lists cannot go stale against curation either way:
``column_value_sets`` is a lazy view over the frozen ``master_df``
captured at ``discover()``, and a curation write does not rebind it.

The click inspector's offcanvas is built here (:func:`_build_inspector`)
rather than in :mod:`._inspector`, which stays Dash-free so the index
producer and the click resolver can be unit-tested without a server.
Its width handle names no JavaScript behaviour of its own: it carries the
data attributes the shared splitter dispatches on, so both ids stay
beside the callbacks that bind them. Among them is the edge the handle
sits on, which a right-docked pane must declare — see
:func:`_build_inspector`.

Still not built here, deliberately: the type/marker sizing steppers.
Nothing reads them, and chrome nothing reads is chrome nobody can trust.
"""

from __future__ import annotations

from typing import Any, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._config import SECTION_GROUP_CAP, splitter_attrs
from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
)
from phenotypic.gui.results_viewer._filtered_state import KEY_OBJECT_LABEL
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    _MEASUREMENT_PREFIXES,
    selectable_axis_columns,
)
from phenotypic.schema import CULTURE
from phenotypic.sdk_ import is_metadata_header

from . import _ids as ids
from ._facets import COMPUTED_FRAME_INDEX

#: Cardinality ceilings per role, from spec section 9. A facet axis is
#: capped well below the section-group cap because rows and columns
#: multiply: `plan_facets` bounds their PRODUCT at `SCATTER_FACET_CAP`,
#: so offering a 60-value column as a facet axis would only ever produce
#: a truncated grid. Hue and shape are capped tighter still, by how many
#: series a reader can actually tell apart -- the palette carries six
#: colours and the symbol set five shapes.
_FACET_AXIS_CAP = 12
_HUE_CAP = 8
_SHAPE_CAP = 6

#: Cardinality ceiling for "is this a sensible default section group?".
#: Deliberately below :data:`SECTION_GROUP_CAP`: a column may be
#: selectable at 60 sections without being a good thing to open on.
_DEFAULT_SECTION_CAP = 50

#: Label for the derived capture-order X axis (spec section 10).
_FRAME_INDEX_LABEL = "frame index (capture order)"

#: The four legend corners, in the order the control offers them. The
#: values are the store payload's ``corner`` and the keys of
#: ``_callbacks._LEGEND_ANCHORS``; a value absent from that table falls
#: back to the default rather than raising.
LEGEND_CORNERS: tuple[tuple[str, str], ...] = (
    ("Top left", "top-left"),
    ("Top right", "top-right"),
    ("Bottom left", "bottom-left"),
    ("Bottom right", "bottom-right"),
)

#: Spec section 9's default: bottom-right, expanded.
LEGEND_CORNER_DEFAULT: str = "bottom-right"

#: Starting width of the click inspector, in px.
_INSPECTOR_WIDTH_DEFAULT = 360

#: Drag bounds for the inspector, passed to the shared splitter as its
#: own rather than inherited. The controller's built-in ``[140, 380]``
#: brackets the QC worklist's 180 px default; against this pane's 360 it
#: leaves 20 px of headroom, so "drag it wider" is a control that does
#: nothing a user would notice. These bracket 360 the way the built-in
#: pair brackets 180.
#:
#: The floor is measured. Its widest measurement row -- a
#: ``MeasureColor`` label/value pair, the longest label being
#: ``ColorLab_DeltaE2000MedianFromMedoid`` -- needs **287 px** of content
#: under Bootstrap's fallback font stack, and ``.offcanvas-end`` takes
#: **33 px** out of a declared width before the content box: 16 px of
#: ``.offcanvas-body`` padding each side plus a 1 px ``border-left``,
#: under a global ``box-sizing: border-box``. 287 + 33 = 320. Both terms
#: are re-derived in
#: ``test_the_inspector_handle_declares_the_edge_it_sits_on``, so this
#: constant cannot be edited without the arithmetic moving with it.
#:
#: Three corrections a review made to a first pass at this number, each
#: of which changed it or what it claims:
#:
#: * **The border was missing.** 287 + 32 = 319 is one px short, and the
#:   earlier chain (278 + 32 = 310) had the same hole.
#: * **The font matters, and 278 was the narrower case.** That figure was
#:   measured with Inter/JetBrains Mono loaded from a CDN. The documented
#:   deployment is an SSH tunnel off a cluster, where the fallback stack
#:   is what renders and the same row measures 287.
#: * **"Clipped" was the wrong mechanism.** ``.offcanvas-body`` resolves
#:   ``overflow-x`` to ``auto``, so below this floor the row *scrolls*
#:   rather than losing a digit. The floor is where the values fit
#:   without scrolling, not where they stop existing.
#:
#: What it therefore does **not** cover: a platform with classic
#: scrollbars (Chrome on Linux/Windows) takes a further ~15 px out of the
#: content box, and this was measured only on macOS overlay scrollbars,
#: where the gutter is free. Below the floor there, the row scrolls -- as
#: it does for a run whose measurers emit longer headers than this one's.
#: This is a comfort threshold for the measured configuration, not a
#: promise about every schema on every platform.
#:
#: The ceiling is chosen, not measured. It leaves exactly half of a
#: 1440 px window unoccluded -- the offcanvas is ``position: fixed`` with
#: ``backdrop=False``, so it covers the figure rather than reflowing it.
_INSPECTOR_WIDTH_MIN = 320
_INSPECTOR_WIDTH_MAX = 720

#: The metadata column the derived frame index stands in for when the run
#: does carry it. Asked of the schema rather than spelled, so it cannot
#: drift from the header the CLI actually writes.
_FRAME_INDEX_COL = str(CULTURE.FRAME_INDEX)


def _axis_options(output_root: OutputRoot, cap: int) -> list[str]:
    """Grouping columns selectable at a given cardinality ceiling.

    Args:
        output_root: Validated handle on the CLI output directory.
        cap: Maximum distinct values a column may carry.

    Returns:
        Column names, metadata first then ``Grid_*`` then the rest --
        :func:`selectable_axis_columns`'s bucketed order.
    """
    return selectable_axis_columns(
        output_root.master_df, output_root.column_value_sets, cap
    )


def _numeric_columns(output_root: OutputRoot) -> list[str]:
    """Columns that can carry a point's position, in frame order.

    Numeric-ness is asked of :meth:`OutputRoot.is_numeric_column`, the
    same predicate the filter sidebar's Range gate uses, so a
    numeric-valued *string* metadata column such as ``Metadata_Time``
    stays offerable.

    The non-empty value-set guard covers the one case that predicate
    answers on dtype alone: a column **typed** numeric that carries no
    values at all -- a metadata field the run's CSV declares and never
    populates. ``is_numeric_column`` returns True for it on
    ``schema[column].is_numeric()`` without ever consulting the values,
    so without the guard it would be offered as an axis with nothing on
    it. An all-null column of dtype ``Null`` is excluded by the predicate
    itself; this guard is not what catches that one.

    Args:
        output_root: Validated handle on the CLI output directory.

    Returns:
        Offerable column names. Empty is a normal answer for a run that
        measured nothing.
    """
    return [
        column
        for column in output_root.master_df.columns
        if column != KEY_OBJECT_LABEL
        and output_root.column_value_sets.get(column)
        and output_root.is_numeric_column(column)
    ]


def _default_section_col(output_root: OutputRoot) -> str | None:
    """First metadata column with a tractable number of distinct values.

    :func:`selectable_axis_columns` sorts metadata columns first, so the
    head of its list is the metadata column to open on when there is one.

    Args:
        output_root: Validated handle on the CLI output directory.

    Returns:
        A column name, or ``None`` when the run carries no suitable
        metadata -- in which case the tab opens on a single section.
    """
    for column in _axis_options(output_root, _DEFAULT_SECTION_CAP):
        if is_metadata_header(column):
            return column
    return None


def _default_y_col(numeric: list[str]) -> str | None:
    """First numeric measurement, falling back to any numeric column.

    Args:
        numeric: Output of :func:`_numeric_columns`.

    Returns:
        A column name, or ``None`` when the run has no numeric column.
    """
    for column in numeric:
        if any(column.startswith(prefix) for prefix in _MEASUREMENT_PREFIXES):
            return column
    return numeric[0] if numeric else None


def _default_x_col(numeric: list[str]) -> str:
    """The run's own frame index when it has one, else the derived one.

    Args:
        numeric: Output of :func:`_numeric_columns`.

    Returns:
        A column name; never ``None``, because the derived capture-order
        index is always offerable.
    """
    return _FRAME_INDEX_COL if _FRAME_INDEX_COL in numeric else COMPUTED_FRAME_INDEX


def _labelled_dropdown(
    label: str,
    component_id: str,
    options: list[str],
    value: str | None,
    *,
    placeholder: str,
    clearable: bool = True,
    extra: list[dict[str, str]] | None = None,
) -> Component:
    """One captioned row of the configuration popover.

    Args:
        label: Caption shown above the control.
        component_id: The dropdown's Dash id.
        options: Column names to offer.
        value: Initial selection, or ``None``.
        placeholder: Empty-state text.
        clearable: Whether "no column" is reachable again after a choice.
        extra: Options prepended ahead of the columns (the derived X
            axis is the only current user).

    Returns:
        A :class:`dash.html.Div` wrapping caption and dropdown.
    """
    entries = list(extra or [])
    entries += [{"label": column, "value": column} for column in options]
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "color": COLOR_NAVY,
                    "fontWeight": 500,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginBottom": "0.15rem",
                },
            ),
            dcc.Dropdown(
                id=component_id,
                options=entries,
                value=value,
                placeholder=placeholder,
                clearable=clearable,
                style={"minWidth": "240px"},
            ),
        ],
        style={"marginBottom": "0.6rem"},
    )


def _build_config_popover(output_root: OutputRoot) -> Component:
    """Build the configuration popover: the eight role bindings.

    Args:
        output_root: Validated handle on the CLI output directory.

    Returns:
        A :class:`dbc.Popover` anchored to the toolbar's config toggle.
    """
    numeric = _numeric_columns(output_root)
    body = [
        _labelled_dropdown(
            "Section group",
            ids.SCATTER_SECTION_COL,
            _axis_options(output_root, SECTION_GROUP_CAP),
            _default_section_col(output_root),
            placeholder="One section per value…",
        ),
        _labelled_dropdown(
            "Facet rows",
            ids.SCATTER_ROW_COL,
            _axis_options(output_root, _FACET_AXIS_CAP),
            None,
            placeholder="Single row",
        ),
        _labelled_dropdown(
            "Facet columns",
            ids.SCATTER_COL_COL,
            _axis_options(output_root, _FACET_AXIS_CAP),
            None,
            placeholder="Single column",
        ),
        _labelled_dropdown(
            "X axis",
            ids.SCATTER_X_COL,
            numeric,
            _default_x_col(numeric),
            placeholder="X axis…",
            clearable=False,
            extra=[{"label": _FRAME_INDEX_LABEL, "value": COMPUTED_FRAME_INDEX}],
        ),
        _labelled_dropdown(
            "Y axis",
            ids.SCATTER_Y_COL,
            numeric,
            _default_y_col(numeric),
            placeholder="Y axis…",
            clearable=False,
        ),
        _labelled_dropdown(
            "Colour",
            ids.SCATTER_HUE_COL,
            _axis_options(output_root, _HUE_CAP),
            None,
            placeholder="Single series",
        ),
        _labelled_dropdown(
            "Marker shape",
            ids.SCATTER_SHAPE_COL,
            _axis_options(output_root, _SHAPE_CAP),
            None,
            placeholder="Circles",
        ),
        _labelled_dropdown(
            "Legend corner",
            ids.SCATTER_LEGEND_CORNER,
            [],
            LEGEND_CORNER_DEFAULT,
            placeholder="Legend corner…",
            clearable=False,
            extra=[
                {"label": label, "value": value}
                for label, value in LEGEND_CORNERS
            ],
        ),
        dbc.Switch(
            id=ids.SCATTER_LEGEND_COLLAPSE,
            label="Collapse the legend",
            value=False,
            style={"fontSize": FONT_SIZE_CAPTION, "color": COLOR_NAVY},
        ),
        dbc.Switch(
            id=ids.SCATTER_SHOW_REMOVED,
            label="Show removed colonies as grey ×",
            value=True,
            style={"fontSize": FONT_SIZE_CAPTION, "color": COLOR_NAVY},
        ),
    ]
    return dbc.Popover(
        dbc.PopoverBody(body, style={"maxHeight": "70vh", "overflowY": "auto"}),
        id=ids.SCATTER_CONFIG_POPOVER,
        target=ids.SCATTER_CONFIG_TOGGLE,
        trigger="legacy",
        placement="bottom-start",
        style={"maxWidth": "320px"},
    )


def _pager_button(glyph: str, component_id: str, label: str) -> Component:
    """Build one icon-only pager arrow.

    `html.Button` with the Bootstrap classes, not `dbc.Button`: dbc
    components declare a closed `_prop_names` and reject `aria-label`
    outright. Both pager arrows are icon-only, so the aria label is the
    only thing that names them to a screen reader -- dropping it to
    satisfy the constructor would trade an exception for a silently
    unusable control. This is what every other icon-only button in the
    viewer does; the QC Review pager (`_qc_tab/review/_layout.py:363`)
    is the same pair of arrows.

    Args:
        glyph: The arrow character shown in the button.
        component_id: The button's Dash id.
        label: Accessible name, used for both ``title`` and
            ``aria-label`` -- the arrow itself names nothing.

    Returns:
        A :class:`dash.html.Button` styled as a small outline button.
    """
    return html.Button(
        glyph,
        id=component_id,
        n_clicks=0,
        title=label,
        className="btn btn-outline-secondary btn-sm",
        type="button",
        **cast(Any, {"aria-label": label}),
    )


def _build_toolbar() -> Component:
    """Build the tab's toolbar: config toggle, section pager, export.

    Returns:
        A :class:`dash.html.Div` styled as a thin top bar.
    """
    config_toggle = dbc.Button(
        "⚙ Plot settings",
        id=ids.SCATTER_CONFIG_TOGGLE,
        n_clicks=0,
        color="secondary",
        outline=True,
        size="sm",
    )
    prev_button = _pager_button(
        "‹", ids.SCATTER_PREV_BTN, "Previous section"
    )
    next_button = _pager_button("›", ids.SCATTER_NEXT_BTN, "Next section")
    pager_label = html.Span(
        id=ids.SCATTER_PAGER_LABEL,
        children="",
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "color": COLOR_MUTED,
            "minWidth": "12rem",
            "textAlign": "center",
        },
    )
    export_button = dbc.Button(
        "⇩ Export PDF",
        id=ids.SCATTER_EXPORT_BTN,
        n_clicks=0,
        color="secondary",
        size="sm",
        title="Render every section to a multi-page PDF",
        style={"marginLeft": "auto"},
    )
    export_status = html.Span(
        id=ids.SCATTER_EXPORT_STATUS,
        children="",
        style={
            "fontSize": FONT_SIZE_CAPTION,
            "color": COLOR_MUTED,
            "maxWidth": "22rem",
        },
    )
    return html.Div(
        [
            config_toggle,
            prev_button,
            pager_label,
            next_button,
            export_button,
            export_status,
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "flexWrap": "wrap",
            "padding": "0.5rem 0.75rem",
            "borderBottom": f"1px solid {COLOR_BORDER}",
        },
    )


def _build_inspector() -> Component:
    """Build the click inspector: a right-docked, resizable offcanvas.

    Three pieces, in the order the eye reads them: the colony's identity,
    the crop plus its Contours/Raw control, and the measurement rows
    grouped by the measurer that emitted them.

    The width handle carries the data attributes the shared splitter
    (``results_viewer.js`` section H) dispatches on: ``-target`` naming
    this offcanvas, ``-store`` naming the store a Python callback
    re-applies from, and ``-edge``/``-min``/``-max`` describing the pane
    the controller cannot see for itself. Both ids live here, beside the
    callbacks that bind them, rather than being spelled a second time in
    JavaScript; that is the whole point of Task 12 generalizing the QC
    worklist splitter.

    ``edge="left"`` is load-bearing, not decoration. This is a
    ``placement="end"`` offcanvas and the handle is pinned at
    ``left: 0``, so it rides the pane's *leading* edge: widening it means
    dragging the cursor left, against the sign a left-docked pane needs.
    Omitting the declaration does not fail loudly -- the pane resizes,
    just in the wrong direction, running its edge away from the cursor.

    Returns:
        A :class:`dbc.Offcanvas`, booting closed.
    """
    handle = html.Div(
        id=ids.SCATTER_INSPECTOR_SPLITTER,
        children=[],
        title="Drag to resize the inspector",
        **splitter_attrs(
            target=ids.SCATTER_INSPECTOR,
            store=ids.STORE_SCATTER_INSPECTOR_WIDTH,
            edge="left",
            min_width=_INSPECTOR_WIDTH_MIN,
            max_width=_INSPECTOR_WIDTH_MAX,
        ),
        style={
            "position": "absolute",
            "top": "0",
            "left": "0",
            "width": "6px",
            "height": "100%",
            "cursor": "col-resize",
            "background": COLOR_BORDER,
        },
    )
    contours = dbc.RadioItems(
        id=ids.SCATTER_CONTOUR_TOGGLE,
        options=[
            {"label": "Contours", "value": 1},
            {"label": "Raw", "value": 0},
        ],
        value=1,
        inline=True,
        class_name="btn-group",
        input_class_name="btn-check",
        label_class_name="btn btn-outline-secondary btn-sm",
    )
    return dbc.Offcanvas(
        [
            handle,
            html.Div(
                id=ids.SCATTER_INSPECTOR_TITLE,
                children="",
                style={
                    "fontFamily": FONT_FAMILY_MONO,
                    "fontSize": FONT_SIZE_CAPTION,
                    "color": COLOR_MUTED,
                    "marginBottom": "0.5rem",
                },
            ),
            html.Img(
                id=ids.SCATTER_INSPECTOR_CROP,
                src="",
                alt="Crop of the clicked colony",
                style={
                    "width": "100%",
                    "imageRendering": "pixelated",
                    "border": f"1px solid {COLOR_BORDER}",
                    "borderRadius": "4px",
                },
            ),
            html.Div(contours, style={"margin": "0.5rem 0"}),
            html.Div(id=ids.SCATTER_INSPECTOR_MEASUREMENTS, children=[]),
        ],
        id=ids.SCATTER_INSPECTOR,
        title="Colony",
        placement="end",
        is_open=False,
        scrollable=True,
        backdrop=False,
        style={"width": f"{_INSPECTOR_WIDTH_DEFAULT}px"},
    )


def build_scatter_tab_body(output_root: OutputRoot) -> Component:
    """Build the Scatter tab's body.

    Args:
        output_root: Validated handle on the CLI output directory. Read
            for its display frame's columns and their cardinalities; the
            frame itself is re-read per render so a Refresh is visible.

    Returns:
        A :class:`dash.html.Div` ready to hand to a ``dbc.Tab``.
    """
    return html.Div(
        [
            _build_toolbar(),
            _build_config_popover(output_root),
            dcc.Graph(
                id=ids.SCATTER_GRAPH,
                figure={"data": [], "layout": {}},
                config={"displaylogo": False, "scrollZoom": True},
                style={"height": "72vh"},
            ),
            _build_inspector(),
            dcc.Download(id=ids.SCATTER_DOWNLOAD),
            dcc.Store(id=ids.STORE_SCATTER_SECTION_INDEX, data=0),
            dcc.Store(id=ids.STORE_SCATTER_FINGERPRINT, data=None),
            dcc.Store(id=ids.STORE_SCATTER_COLONY, data=None),
            dcc.Store(
                id=ids.STORE_SCATTER_LEGEND,
                data={"corner": LEGEND_CORNER_DEFAULT, "collapsed": False},
            ),
            dcc.Store(
                id=ids.STORE_SCATTER_INSPECTOR_WIDTH,
                data=_INSPECTOR_WIDTH_DEFAULT,
            ),
        ],
        className="scatter-view-root",
    )

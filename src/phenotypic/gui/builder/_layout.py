"""Pure layout builders for the Dash pipeline builder.

This module is intentionally side-effect-free: every public function returns a
Dash component tree given some immutable inputs. Phase 3 owns all
``@callback`` registration and is the only place where ids declared here are
read or written.

Layout is composed top-to-bottom as:

* :func:`build_breadcrumb` — drill-down indicator + clickable parent links.
* Three-column body:
    - :func:`build_palette` (left) — categorised operation buttons.
    - :func:`build_canvas` (center) — the cytoscape chain.
    - :func:`build_inspector` (right) — node label + param form + preview.
* :func:`build_footer` — image source picker, run/save/load controls.

:func:`build_app_layout` stitches the pieces together and mounts the
``dcc.Store`` instances callbacks read from in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import dash_cytoscape as cyto  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._modal_browser import (
    load_image_modal,
    load_picker_modal,
    save_pipeline_modal,
)
from phenotypic.gui.builder._param_form import param_form
from phenotypic.gui.builder._state import (
    PIPELINE_CLASS_NAME,
    BuilderScope,
    BuilderState,
    _ensure_param_scope,
    current_scope,
    stage_of,
    state_to_json,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from phenotypic.gui._operation_registry import (
        OperationRegistry,
    )


# ---------------------------------------------------------------------------
# Stage colour palette
# ---------------------------------------------------------------------------

#: Background colour for canvas nodes by inferred stage (or pipeline sentinel).
#: Light tints matched to the palette accordion stage colours so a glance at
#: the canvas tells the user which stage of the pipeline a node belongs to.
_STAGE_COLORS = {
    "ops": "#dbe8f5",       # navy-tinted (image ops)
    "meas": "#fdebc7",      # gold-tinted (measurements)
    "post": "#cfeee2",      # green-tinted (post-measurements)
    "pipeline": "#e8e0f0",  # purple-tinted (nested pipeline sentinel)
}

#: Text colour for palette buttons by stage. Matches canvas backgrounds in
#: spirit but uses outline styling so the accordion stays readable.
_STAGE_BUTTON_OUTLINE_COLOR = {
    "ops": "primary",
    "meas": "warning",
    "post": "success",
    "pipeline": "secondary",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_stage(class_name: str) -> str:
    """Return a stage label for *class_name*, falling back to ``"ops"``.

    :func:`stage_of` raises ``KeyError`` for unknown classes (and we can't
    pretend a class we don't know is a measurement op). Layout code should
    degrade gracefully on stale state, so we collapse the error to ``"ops"``.
    """

    if class_name == PIPELINE_CLASS_NAME:
        return "pipeline"
    try:
        return stage_of(class_name)
    except KeyError:
        return "ops"


def _scope_path_labels(state: BuilderState) -> List[str]:
    """Return display labels for each level the breadcrumb walks through.

    Args:
        state: Full builder state.

    Returns:
        Labels: ``["Pipeline", <node label>, <inner node label>, ...]``.
    """

    labels: List[str] = [state.root.name or "Pipeline"]
    scope: BuilderScope = state.root
    for raw in state.breadcrumb:
        if isinstance(raw, str):
            node_id, param = raw, None
        else:
            node_id = raw.get("node_id")
            param = raw.get("param")
        node = next((n for n in scope.nodes if n.node_id == node_id), None)
        if node is None:
            # Stale breadcrumb: bail gracefully.
            labels.append("?")
            break
        if param is None:
            labels.append(node.label or node.class_name)
            if node.nested is None:
                break
            scope = node.nested
        else:
            # Param drill: synthesize a label like "GaussianBlur.sub_op" so
            # the user can see where they are without exposing internal
            # storage details.
            base = node.label or node.class_name
            labels.append(f"{base}.{param}")
            scope = _ensure_param_scope(node, str(param))
    return labels


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


#: Categories that map to the ``_ops`` chain (image-domain operations).
_OPS_STAGE_CATEGORIES = {"Corrector", "Detector", "Enhancer", "Refiner"}

#: Category that maps to the ``_meas`` chain.
_MEAS_STAGE_CATEGORIES = {"Measure"}

#: Category that maps to the ``_post`` chain.
_POST_STAGE_CATEGORIES = {"Post"}


def _palette_for_categories(
    registry: "OperationRegistry",
    *,
    accordion_id: str,
    category_filter: set[str],
) -> dbc.Accordion:
    """Build a categorised palette accordion filtered to *category_filter*.

    Args:
        registry: A pre-populated :class:`OperationRegistry`.
        accordion_id: DOM id assigned to the resulting :class:`dbc.Accordion`.
        category_filter: Subset of registry category names to include.

    Returns:
        A :class:`dbc.Accordion`. When no categories survive the filter the
        accordion is empty (callers can detect this via ``not items``).
    """

    items: List[dbc.AccordionItem] = []
    for category in registry.get_categories():
        if category not in category_filter:
            continue
        ops = sorted(
            registry.get_by_category(category),
            key=lambda info: info.name.lower(),
        )
        if not ops:
            continue

        buttons: List[Any] = []
        for op_info in ops:
            stage = _safe_stage(op_info.name)
            buttons.append(
                dbc.Button(
                    op_info.name,
                    id=ids.palette_button_id(op_info.name),
                    color=_STAGE_BUTTON_OUTLINE_COLOR.get(stage, "primary"),
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className="text-start w-100 mb-1",
                )
            )

        items.append(
            dbc.AccordionItem(
                buttons,
                title=f"{category} ({len(ops)})",
                item_id=f"palette-cat-{category.lower()}",
            )
        )

    return dbc.Accordion(
        items,
        id=accordion_id,
        always_open=True,
        flush=True,
        active_item=[items[0].item_id] if items else None,
    )


def build_palette(registry: "OperationRegistry") -> dbc.Accordion:
    """Build the image-stage palette (Corrector / Detector / Enhancer / Refiner).

    The Measure and Post stages have their own palettes — see
    :func:`build_measure_palette` and :func:`build_post_palette` — so the user
    can tell at a glance which sections of the pipeline (image ops vs.
    measurements vs. post-measurement transforms) a node will land in.

    Args:
        registry: A pre-populated :class:`OperationRegistry`.

    Returns:
        A :class:`dbc.Accordion` ready to drop into the page layout. Always
        carries id :data:`PALETTE_CONTAINER` for the existing tests / callbacks
        that target the canonical palette container.
    """

    return _palette_for_categories(
        registry,
        accordion_id=ids.PALETTE_CONTAINER,
        category_filter=_OPS_STAGE_CATEGORIES,
    )


def build_measure_palette(registry: "OperationRegistry") -> dbc.Accordion:
    """Build the Measurements palette (the ``Measure`` category)."""

    return _palette_for_categories(
        registry,
        accordion_id="palette-meas",
        category_filter=_MEAS_STAGE_CATEGORIES,
    )


def build_post_palette(registry: "OperationRegistry") -> dbc.Accordion:
    """Build the Post-measurements palette (the ``Post`` category)."""

    return _palette_for_categories(
        registry,
        accordion_id="palette-post",
        category_filter=_POST_STAGE_CATEGORIES,
    )


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def _canvas_stylesheet() -> List[dict]:
    """Cytoscape stylesheet used by :func:`build_canvas`.

    Phase 3 may extend this list (e.g. to highlight nodes with hot
    intermediates), so we keep it as a function for easy reuse.
    """

    return [
        {
            "selector": "node",
            "style": {
                "shape": "round-rectangle",
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "data(bg)",
                "border-color": "#dde3ed",
                "border-width": 1,
                "padding": "12px",
                "font-family": "DM Mono, Courier New, monospace",
                "font-size": "12px",
                "font-weight": "500",
                "width": "label",
                "height": 40,
                "min-width": 80,
                "color": "#003660",
            },
        },
        {
            "selector": "node.selected",
            "style": {
                "border-color": "#1b75bc",
                "border-width": 3,
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#8892a4",
                "line-color": "#8892a4",
                "width": 1.5,
            },
        },
    ]


def build_canvas(
    scope: BuilderScope,
    selected_node_id: Optional[str],
) -> cyto.Cytoscape:
    """Render the linear chain for *scope* as a cytoscape canvas.

    Each :class:`StepNode` becomes one cytoscape node; consecutive nodes are
    joined by directed edges. Nested ``ImagePipeline`` nodes get a folder
    glyph in their label so the user can tell drillable nodes apart.

    Args:
        scope: The :class:`BuilderScope` currently in view.
        selected_node_id: If set, the matching node gets the ``"selected"``
            class so the stylesheet highlights it.

    Returns:
        A :class:`dash_cytoscape.Cytoscape` component populated with elements
        for *scope*. Layout is ``"grid"`` with one row to keep the chain
        horizontal.
    """

    elements: List[dict] = []
    prev_id: Optional[str] = None

    for node in scope.nodes:
        stage = _safe_stage(node.class_name)
        label = node.label or node.class_name
        if node.class_name == PIPELINE_CLASS_NAME:
            label = f"\U0001F4C1 {label}"

        node_classes = "selected" if node.node_id == selected_node_id else ""
        elements.append(
            {
                "data": {
                    "id": node.node_id,
                    "label": label,
                    "bg": _STAGE_COLORS.get(stage, _STAGE_COLORS["ops"]),
                    "stage": stage,
                    "class_name": node.class_name,
                },
                "classes": node_classes,
                "selectable": True,
                "grabbable": True,
            }
        )
        if prev_id is not None:
            elements.append(
                {
                    "data": {
                        "id": f"{prev_id}__{node.node_id}",
                        "source": prev_id,
                        "target": node.node_id,
                    },
                }
            )
        prev_id = node.node_id

    return cyto.Cytoscape(
        id=ids.CANVAS_CYTOSCAPE,
        elements=elements,
        layout={
            "name": "grid",
            "rows": 1,
            "cols": max(len(scope.nodes), 1),
            "fit": True,
            "padding": 24,
        },
        # Absolutely position the cytoscape inside its (relative-positioned)
        # ``cytoscape_slot`` parent so it fills the slot regardless of
        # whether intermediate containers use flex, block, or grid layout.
        # Plain ``height: 100%`` was unreliable because percentage heights
        # don't resolve through ``display: block`` parents whose height was
        # set by flex.
        style={
            "position": "absolute",
            "top": 0,
            "left": 0,
            "right": 0,
            "bottom": 0,
        },
        stylesheet=_canvas_stylesheet(),
        autoungrabify=False,
        userPanningEnabled=True,
        userZoomingEnabled=True,
        boxSelectionEnabled=False,
    )


def build_canvas_section(
    scope: BuilderScope, selected_node_id: Optional[str]
) -> html.Div:
    """Wrap :func:`build_canvas` with a header and zoom / reset controls.

    The three controls (Zoom out, Zoom in, Reset view) are wired by
    clientside callbacks registered in :mod:`_callbacks`. They call the
    underlying cytoscape.js API directly via ``window.phenoGetCy()`` —
    ``cy.zoom()`` / ``cy.fit()`` — rather than going through dash-
    cytoscape's prop change detection, which can ignore identical
    layout dicts.
    """

    def _control(label: str, btn_id: str, tooltip: str) -> dbc.Button:
        """Render one canvas-control button with consistent chrome."""

        return dbc.Button(
            label,
            id=btn_id,
            color="secondary",
            outline=True,
            size="sm",
            n_clicks=0,
            title=tooltip,
        )

    controls = dbc.ButtonGroup(
        [
            _control("−", ids.BTN_CANVAS_ZOOM_OUT, "Zoom out"),
            _control("+", ids.BTN_CANVAS_ZOOM_IN, "Zoom in"),
            _control("Reset view", ids.BTN_CANVAS_FIT, "Recenter and zoom-to-fit"),
        ],
        size="sm",
    )

    header = html.Div(
        [
            html.H6("Canvas", className="mb-0"),
            controls,
        ],
        className="d-flex justify-content-between align-items-center mb-2",
    )
    # Cytoscape (``height: 100%``) needs a flex-grown sibling slot so the
    # browser can resolve its percentage height. The header has natural
    # height; the cytoscape wrapper takes the remaining flex space via
    # ``flex: 1 1 0; min-height: 0``. We deliberately leave the slot as a
    # plain block (no inner ``display: flex``) — making it a row-flex
    # container would force its child's width to its content size, which
    # is 0 for a cytoscape (no intrinsic content), collapsing the canvas to
    # zero width.
    # The slot doubles as the swappable ``canvas-cytoscape-wrapper`` whose
    # ``children`` callbacks rewrite to mount a fresh cytoscape after a state
    # mutation (avoids fighting dash-cytoscape's prop diffing). The id also
    # gives ``window.phenoGetCy()`` a reliable anchor for the React fiber walk.
    cytoscape_slot = html.Div(
        build_canvas(scope, selected_node_id),
        id="canvas-cytoscape-wrapper",
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "position": "relative",
        },
    )
    return html.Div(
        [header, cytoscape_slot],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "minHeight": 0,
        },
    )


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


_HIDDEN_STYLE = {"display": "none"}


def _hidden_inspector_widgets() -> List[Any]:
    """Always-rendered hidden placeholders for conditionally-visible inspector
    widgets.

    The fan-in callback in :mod:`phenotypic.gui.builder._callbacks` declares
    ``Input(BTN_DRILL_IN, ...)`` and ``Input(INPUT_NODE_LABEL, ...)``. Dash's
    client-side renderer raises ``ReferenceError`` when a callback fires and
    one of its referenced ids is missing from the live layout, even with
    ``suppress_callback_exceptions=True`` (which only skips server-side
    validation at registration time). To keep the inputs always resolvable,
    every branch of :func:`build_inspector` emits these hidden widgets so the
    ids stay in the DOM whether or not a pipeline node happens to be
    selected.
    """

    return [
        dbc.Input(id=ids.INPUT_NODE_LABEL, type="text", style=_HIDDEN_STYLE),
        dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
    ]


def _empty_inspector_card() -> dbc.Card:
    """Friendly placeholder shown when no canvas node is selected."""

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Inspector", className="card-title"),
                html.P(
                    "Click a node on the canvas to edit its parameters, "
                    "or drag an operation from the palette.",
                    className="text-muted mb-0",
                ),
            ]
        ),
        className="h-100",
    )


def _empty_inspector_div() -> html.Div:
    """Inspector container shown when no node is selected (or selection is stale)."""

    return html.Div(
        [_empty_inspector_card(), *_hidden_inspector_widgets()],
        id=ids.INSPECTOR_CONTAINER,
    )


def build_inspector(
    state: BuilderState,
    registry: "OperationRegistry",
) -> html.Div:
    """Render the inspector pane for the current selection.

    When ``state.selected_node_id`` matches a node in
    :func:`current_scope`, the inspector renders:

    1. A label-text input (id :data:`INPUT_NODE_LABEL`) for renaming.
    2. The auto-generated :func:`param_form` from
       :mod:`phenotypic.gui.builder._param_form` — except for the
       ``ImagePipeline`` sentinel, where a "Drill in" button is shown
       instead so Phase 3 can push the breadcrumb.
    3. A placeholder :data:`INSPECTOR_PREVIEW` div — Phase 3 fills it with a
       channel thumbnail (via :func:`to_data_uri`) for image-stage nodes or a
       :class:`dash_table.DataTable` for measurement steps.

    Args:
        state: The full builder state (used to resolve the active scope and
            selection via :func:`current_scope`).
        registry: Operation registry consulted for parameter metadata.

    Returns:
        A :class:`dash.html.Div` wrapping the inspector card. Always carries
        the :data:`INSPECTOR_CONTAINER` id so callbacks can swap children.
    """

    if state.selected_node_id is None:
        return _empty_inspector_div()

    try:
        scope = current_scope(state)
    except KeyError:
        return _empty_inspector_div()

    node = next(
        (n for n in scope.nodes if n.node_id == state.selected_node_id),
        None,
    )
    if node is None:
        return _empty_inspector_div()

    label_value = node.label or node.class_name
    header_children: List[Any] = [
        html.H5(node.class_name, className="card-title mb-3"),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Label"),
                dbc.Input(
                    id=ids.INPUT_NODE_LABEL,
                    type="text",
                    value=label_value,
                    debounce=True,
                ),
            ],
            className="mb-3",
        ),
    ]

    if node.class_name == PIPELINE_CLASS_NAME:
        body_children: List[Any] = [
            *header_children,
            html.Div(
                [
                    html.P(
                        "This step is a nested pipeline. Drill in to edit "
                        "its operations.",
                        className="text-muted",
                    ),
                    dbc.Button(
                        "Drill in ▸",
                        id=ids.BTN_DRILL_IN,
                        color="primary",
                        outline=True,
                        n_clicks=0,
                    ),
                ]
            ),
            html.Hr(),
            html.Div(id=ids.INSPECTOR_PARAM_FORM),
            html.Div(
                id=ids.INSPECTOR_PREVIEW,
                className="mt-3",
            ),
        ]
        return html.Div(
            dbc.Card(dbc.CardBody(body_children), className="h-100"),
            id=ids.INSPECTOR_CONTAINER,
        )

    op_info = registry.get(node.class_name)
    if op_info is None:
        form: Any = html.Div(
            f"Unknown operation '{node.class_name}'. "
            "It may have been removed from the registry.",
            className="text-warning",
        )
    else:
        form = html.Div(
            param_form(
                op_info,
                current_values=node.params,
                form_id_prefix=node.node_id,
            ),
            id=ids.INSPECTOR_PARAM_FORM,
        )

    body_children = [
        *header_children,
        form,
        html.Hr(),
        html.Div(
            "(Run preview to populate)",
            id=ids.INSPECTOR_PREVIEW,
            className="text-muted small fst-italic",
        ),
        # Non-pipeline nodes don't render a visible Drill-in button, but the
        # fan-in callback still references BTN_DRILL_IN as an Input — keep the
        # id resolvable with a hidden placeholder.
        dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
    ]

    return html.Div(
        dbc.Card(dbc.CardBody(body_children), className="h-100"),
        id=ids.INSPECTOR_CONTAINER,
    )


# ---------------------------------------------------------------------------
# Breadcrumb
# ---------------------------------------------------------------------------


def build_breadcrumb(state: BuilderState) -> html.Nav:
    """Render the breadcrumb showing scope drill path.

    Each segment except the last is a clickable button with a pattern-
    matching id from :func:`breadcrumb_link_id` (depth = position in the
    chain, 0 = root). Phase 3's drill-out callback maps the depth back to a
    breadcrumb truncation length.

    Args:
        state: The full builder state.

    Returns:
        ``html.Nav`` whose children are alternating buttons (clickable
        ancestor scopes) and ``" / "`` separators, terminated by a bold
        ``html.Span`` for the active scope.
    """

    labels = _scope_path_labels(state)
    last_index = len(labels) - 1

    children: List[Any] = []
    for depth, label in enumerate(labels):
        is_last = depth == last_index
        if is_last:
            children.append(html.Span(label, className="fw-bold"))
        else:
            children.append(
                dbc.Button(
                    label,
                    id=ids.breadcrumb_link_id(depth),
                    color="link",
                    size="sm",
                    className="px-1",
                    n_clicks=0,
                )
            )
            children.append(html.Span(" / ", className="text-muted mx-1"))

    return html.Nav(
        children,
        id=ids.BREADCRUMB_CONTAINER,
        className="pheno-breadcrumb d-flex align-items-center mb-2 small",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def _action_card(title: str, buttons: List[Any]) -> dbc.Card:
    """Build a small card with a title and a single ``ButtonGroup`` row.

    Used for the Pipeline I/O and Structure cards in the right column of the
    body row — both have identical chrome (small heading, bordered card,
    full-width button group) so they share a builder.
    """

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="mb-2"),
                dbc.ButtonGroup(buttons, className="w-100", size="sm"),
            ],
            className="py-2",
        ),
        className="mb-2",
    )


def _pipeline_io_card() -> dbc.Card:
    """Build the "Pipeline I/O" card with Save and Load buttons.

    Renders a small :class:`dbc.ButtonGroup` containing two outline buttons
    that each open a modal file browser defined in :mod:`._modal_browser`:

    * **Save** (:data:`ids.BTN_SAVE`) — opens :func:`~_modal_browser.save_pipeline_modal`,
      a folder browser where the user navigates to a target directory and
      enters a filename before confirming the write.
    * **Load** (:data:`ids.BTN_LOAD`) — opens :func:`~_modal_browser.load_picker_modal`,
      a two-stage chooser offering either a JSON file browser (for pipelines
      previously saved with ``ImagePipeline.to_json()``) or a prefab list
      (built-in pipelines from :mod:`phenotypic.prefab`).

    The card lives above the inspector in the right column of the builder
    layout, assembled by :func:`build_app_layout`.

    Returns:
        A :class:`dbc.Card` containing the labelled button group, with
        ``className="mb-2"`` for spacing.
    """

    return _action_card(
        "Pipeline I/O",
        [
            dbc.Button(
                "Save",
                id=ids.BTN_SAVE,
                color="primary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Load",
                id=ids.BTN_LOAD,
                color="primary",
                outline=True,
                n_clicks=0,
            ),
        ],
    )


def _structure_card() -> dbc.Card:
    """+ Pipeline / Delete buttons. Lives above the inspector in the right column."""

    return _action_card(
        "Structure",
        [
            dbc.Button(
                "+ Pipeline",
                id=ids.BTN_NEW_PIPELINE_NODE,
                color="info",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Delete selected",
                id=ids.BTN_DELETE_NODE,
                color="danger",
                outline=True,
                n_clicks=0,
            ),
        ],
    )


def build_footer(image_root: Optional[Path]) -> dbc.Card:
    """Render the footer row with image-source, grid override, and run-preview controls.

    Pipeline I/O (Save/Load) and Structure (+ Pipeline / Delete) have their
    own cards above the inspector in the right column — see
    :func:`_pipeline_io_card` and :func:`_structure_card`. This footer card
    handles the image-selection and execution surface.

    Layout (left to right inside the card body):

    * **Image source** (left column): two buttons and a status label.
      "Load image…" (:data:`ids.BTN_LOAD_IMAGE`) opens
      :func:`~_modal_browser.load_image_modal`, a directory browser filtered
      to plate image formats. "Use synthetic plate"
      (:data:`ids.BTN_USE_SYNTHETIC`) immediately loads the bundled synthetic
      yeast plate without opening any modal — useful for quick iteration
      without a real plate image. The :data:`ids.ACTIVE_IMAGE_LABEL` below
      the buttons shows the basename of the currently active image so the
      user can confirm the selection before running.
    * **Grid** (right column): optional ``nrows`` / ``ncols`` inputs
      (:data:`ids.INPUT_NROWS`, :data:`ids.INPUT_NCOLS`) for pipelines that
      contain :class:`~phenotypic.abc_.GridOperation` steps. Leave blank to
      use the default ``8 × 12`` grid assumed by ``GridImage.imread()``.
    * **Run preview**: :data:`ids.BTN_RUN_PREVIEW` wrapped in a
      :class:`dcc.Loading` spinner (:data:`ids.PREVIEW_LOADING`).

    Args:
        image_root: Accepted for API parity with the previous inline picker
            but not consumed directly by this function. The modal factories
            in :mod:`._modal_browser` receive it via :func:`build_app_layout`,
            which mounts the three modals separately.

    Returns:
        A :class:`dbc.Card` with id :data:`ids.FOOTER_CONTAINER` whose body
        is the two-column footer row.
    """
    del image_root  # consumed by load_image_modal in build_app_layout

    grid_inputs = dbc.Row(
        [
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("nrows"),
                        dbc.Input(
                            id=ids.INPUT_NROWS,
                            type="number",
                            min=1,
                            step=1,
                            placeholder="auto",
                            debounce=True,
                        ),
                    ],
                    size="sm",
                ),
                width=6,
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("ncols"),
                        dbc.Input(
                            id=ids.INPUT_NCOLS,
                            type="number",
                            min=1,
                            step=1,
                            placeholder="auto",
                            debounce=True,
                        ),
                    ],
                    size="sm",
                ),
                width=6,
            ),
        ],
        className="g-1 mb-2",
    )

    run_button = dcc.Loading(
        id=ids.PREVIEW_LOADING,
        type="default",
        children=dbc.Button(
            "Run preview",
            id=ids.BTN_RUN_PREVIEW,
            color="success",
            n_clicks=0,
            className="w-100",
        ),
    )

    image_source = html.Div(
        [
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "Load image…",
                        id=ids.BTN_LOAD_IMAGE,
                        color="primary",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Use synthetic plate",
                        id=ids.BTN_USE_SYNTHETIC,
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                    ),
                ],
                size="sm",
                className="mb-2",
            ),
            html.Div(
                "(no image loaded)",
                id=ids.ACTIVE_IMAGE_LABEL,
                className="text-muted small text-monospace",
                style={"wordBreak": "break-all"},
            ),
        ]
    )

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Image source", className="mb-2"),
                            image_source,
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.H6("Grid", className="mb-2"),
                            grid_inputs,
                            run_button,
                        ],
                        md=4,
                    ),
                ],
                className="g-3",
            )
        ),
        id=ids.FOOTER_CONTAINER,
        className="mt-3",
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------


def build_app_layout(
    state: BuilderState,
    registry: "OperationRegistry",
    image_root: Optional[Path],
) -> html.Div:
    """Compose the top-level page layout.

    Three vertical sections wrapped in a ``dbc.Container(fluid=True)``:

    * Breadcrumb nav (full width).
    * Three-column body — palette, canvas, inspector — sized 3/6/3.
    * Footer card with image source + I/O controls.

    Mounts the three :class:`dcc.Store` instances callbacks need:

    * :data:`STORE_BUILDER_STATE` seeded with ``state_to_json(state)``.
    * :data:`STORE_SESSION_ID` configured for ``storage_type='session'``;
      Phase 3 fills it on first interaction (a fresh uuid per browser tab).
    * :data:`STORE_INTERMEDIATE_KEYS` for synchronising the canvas with the
      session-side intermediates cache.

    A floating :class:`dbc.Toast` at id :data:`TOAST_NOTIFICATION` carries
    save/load/preview status messages.

    Args:
        state: Initial :class:`BuilderState`. Stored serialised in
            :data:`STORE_BUILDER_STATE`.
        registry: Operation registry used to populate the palette and the
            inspector.
        image_root: Optional directory root for the directory picker.

    Returns:
        A :class:`dash.html.Div` ready to assign to ``app.layout``.
    """

    # Vertical layout strategy:
    # - The right column has natural height (Pipeline I/O + Structure +
    #   inspector content). When a node is selected, its param form grows the
    #   column, which grows the row.
    # - dbc.Row's default ``align-items: stretch`` stretches every other
    #   column to match the row height.
    # - The palette column is made into a flex column whose inner scroll
    #   wrapper uses ``flex: 1 1 0; min-height: 0`` so the palette content
    #   does NOT contribute to the row's natural-height calculation. With
    #   ``flex-basis: 0`` the palette wrapper starts at zero, then stretches
    #   to fill whatever height the row settles on, and its overflow scrolls.
    # Picked so the canvas 30% slice gets at least ~210 px and the inspector
    # 70% slice gets ~490 px — comfortable for a typical 3-5 node pipeline
    # with a moderate param form.
    _ROW_MIN_HEIGHT = "700px"
    _SCROLL_FILL_STYLE = {
        "flex": "1 1 0",
        "minHeight": 0,
        "overflowY": "auto",
    }

    # Inner container holds the actual scrolling content; the outer wrapper
    # gives the chevron pseudo-elements a positioning context (see
    # ``assets/builder.css`` and ``assets/builder.js``).
    def _palette_section(title: str, accordion: dbc.Accordion) -> html.Div:
        """Wrap one palette accordion under a labelled section heading."""

        return html.Div(
            [html.H6(title, className="mb-2"), accordion],
            className="pheno-palette-section",
        )

    palette_inner = html.Div(
        [
            _palette_section("Operations", build_palette(registry)),
            _palette_section("Measurements", build_measure_palette(registry)),
            _palette_section(
                "Post-measurements", build_post_palette(registry)
            ),
        ],
        className="pheno-scroll pe-2",
        style=_SCROLL_FILL_STYLE,
    )
    # palette_column is a flex column so its child can use ``flex: 1`` to
    # fill available space without contributing content height to its
    # parent's natural-size calculation.
    palette_column = html.Div(
        palette_inner,
        className="pheno-scroll-wrap",
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
        },
    )

    # Inspector now sits under the canvas in the middle column. Its wrapper
    # uses ``flex: 1`` so it fills whatever flex slot the middle column
    # allocates (the 70% slice — see ``middle_column`` below). Internal
    # overflow scrolls when the param form is taller than the slot.
    inspector_inner = html.Div(
        build_inspector(state, registry),
        className="pheno-scroll pe-2",
        style=_SCROLL_FILL_STYLE,
    )
    inspector_wrap = html.Div(
        inspector_inner,
        className="pheno-scroll-wrap",
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
        },
    )

    # Right portion of the body is itself a 50 / 50 vertical split:
    #   Top half  = Canvas (md=8 of 9) + Pipeline I/O & Structure (md=4 of 9)
    #   Bottom half = Inspector (full width — spans both columns)
    # The two halves both use ``flex: 1 1 0`` so the split is exact regardless
    # of content size; ``min-height: 0`` lets each half shrink without the
    # inspector content forcing growth.
    top_half = dbc.Row(
        [
            dbc.Col(
                build_canvas_section(state.root, state.selected_node_id),
                md=8,
                className="d-flex flex-column",
                style={"minHeight": 0},
            ),
            dbc.Col(
                [_pipeline_io_card(), _structure_card()],
                md=4,
                className="border-start ps-3",
            ),
        ],
        className="g-3",
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "marginLeft": 0,
            "marginRight": 0,
        },
    )

    bottom_half = html.Div(
        inspector_wrap,
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
        },
    )

    # ``html.Hr`` between halves gives a clear visual separator without
    # competing with the flex sizing. ``flex-shrink: 0`` keeps the rule from
    # being absorbed when the row gets short; ``my-2`` collapses Bootstrap's
    # default ``Hr`` margins to a single tidy gap.
    divider = html.Hr(
        className="my-2",
        style={"flexShrink": 0, "width": "100%"},
    )

    right_section = html.Div(
        [top_half, divider, bottom_half],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "minHeight": 0,
            "width": "100%",
        },
    )

    body_row = dbc.Row(
        [
            # ``d-flex flex-column`` makes this column a flex container so the
            # palette wrapper's ``flex: 1`` actually has a flex parent to grow
            # against. Without it the wrapper would still see ``display:
            # block`` and ignore its flex sizing.
            dbc.Col(
                palette_column,
                md=3,
                className="border-end pe-3 d-flex flex-column",
            ),
            dbc.Col(
                right_section,
                md=9,
                className="ps-3 d-flex flex-column",
            ),
        ],
        className="g-3",
        style={"minHeight": _ROW_MIN_HEIGHT, "alignItems": "stretch"},
    )

    stores = html.Div(
        [
            dcc.Store(
                id=ids.STORE_BUILDER_STATE,
                data=state_to_json(state),
            ),
            dcc.Store(
                id=ids.STORE_SESSION_ID,
                storage_type="session",
                data="",
            ),
            dcc.Store(
                id=ids.STORE_INTERMEDIATE_KEYS,
                data=[],
            ),
            # Sink for clientside canvas-control callbacks (zoom in/out, fit).
            dcc.Store(
                id=ids.STORE_CANVAS_CONTROL,
                data=0,
            ),
            # Active image path; populated by the directory browser, consumed
            # by Run preview.
            dcc.Store(
                id=ids.STORE_IMAGE_PATH,
                data="",
            ),
        ]
    )

    toast = dbc.Toast(
        id=ids.TOAST_NOTIFICATION,
        header="Pipeline builder",
        is_open=False,
        dismissable=True,
        duration=5000,
        icon="primary",
        style={
            "position": "fixed",
            "top": 20,
            "right": 20,
            "minWidth": 300,
            "zIndex": 1080,
        },
    )

    modals = html.Div(
        [
            save_pipeline_modal(image_root),
            load_picker_modal(image_root),
            load_image_modal(image_root),
        ]
    )

    header = html.Div(
        [
            html.Img(
                src="/assets/pheno_logo.png",
                alt="PhenoTypic",
                className="pheno-app-header__logo",
            ),
            html.Div(
                [
                    html.H1(
                        "PhenoTypic",
                        className="pheno-app-header__title",
                    ),
                    html.Div(
                        "Pipeline Builder",
                        className="pheno-app-header__subtitle",
                    ),
                ],
            ),
        ],
        className="pheno-app-header",
    )

    return html.Div(
        [
            stores,
            toast,
            modals,
            dbc.Container(
                [header, build_breadcrumb(state), body_row, build_footer(image_root)],
                fluid=True,
            ),
        ]
    )


__all__ = [
    "build_palette",
    "build_canvas",
    "build_inspector",
    "build_breadcrumb",
    "build_footer",
    "build_app_layout",
]

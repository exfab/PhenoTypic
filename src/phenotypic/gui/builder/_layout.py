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

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import dash_cytoscape as cyto  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    FONT_FAMILY_MONO,
    FONT_SIZE_LABEL,
    OI_PURPLE,
)
from phenotypic.gui._shared import SHARED_LOGO_PATH
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._ids import StageName
from phenotypic.gui.builder._modal_browser import (
    load_image_modal,
    load_picker_modal,
    save_pipeline_modal,
)
from phenotypic.gui.builder._param_form import param_form
from phenotypic.gui.builder._point_picker import build_point_picker_modal
from phenotypic.gui.builder._state import (
    PIPELINE_CLASS_NAME,
    BuilderScope,
    BuilderState,
    StepNode,
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
_STAGE_COLORS: dict[StageName | str, str] = {
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
            button_children: List[Any] = [html.Span(op_info.name)]
            button_class = "text-start w-100 mb-1"
            if op_info.is_point_pickable:
                button_children.append(
                    dbc.Badge(
                        "PICK",
                        className="shell-badge shell-badge-pickable ms-2",
                        pill=True,
                    )
                )
                button_class = f"{button_class} builder-op-pickable"
            buttons.append(
                dbc.Button(
                    button_children,
                    id=ids.palette_button_id(op_info.name),
                    color=_STAGE_BUTTON_OUTLINE_COLOR.get(stage, "primary"),
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className=button_class,
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

    The popover-anchored aux design renders two extra visible markers per
    consumer node:

    * **Main I/O ports** — small blue circles on the left (input) and
      right (output) edges of every ribbon node. Image-flow edges connect
      the previous node's main-output port to the next node's main-input
      port (the wire visibly *enters* and *exits* each operation rather
      than routing through node centers).
    * **Aux ports** — small purple rounded-square markers on the
      BOTTOM edge of every consumer that has op-typed parameters. One
      marker per param (regardless of slot cardinality); list-typed
      params still show a single marker. Tapping the marker opens the
      canvas-anchored popover which then handles slot management.

    These markers live as additional cytoscape *nodes* (not edges) so
    they're positionable independently and the popover's clientside JS
    glue can target them by ``id`` prefix. The cytoscape side intentionally
    references ``OI_PURPLE`` directly (the value behind
    ``--color-interactive`` / ``--oi-purple``) because cytoscape's canvas
    renderer cannot resolve CSS custom properties.
    """

    return [
        {
            "selector": "node",
            "style": {
                "shape": "round-rectangle",
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "ellipsis",
                "text-max-width": 160,
                "background-color": "data(bg)",
                "border-color": COLOR_BORDER,
                "border-width": 1,
                "padding": "8px",
                "font-family": FONT_FAMILY_MONO,
                # Cytoscape canvas-renders labels and only accepts pixel
                # values for font-size; rem units silently fall back.
                "font-size": "12px",
                "font-weight": "500",
                # Fixed-width ribbon nodes so I/O port placement is
                # predictable (text-wrap kicks in for long class names).
                "width": 180,
                "height": 54,
                "color": COLOR_NAVY,
            },
        },
        {
            "selector": "node.selected",
            "style": {
                "border-color": COLOR_BLUE,
                "border-width": 3,
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": COLOR_MUTED,
                "line-color": COLOR_MUTED,
                "width": 1.5,
            },
        },
        # Image-flow edges between consecutive main-ribbon nodes. Endpoints
        # are the upstream node's main-output port and the downstream
        # node's main-input port (small blue circles) — the wire visibly
        # exits the upstream op on its right and enters the downstream
        # op on its left. ``outside-to-node`` keeps the endpoint clean
        # against the port marker rather than routing through it.
        {
            "selector": "edge.image-flow",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": COLOR_MUTED,
                "line-color": COLOR_MUTED,
                "width": 1.5,
                "source-endpoint": "outside-to-node",
                "target-endpoint": "outside-to-node",
            },
        },
        # Main I/O port: small blue circle on the LEFT (input) or RIGHT
        # (output) edge of every ribbon node. Always filled — wired state
        # for the image flow lives in the edge, not in the port itself.
        # No label and no padding so the marker reads as a discrete dot
        # rather than a tiny rectangle.
        {
            "selector": "node.main-port",
            "style": {
                "shape": "ellipse",
                "label": "",
                "width": 10,
                "height": 10,
                "background-color": COLOR_BLUE,
                "border-color": COLOR_BLUE,
                "border-width": 1,
                "padding": 0,
            },
        },
        # Bottom-edge aux port: small purple rounded-square marker. One
        # per op-typed parameter on the consumer (regardless of slot
        # cardinality). Empty (no slot wired) renders hollow — gray fill
        # with a purple border — so users can scan a dense canvas and
        # tell at a glance which consumers still need an aux configured.
        {
            "selector": "node.aux-port",
            "style": {
                "shape": "round-rectangle",
                "label": "",
                "width": 10,
                "height": 10,
                "background-color": COLOR_BORDER,
                "border-color": OI_PURPLE,
                "border-width": 1.5,
                "padding": 0,
            },
        },
        # Wired aux port: solid purple fill. Any slot non-empty flips the
        # marker from hollow to filled so the canvas state mirrors the
        # popover state without needing the popover to be open.
        {
            "selector": "node.aux-port.aux-port--wired",
            "style": {
                "background-color": OI_PURPLE,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Preset-layout positioning
# ---------------------------------------------------------------------------
#
# Cytoscape's ``preset`` layout requires explicit ``(x, y)`` for every node;
# the constants below pin the geometry of the main ribbon and its
# attached I/O / aux port markers so callbacks can stay layout-agnostic.
# Numbers are tuned against the ~700px canvas slot configured in
# ``build_app_layout``.

#: Horizontal step between consecutive main-ribbon nodes (px).
_RIBBON_X_STEP: int = 180

#: Left padding before the first ribbon node (px).
_RIBBON_X_OFFSET: int = 24

#: Y position of every main-ribbon node (px).
_RIBBON_Y: int = 80

#: Half-width of a ribbon operation node (px). The cytoscape stylesheet
#: pins ``width: 180`` so the consumer's left edge is at
#: ``ribbon_x - _RIBBON_HALF_WIDTH`` and its right edge at
#: ``ribbon_x + _RIBBON_HALF_WIDTH``.
_RIBBON_HALF_WIDTH: int = 90

#: Half-height of a ribbon operation node (px). The cytoscape stylesheet
#: pins ``height: 54`` so the consumer's bottom edge sits at
#: ``_RIBBON_Y + _RIBBON_HALF_HEIGHT``.
_RIBBON_HALF_HEIGHT: int = 27

#: Gap between the consumer's bottom edge and the aux-port marker's
#: center (px). The aux marker is 10px tall so a small offset keeps it
#: visually attached to the consumer without overlapping its bottom
#: border.
_AUX_PORT_Y_OFFSET: int = 8

#: Horizontal spacing between adjacent aux ports on the same consumer
#: (px). When a consumer has multiple op-typed parameters (e.g.
#: ``CompositeDetector.detectors`` + ``CompositeDetector.shape_detector``),
#: their markers spread across the bottom edge centered on the consumer.
_AUX_PORT_X_SPACING: int = 40


def _aux_param_names(
    node: StepNode, registry: "OperationRegistry"
) -> List[str]:
    """Return the op-typed parameter names a consumer node exposes.

    Walks the registry's parameter metadata (which preserves declaration
    order via :class:`inspect.Signature`) and returns the subset of
    parameter names whose annotation is operation- or pipeline-typed —
    the canvas renders one bottom-edge aux port marker per such param.

    Parameters not in the registry are skipped silently; the node may
    have a stale ``aux_ports`` map for a class that was renamed/removed,
    and the canvas should degrade gracefully rather than render markers
    for parameters that no longer exist.

    Args:
        node: The consumer :class:`StepNode` being rendered.
        registry: Operation registry consulted for parameter metadata.

    Returns:
        Ordered list of aux-eligible parameter names (preserves the
        order they were declared on the class).
    """

    info = registry.get(node.class_name)
    if info is None:
        return []

    return [
        param_name
        for param_name, p in info.parameters.items()
        if p.is_operation or p.is_pipeline
    ]


def build_canvas(
    scope: BuilderScope,
    selected_node_id: Optional[str],
) -> cyto.Cytoscape:
    """Render the linear chain for *scope* as a cytoscape canvas.

    Each :class:`StepNode` becomes one cytoscape node; consecutive
    main-ribbon nodes are joined by ``image-flow`` edges that route
    through the upstream node's main-output port and the downstream
    node's main-input port (small blue circles attached to each
    ribbon node's left/right edges). Nested ``ImagePipeline`` nodes
    get a folder glyph in their label so the user can tell drillable
    nodes apart.

    Popover-anchored aux additions:

    * Every ribbon node renders two main I/O ports as additional
      cytoscape elements — a blue circle on the LEFT edge (input) and
      one on the RIGHT edge (output). Image-flow edges connect upstream
      output to downstream input so the wire visibly enters and exits
      each operation.
    * Consumer nodes with op-typed parameters (e.g.
      ``FilamentousFungiDetector.inoculum_detector``) render one small
      purple aux-port marker per param on their BOTTOM edge. Tapping a
      marker opens the canvas-anchored popover where the user picks /
      edits / disconnects the wired aux. Aux StepNodes themselves are
      NOT rendered on the main canvas — they live embedded inside their
      consumer's ``aux_ports`` slot list and only render when the user
      drills into them.

    Args:
        scope: The :class:`BuilderScope` currently in view.
        selected_node_id: If set, the matching node gets the
            ``"selected"`` class so the stylesheet highlights it.

    Returns:
        A :class:`dash_cytoscape.Cytoscape` component populated with
        elements for *scope*. Layout is ``"preset"`` — Python computes
        ``(x, y)`` for every element so callbacks can stay layout-agnostic.
    """

    # Local import so the module's top-level import graph stays Dash-only.
    # ``get_registry`` walks every operation module; calling it once per
    # render is fine because the registry is a process-wide singleton.
    from phenotypic.gui._operation_registry import get_registry

    registry = get_registry()

    elements: List[dict] = []

    # ── 1. Compute positions for the main ribbon. ────────────────────────
    ribbon_x_by_id: Dict[str, int] = {}
    for i, node in enumerate(scope.nodes):
        ribbon_x_by_id[node.node_id] = _RIBBON_X_OFFSET + i * _RIBBON_X_STEP

    # ── 2. Image-flow edges (drawn first so they sit underneath nodes).
    #      Endpoints are the upstream output port and the downstream
    #      input port — the wire visibly enters/exits each operation
    #      through its port markers rather than routing through node
    #      centers. ──────────────────────────────────────────────────────
    prev_id: Optional[str] = None
    for node in scope.nodes:
        if prev_id is not None:
            elements.append(
                {
                    "data": {
                        "id": f"{prev_id}__{node.node_id}",
                        "source": ids.main_output_port_id(prev_id),
                        "target": ids.main_input_port_id(node.node_id),
                    },
                    "classes": "image-flow",
                }
            )
        prev_id = node.node_id

    # ── 3. Main-ribbon nodes. ───────────────────────────────────────────
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
                "position": {
                    "x": ribbon_x_by_id[node.node_id],
                    "y": _RIBBON_Y,
                },
            }
        )

    # ── 4. Main I/O ports (small blue circles on left/right edges). ─────
    # Emitted as cytoscape nodes (not edges) so they're positionable
    # independently; image-flow edges target their ids so the wires
    # visibly attach to the port markers rather than node centers.
    for node in scope.nodes:
        consumer_x = ribbon_x_by_id[node.node_id]
        elements.append(
            {
                "data": {
                    "id": ids.main_input_port_id(node.node_id),
                    "parent_node_id": node.node_id,
                    "side": "input",
                },
                "classes": "main-port main-port--input",
                "selectable": False,
                "grabbable": False,
                "position": {
                    "x": consumer_x - _RIBBON_HALF_WIDTH,
                    "y": _RIBBON_Y,
                },
            }
        )
        elements.append(
            {
                "data": {
                    "id": ids.main_output_port_id(node.node_id),
                    "parent_node_id": node.node_id,
                    "side": "output",
                },
                "classes": "main-port main-port--output",
                "selectable": False,
                "grabbable": False,
                "position": {
                    "x": consumer_x + _RIBBON_HALF_WIDTH,
                    "y": _RIBBON_Y,
                },
            }
        )

    # ── 5. Bottom-edge aux ports (one per op-typed param). ──────────────
    # Multiple aux params spread evenly across the bottom edge centered
    # on the consumer's x. Wired state (any slot non-empty) flips the
    # marker class to ``aux-port--wired`` so the stylesheet fills it.
    for consumer in scope.nodes:
        aux_param_names = _aux_param_names(consumer, registry)
        if not aux_param_names:
            continue
        n_aux = len(aux_param_names)
        consumer_x = ribbon_x_by_id[consumer.node_id]
        aux_y = _RIBBON_Y + _RIBBON_HALF_HEIGHT + _AUX_PORT_Y_OFFSET
        for i, param_name in enumerate(aux_param_names):
            x_offset = int((i - (n_aux - 1) / 2) * _AUX_PORT_X_SPACING)
            slots = consumer.aux_ports.get(param_name) or []
            wired_count = sum(1 for slot in slots if slot is not None)
            wired = wired_count > 0
            port_classes = "aux-port aux-port--wired" if wired else "aux-port"
            elements.append(
                {
                    "data": {
                        "id": ids._encode_aux_port_id(
                            consumer.node_id, param_name
                        ),
                        "parent_node_id": consumer.node_id,
                        "param": param_name,
                        "wired": wired,
                        "wired_count": wired_count,
                        "total_count": len(slots),
                    },
                    "classes": port_classes,
                    "selectable": True,
                    "grabbable": False,
                    "position": {
                        "x": consumer_x + x_offset,
                        "y": aux_y,
                    },
                }
            )

    return cyto.Cytoscape(
        id=ids.CANVAS_CYTOSCAPE,
        elements=elements,
        layout={
            "name": "preset",
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
        # Cap auto-fit zoom so a sparse canvas (1-2 nodes) doesn't
        # balloon the consumer to fill the viewport. 1.0 keeps the
        # absolute Python-computed positions intact; the user can still
        # zoom in further via the toolbar.
        maxZoom=1.0,
        minZoom=0.25,
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

    # Destructive action sits next to the zoom group (not inside it) so the
    # red outline reads as separate from the neutral pan/zoom controls.
    delete_btn = dbc.Button(
        "Delete selected",
        id=ids.BTN_DELETE_NODE,
        color="danger",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Remove the selected node from the pipeline",
    )

    header = html.Div(
        [
            html.H6("Canvas", className="mb-0"),
            html.Div(
                [controls, delete_btn],
                className="d-flex align-items-center gap-2",
            ),
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
    #
    # The popover container is mounted as a sibling of the cytoscape canvas
    # inside the same relative-positioned wrapper so ``cytoscape-popper``
    # (via popper.js) can position it relative to the tapped aux port and
    # pan/zoom along with the canvas. The container is hidden by default;
    # the ``aux_popover.js`` clientside glue toggles its ``display`` and
    # writes the structured tap data to :data:`PORT_CLICK_STORE` so server-
    # side callbacks (Wave 4) can fill in its children based on
    # ``state.inspector_focus_aux`` + the active port click.
    # Popover container is a SIBLING of the cytoscape wrapper (not a
    # child). The fan-in callback in ``_callbacks.py`` writes to
    # ``canvas-cytoscape-wrapper.children`` on every state mutation,
    # which would wipe the popover container if it lived inside the
    # wrapper. By placing it as a sibling, it persists in the DOM and
    # the popover-content callback can write to it independently
    # without racing the wrapper-replacement. Popper.js positions it
    # absolutely on screen relative to the tapped aux-port cytoscape
    # node, so the DOM hierarchy doesn't matter for layout.
    popover_container = html.Div(
        id=ids.POPOVER_CONTAINER,
        className="cy-popover",
        style={"display": "none"},
        children=[],
    )
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
        [header, cytoscape_slot, popover_container],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "minHeight": 0,
            "position": "relative",  # establish stacking context for popover
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
        *_doc_section_widgets(None),
    ]


def _doc_section_widgets(docstring: Optional[str]) -> List[Any]:
    """Return the Inspector "Documentation" section as a list of components.

    Operations carry a Google-style class docstring on the ``OperationInfo``
    record; surfacing it in-place lets users browse "what does this op do?"
    without leaving the Inspector. The section is collapsed by default so
    it never crowds out the parameter form.

    The toggle callback in :mod:`phenotypic.gui.builder._callbacks` keys on
    ``INSPECTOR_DOC_TOGGLE`` and ``INSPECTOR_DOC_COLLAPSE``; both ids must
    therefore exist on every render path. When the operation has no
    docstring (or for branches like ``_empty_inspector_div`` and the
    nested-pipeline branch where there is nothing meaningful to render) we
    emit hidden placeholders carrying the same ids so the callback's
    ``Input``/``State`` always resolve.

    Args:
        docstring: Raw ``cls.__doc__`` value from the operation registry,
            or ``None`` (empty inspector / pipeline sentinel / unknown
            operation).

    Returns:
        Single-element list with the visible doc section when ``docstring``
        is non-empty, otherwise a two-element list of hidden placeholders.
    """

    if docstring and docstring.strip():
        cleaned = inspect.cleandoc(docstring)
        return [
            html.Div(
                [
                    dbc.Button(
                        "Documentation ▾",
                        id=ids.INSPECTOR_DOC_TOGGLE,
                        color="link",
                        size="sm",
                        n_clicks=0,
                        className="inspector-doc-toggle p-0",
                    ),
                    dbc.Collapse(
                        html.Pre(
                            cleaned,
                            className="inspector-doc-body",
                            style={
                                "whiteSpace": "pre-wrap",
                                "fontFamily": FONT_FAMILY_MONO,
                                "fontSize": FONT_SIZE_LABEL,
                                "color": COLOR_MUTED,
                                "marginBottom": 0,
                            },
                        ),
                        id=ids.INSPECTOR_DOC_COLLAPSE,
                        is_open=False,
                    ),
                ],
                className="inspector-doc-section mb-3",
            )
        ]

    return [
        dbc.Button(
            id=ids.INSPECTOR_DOC_TOGGLE,
            n_clicks=0,
            style=_HIDDEN_STYLE,
        ),
        dbc.Collapse(
            html.Div(),
            id=ids.INSPECTOR_DOC_COLLAPSE,
            is_open=False,
            style=_HIDDEN_STYLE,
        ),
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


def _compatible_classes_for_port(
    param_info: Any, registry: "OperationRegistry"
) -> List[str]:
    """Return registry classes that satisfy a given aux-port type.

    Used by :func:`build_popover_contents` to filter the popover's class
    palette down to ops/pipelines that the consumer's port will actually
    accept (the ``wire_create`` dispatch validates the same contract;
    pre-filtering here saves the user a useless click).

    Args:
        param_info: ``ParamInfo`` for the consumer's aux-port-eligible
            parameter; expected ``is_operation`` or ``is_pipeline`` true.
        registry: Operation registry to enumerate.

    Returns:
        Sorted list of class-name strings. The
        :data:`PIPELINE_CLASS_NAME` sentinel appears when the port accepts
        an :class:`~phenotypic.ImagePipeline`.
    """

    from phenotypic.abc_ import ImageOperation

    classes: List[str] = []
    if param_info.is_operation:
        for cls_name, info in registry.get_all().items():
            cls = getattr(info, "cls", None)
            if isinstance(cls, type) and issubclass(cls, ImageOperation):
                classes.append(cls_name)
    if param_info.is_pipeline:
        classes.append(PIPELINE_CLASS_NAME)
    return sorted(set(classes))


def _resolve_inspector_focus_target(
    state: BuilderState, scope: BuilderScope
) -> Optional[tuple[StepNode, StepNode, str, int]]:
    """Resolve ``state.inspector_focus_aux`` to a concrete consumer + aux.

    The inspector focus override lets the user edit a wired aux's params
    without leaving the canvas-selected consumer's context. This helper
    walks ``state.inspector_focus_aux`` (shape ``{"target_node_id",
    "param", "slot"}``), locates the consumer in *scope*, and resolves
    the embedded aux ``StepNode`` at ``consumer.aux_ports[param][slot]``.

    Returns ``None`` when the focus is unset or unresolvable (e.g. the
    consumer was deleted, the slot is empty, or the param doesn't exist
    anymore). Callers should fall back to rendering the canvas-selected
    consumer's params in that case.

    Args:
        state: Full :class:`BuilderState`.
        scope: The current :class:`BuilderScope` (already resolved via
            :func:`current_scope`).

    Returns:
        Tuple ``(consumer_node, aux_node, param_name, slot_idx)`` when
        focus resolves cleanly; ``None`` otherwise.
    """

    focus = state.inspector_focus_aux
    if focus is None:
        return None

    target_node_id = focus.get("target_node_id")
    param = focus.get("param")
    raw_slot = focus.get("slot", 0)
    if not isinstance(target_node_id, str) or not isinstance(param, str):
        return None
    try:
        slot = int(raw_slot)
    except (TypeError, ValueError):
        return None

    consumer = next(
        (n for n in scope.nodes if n.node_id == target_node_id), None
    )
    if consumer is None:
        return None

    slots = consumer.aux_ports.get(param) or []
    if slot < 0 or slot >= len(slots):
        return None

    aux_node = slots[slot]
    if aux_node is None:
        return None
    return consumer, aux_node, param, slot


#: DOM id of the inspector's aux-focus banner. The banner is rendered
#: only when ``state.inspector_focus_aux`` is set, so Wave 4 callbacks
#: must use ``allow_optional`` / ``suppress_callback_exceptions`` when
#: wiring it as an Input.
INSPECTOR_FOCUS_AUX_BANNER_ID: str = "inspector-focus-aux-banner"


def _inspector_focus_aux_banner(
    consumer: StepNode, param: str, slot: int
) -> html.Div:
    """Build the breadcrumb-style banner shown when the inspector mirrors an aux.

    The banner sits at the top of the inspector pane when
    ``state.inspector_focus_aux`` is set. Clicking it dispatches
    ``set_inspector_focus(focus="consumer", ...)`` (handled by a Wave 4
    callback) so the user can revert to the consumer's params.

    Args:
        consumer: The canvas-selected consumer node whose param is being
            mirrored.
        param: Name of the consumer's op-typed parameter whose wired aux
            is currently displayed.
        slot: Zero-based slot index inside that param's slot list.

    Returns:
        A ``html.Div`` styled by ``.inspector-focus-aux-banner`` in
        ``builder.css``. The container carries the
        :data:`INSPECTOR_FOCUS_AUX_BANNER_ID` id so Wave 4 callbacks can
        listen for ``n_clicks`` to clear the focus override.
    """

    label = consumer.label or consumer.class_name
    return html.Div(
        [
            html.Span("← ", className="me-1"),
            html.Span(f"{label}.{param}", className="fw-semibold"),
            html.Span(f" / slot {slot}", className="text-muted ms-1"),
        ],
        id=INSPECTOR_FOCUS_AUX_BANNER_ID,
        className="inspector-focus-aux-banner",
        title="Click to return to the consumer's params",
        n_clicks=0,
    )


def build_inspector(
    state: BuilderState,
    registry: "OperationRegistry",
) -> html.Div:
    """Render the inspector pane for the current selection.

    The inspector mirrors one node's parameters at a time. In the
    popover-anchored aux design two distinct modes are possible:

    * **Consumer mode** (default): renders the canvas-selected node's
      param form. This is the path used whenever
      ``state.inspector_focus_aux`` is ``None``.
    * **Aux-focus mode**: triggered when the user opens the popover for
      a wired aux (or wires a new one). ``state.inspector_focus_aux``
      carries ``{"target_node_id", "param", "slot"}``; the inspector
      looks up the embedded aux :class:`StepNode` at that slot and
      renders ITS params instead, prefixed by a clickable breadcrumb
      banner that reverts to consumer mode.

    Args:
        state: The full builder state (used to resolve the active scope,
            selection, and aux focus via :func:`current_scope` +
            :func:`_resolve_inspector_focus_target`).
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

    consumer = next(
        (n for n in scope.nodes if n.node_id == state.selected_node_id),
        None,
    )
    if consumer is None:
        return _empty_inspector_div()

    # Decide which node's params the inspector should mirror — the
    # canvas-selected consumer, or a wired aux it points at via the
    # ``inspector_focus_aux`` override. The aux-focus path falls back
    # to the consumer when the focus can't be resolved (slot empty,
    # param gone, etc.) so a stale override never silently swallows
    # the inspector.
    focus_target = _resolve_inspector_focus_target(state, scope)
    banner: Optional[Any] = None
    if focus_target is not None:
        consumer, render_node, focus_param, focus_slot = focus_target
        banner = _inspector_focus_aux_banner(consumer, focus_param, focus_slot)
    else:
        render_node = consumer

    label_value = render_node.label or render_node.class_name
    header_children: List[Any] = [
        html.H5(render_node.class_name, className="card-title mb-3"),
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

    # Prepended only when an aux-focus override is active; empty otherwise so
    # the canvas-consumer rendering path stays unchanged.
    banner_prefix: List[Any] = [banner] if banner is not None else []

    if render_node.class_name == PIPELINE_CLASS_NAME:
        body_children: List[Any] = [
            *banner_prefix,
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
            # Hidden placeholders so the Documentation toggle callback's
            # Input/State ids resolve even on the pipeline-sentinel branch.
            *_doc_section_widgets(None),
        ]
        return html.Div(
            dbc.Card(dbc.CardBody(body_children), className="h-100"),
            id=ids.INSPECTOR_CONTAINER,
        )

    op_info = registry.get(render_node.class_name)
    if op_info is None:
        form: Any = html.Div(
            f"Unknown operation '{render_node.class_name}'. "
            "It may have been removed from the registry.",
            className="text-warning",
        )
    else:
        # The popover renderer owns aux-port slot management now; the
        # inspector form just renders the focused node's own parameters.
        form = html.Div(
            param_form(
                op_info,
                current_values=render_node.params,
                form_id_prefix=render_node.node_id,
            ),
            id=ids.INSPECTOR_PARAM_FORM,
        )

    body_children = [
        *banner_prefix,
        *header_children,
        # Documentation section is collapsed by default; emits hidden
        # placeholders carrying the same ids when ``op_info.docstring`` is
        # empty so the toggle callback's Input/State always resolve.
        *_doc_section_widgets(op_info.docstring if op_info else None),
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
# Popover content renderer
# ---------------------------------------------------------------------------


#: Pattern-match ``type`` key for popover action buttons. Wave 4
#: callbacks subscribe via ``Input({"type": _POPOVER_ACTION_TYPE,
#: "action": ALL, ...}, "n_clicks")``. Defined here (rather than in
#: ``_ids.py``) because the popover renderer is the sole producer; the
#: clientside ``aux_popover.js`` glue serialises clicks into
#: :data:`PORT_CLICK_STORE` and :data:`POPOVER_ACTION_STORE`, which is
#: where Wave 4 picks the dispatch up.
_POPOVER_ACTION_TYPE: str = "popover-action"


def _popover_action_id(
    *,
    action: str,
    target_node_id: str,
    param: str,
    slot: int,
    class_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the pattern-matching id for a popover action button.

    Args:
        action: One of ``"edit"``, ``"drill"``, ``"disconnect"``,
            ``"add_slot"``, ``"pick_class"``.
        target_node_id: Consumer node id the popover is anchored to.
        param: Name of the consumer's op-typed parameter the popover
            edits.
        slot: Slot index inside that param's slot list. List-typed
            params use the explicit slot; scalar params always use ``0``;
            "add_slot" uses ``-1`` as a synthetic sentinel because no
            slot exists yet.
        class_name: Class to wire (``"pick_class"`` only); ``None``
            otherwise.

    Returns:
        Dict pattern-matching id keyed for Wave 4 callbacks.
    """

    return {
        "type": _POPOVER_ACTION_TYPE,
        "action": action,
        "target_node_id": target_node_id,
        "param": param,
        "slot": slot,
        "class_name": class_name or "",
    }


def _popover_header(param: str, *, consumer_label: str) -> html.Div:
    """Render the popover header row (param-name title + close button).

    Args:
        param: Name of the consumer's op-typed parameter (e.g.
            ``"inoculum_detector"``).
        consumer_label: Human-readable consumer label for the title
            (e.g. ``"FilamentousFungiDetector"``).

    Returns:
        Header :class:`html.Div` matching the ``.cy-popover-header``
        rule in ``builder.css``.
    """

    return html.Div(
        [
            html.Span(
                f"{consumer_label}.{param}",
                className="cy-popover-header__title",
            ),
        ],
        className="cy-popover-header",
    )


def _popover_palette(
    compatible: List[str],
    *,
    target_node_id: str,
    param: str,
    slot: int,
) -> html.Div:
    """Render the class-pick palette grid for an empty popover slot.

    Args:
        compatible: List of class names accepted by the port's type
            contract (already filtered via
            :func:`_compatible_classes_for_port`).
        target_node_id: Consumer node id the popover is anchored to.
        param: Consumer's op-typed param name.
        slot: Slot index inside ``param``'s slot list.

    Returns:
        A :class:`html.Div` with one button per compatible class,
        matching the ``.cy-popover-palette`` rule in ``builder.css``.
    """

    buttons = [
        dbc.Button(
            cls_name,
            id=_popover_action_id(
                action="pick_class",
                target_node_id=target_node_id,
                param=param,
                slot=slot,
                class_name=cls_name,
            ),
            className="cy-popover-palette-button",
            n_clicks=0,
        )
        for cls_name in compatible
    ]
    return html.Div(buttons, className="cy-popover-palette")


def _popover_wired_row(
    aux_node: StepNode,
    *,
    target_node_id: str,
    param: str,
    slot: int,
) -> html.Div:
    """Render the wired-aux row (class name + Edit / Drill / Disconnect).

    Args:
        aux_node: Embedded aux :class:`StepNode` at
            ``consumer.aux_ports[param][slot]``.
        target_node_id: Consumer node id (anchor for the action ids).
        param: Consumer's op-typed param name.
        slot: Slot index inside ``param``'s slot list.

    Returns:
        A :class:`html.Div` matching the ``.cy-popover-wired-row`` rule
        in ``builder.css``.
    """

    cls_label = aux_node.label or aux_node.class_name
    actions = [
        dbc.Button(
            "✎ Edit",
            id=_popover_action_id(
                action="edit",
                target_node_id=target_node_id,
                param=param,
                slot=slot,
            ),
            color="secondary",
            outline=True,
            size="sm",
            n_clicks=0,
        ),
        dbc.Button(
            "Drill in →",
            id=_popover_action_id(
                action="drill",
                target_node_id=target_node_id,
                param=param,
                slot=slot,
            ),
            color="secondary",
            outline=True,
            size="sm",
            n_clicks=0,
        ),
        dbc.Button(
            "⨯ Disconnect",
            id=_popover_action_id(
                action="disconnect",
                target_node_id=target_node_id,
                param=param,
                slot=slot,
            ),
            color="danger",
            outline=True,
            size="sm",
            n_clicks=0,
        ),
    ]
    return html.Div(
        [
            html.Span(cls_label, className="cy-popover-wired-row__class-name"),
            html.Div(actions, className="cy-popover-wired-row__actions"),
        ],
        className="cy-popover-wired-row",
    )


def _popover_slot_row(
    aux_node: Optional[StepNode],
    *,
    compatible: List[str],
    target_node_id: str,
    param: str,
    slot: int,
) -> html.Div:
    """Render one row inside a list-typed popover (wired or empty).

    Each row is prefixed by a slot-index label and ends with either the
    wired-row contents (when ``aux_node`` is non-None) or an inline
    class palette (when empty).

    Args:
        aux_node: Embedded aux at this slot, or ``None`` if empty.
        compatible: Compatible-class list for empty rows.
        target_node_id: Consumer node id.
        param: Consumer's op-typed param name.
        slot: Slot index this row represents.

    Returns:
        A :class:`html.Div` matching ``.cy-popover-slot-row`` in
        ``builder.css``.
    """

    index_label = html.Span(
        f"slot {slot}", className="cy-popover-slot-row__index"
    )
    if aux_node is None:
        body: Any = _popover_palette(
            compatible,
            target_node_id=target_node_id,
            param=param,
            slot=slot,
        )
    else:
        body = _popover_wired_row(
            aux_node,
            target_node_id=target_node_id,
            param=param,
            slot=slot,
        )
    return html.Div([index_label, body], className="cy-popover-slot-row")


def build_popover_contents(
    state: BuilderState, registry: "OperationRegistry"
) -> List[Any]:
    """Build the children list for the canvas-anchored popover.

    Reads :attr:`BuilderState.inspector_focus_aux` to decide which port
    the popover is anchored to (the ``aux_popover.js`` clientside glue
    keeps the focus override in sync with the active port click). The
    contents are picked based on the param's cardinality and slot
    occupancy:

    * **Empty scalar port** (one slot, slot value is ``None``):
      header + class palette.
    * **Wired scalar port** (one slot, slot value is a StepNode):
      header + wired row (class label + Edit / Drill / Disconnect).
    * **List port** (zero or more slots): header + one slot row per
      slot (each empty or wired with per-slot actions) + ``+ Add slot``
      button.

    The popover hides itself entirely when ``state.inspector_focus_aux``
    is ``None`` — Wave 4 callbacks check the returned list and toggle
    the popover container's ``display`` accordingly.

    Args:
        state: Full :class:`BuilderState`.
        registry: Operation registry consulted for parameter metadata
            and the compatibility filter.

    Returns:
        Ordered list of components for the popover container's
        ``children``. Empty list when the popover should be hidden.
    """

    focus = state.inspector_focus_aux
    if focus is None:
        return []

    target_node_id = focus.get("target_node_id")
    param = focus.get("param")
    if not isinstance(target_node_id, str) or not isinstance(param, str):
        return []

    try:
        scope = current_scope(state)
    except KeyError:
        return []

    consumer = next(
        (n for n in scope.nodes if n.node_id == target_node_id), None
    )
    if consumer is None:
        return []

    op_info = registry.get(consumer.class_name)
    if op_info is None:
        return []

    param_info = op_info.parameters.get(param)
    if param_info is None:
        return []

    compatible = _compatible_classes_for_port(param_info, registry)
    consumer_label = consumer.label or consumer.class_name
    header = _popover_header(param, consumer_label=consumer_label)

    children: List[Any] = [header]

    if param_info.is_list:
        # List-typed port: render every existing slot as its own row,
        # plus an "+ Add slot" affordance at the bottom. We use
        # ``slot=-1`` for the add-slot button so its id is distinct from
        # any real slot's pick_class id (real slots are 0..len-1).
        slots = consumer.aux_ports.get(param) or []
        if not slots:
            # Surfacing the add-slot button is the only way to bootstrap
            # an entirely-empty list-typed port — there are no slot rows
            # yet to host a palette.
            children.append(
                html.Div(
                    "No slots yet — add one to start wiring.",
                    className="text-muted small mb-2",
                )
            )
        for slot_idx, slot_value in enumerate(slots):
            children.append(
                _popover_slot_row(
                    slot_value,
                    compatible=compatible,
                    target_node_id=target_node_id,
                    param=param,
                    slot=slot_idx,
                )
            )
        children.append(
            dbc.Button(
                "+ Add slot",
                id=_popover_action_id(
                    action="add_slot",
                    target_node_id=target_node_id,
                    param=param,
                    slot=-1,
                ),
                className="cy-popover-add-slot",
                n_clicks=0,
            )
        )
    else:
        # Scalar port: a length-1 list, so a single slot at index 0.
        slots = consumer.aux_ports.get(param) or [None]
        slot_value = slots[0]
        if slot_value is None:
            children.append(
                _popover_palette(
                    compatible,
                    target_node_id=target_node_id,
                    param=param,
                    slot=0,
                )
            )
        else:
            children.append(
                _popover_wired_row(
                    slot_value,
                    target_node_id=target_node_id,
                    param=param,
                    slot=0,
                )
            )

    return children


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
        className="pheno-breadcrumb d-inline-flex align-items-center mb-2",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def build_footer(image_root: Optional[Path]) -> dbc.Card:
    """Render the footer row with image-source, grid override, and run-preview controls.

    Pipeline I/O (Save/Load) lives in the title bar header next to the logo;
    ``+ Pipeline`` and ``Delete selected`` live with the Operations palette
    and canvas-control toolbar respectively. This footer card handles the
    image-selection and execution surface.

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
    *,
    url_prefix: str = "/",
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

    # ``+ Pipeline`` adds a nested ImagePipeline node — conceptually a
    # palette item, but kept as a sticky button above the Operations
    # accordion so it's never hidden by a collapsed section.
    new_pipeline_btn = dbc.Button(
        "+ Pipeline",
        id=ids.BTN_NEW_PIPELINE_NODE,
        color="primary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="w-100 mb-2",
        title="Add a nested ImagePipeline container to the chain",
    )
    operations_section = html.Div(
        [
            html.H6("Operations", className="mb-2"),
            new_pipeline_btn,
            build_palette(registry),
        ],
        className="pheno-palette-section",
    )

    palette_inner = html.Div(
        [
            operations_section,
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
    #   Top half    = Canvas (full width — Pipeline I/O moved to the title bar,
    #                 Delete + Pipeline relocated to the canvas / palette)
    #   Bottom half = Inspector (full width)
    # The two halves both use ``flex: 1 1 0`` so the split is exact regardless
    # of content size; ``min-height: 0`` lets each half shrink without the
    # inspector content forcing growth.
    top_half = html.Div(
        build_canvas_section(state.root, state.selected_node_id),
        style={
            "flex": "1 1 0",
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
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
            # Canvas-anchored popover event channels (Wave 4 callbacks read
            # these). ``aux_popover.js`` writes ``PORT_CLICK_STORE`` when the
            # user taps an aux port; ``POPOVER_DISMISS_STORE`` when the
            # popover should dismiss (click-outside / Escape / canvas pan);
            # and ``POPOVER_ACTION_STORE`` when an action button inside the
            # popover fires. Each store carries a monotonic timestamp so
            # repeat events on the same port still trigger change detection.
            dcc.Store(
                id=ids.PORT_CLICK_STORE,
                data=None,
            ),
            dcc.Store(
                id=ids.POPOVER_DISMISS_STORE,
                data=None,
            ),
            dcc.Store(
                id=ids.POPOVER_ACTION_STORE,
                data=None,
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
            build_point_picker_modal(),
        ]
    )

    # Pipeline I/O lives on the header right edge (right-aligned via
    # ``.pheno-app-header__io { margin-left: auto }``). Save and Load were
    # previously a vertical card in the right column; flattened here so the
    # canvas can take the freed width.
    pipeline_io = html.Div(
        dbc.ButtonGroup(
            [
                dbc.Button(
                    "Save",
                    id=ids.BTN_SAVE,
                    color="primary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                ),
                dbc.Button(
                    "Load",
                    id=ids.BTN_LOAD,
                    color="primary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                ),
            ],
            size="sm",
        ),
        className="pheno-app-header__io",
    )

    header = html.Div(
        [
            html.Img(
                src=f"{url_prefix}{SHARED_LOGO_PATH}",
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
            pipeline_io,
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
    "build_canvas_section",
    "build_inspector",
    "build_breadcrumb",
    "build_footer",
    "build_app_layout",
    "build_popover_contents",
    "INSPECTOR_FOCUS_AUX_BANNER_ID",
]

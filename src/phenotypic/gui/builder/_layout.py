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

import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    BuilderScope,
    BuilderState,
    Edge,
    StepNode,
    _DagBuilderScope,
    _DagBuilderState,
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


@functools.lru_cache(maxsize=256)
def _safe_stage(class_name: str) -> str:
    """Return a stage label for *class_name*, falling back to ``"ops"``.

    :func:`stage_of` raises ``KeyError`` for unknown classes (and we can't
    pretend a class we don't know is a measurement op). Layout code should
    degrade gracefully on stale state, so we collapse the error to ``"ops"``.

    Cached because :func:`build_canvas_elements_dag` calls this twice per
    block per render (once for the class list, once for ``data.bg``) and
    :func:`stage_of` itself walks the registry on every call. The cache
    is keyed only on ``class_name`` — stage membership is a property of
    the operation class and never changes at runtime.
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
            button_class = "text-start w-100 mb-1 palette-button"
            if op_info.is_point_pickable:
                button_children.append(
                    dbc.Badge(
                        "PICK",
                        className="shell-badge shell-badge-pickable ms-2",
                        pill=True,
                    )
                )
                button_class = f"{button_class} builder-op-pickable"
            # ``draggable`` + ``data-palette-class`` enable the HTML5
            # drag-and-drop bridge in ``assets/palette_dnd.js``.
            # ``dbc.Button`` forwards arbitrary ``**kwargs`` through to
            # the underlying ``<button>`` element so these attributes
            # survive Dash's component layer.
            buttons.append(
                dbc.Button(
                    button_children,
                    id=ids.palette_button_id(op_info.name),
                    color=_STAGE_BUTTON_OUTLINE_COLOR.get(stage, "primary"),
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className=button_class,
                    **{
                        "draggable": "true",
                        "data-palette-class": op_info.name,
                    },
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


def build_canvas_elements(
    scope: BuilderScope,
    selected_node_id: Optional[str],
) -> List[dict]:
    """Compute the cytoscape ``elements`` list for *scope*.

    Returned list is wired straight to
    ``Output(ids.CANVAS_CYTOSCAPE, "elements", allow_duplicate=True)``
    by every state-mutation callback; the ``Cytoscape`` component itself
    is mounted once at initial paint (see :func:`build_canvas`).

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
        Ordered list of cytoscape element dicts (nodes + edges + port
        markers). Layout is ``"preset"`` — Python computes ``(x, y)``
        for every element so callbacks can stay layout-agnostic.
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

    return elements


# ---------------------------------------------------------------------------
# DAG canvas elements (spec §5.5 — Phase 2 of the builder redesign)
# ---------------------------------------------------------------------------
#
# The DAG canvas renders every block + every port + every edge as a
# first-class cytoscape element.  Container blocks (``class_name ==
# PIPELINE_CLASS_NAME``) act as cytoscape compound parents so their inner
# scope's blocks visually nest inside the container's bounding box.
#
# This implementation is **layout-agnostic** — no ``position`` is emitted
# on any element.  The clientside ``viewport_ops.js`` runs a per-scope
# dagre pass at first paint and after every state mutation; the DAG
# canvas's stylesheet treats compound parents as auto-sized nodes so
# dagre's per-scope output fits inside the outer scope's pass.

#: Default label for the auto-seeded InputImage sentinel block when its
#: own ``label`` field is empty (spec §4.1).
_DAG_INPUT_IMAGE_LABEL = "Input Image"


@functools.lru_cache(maxsize=512)
def _resolve_dag_accepts_for_class_port(
    class_name: str,
    port_name: str,
    registry_id: int,
) -> Optional[Tuple[str, ...]]:
    """Cached wrapper around :func:`_resolve_dag_accepts` keyed on
    ``(class_name, port_name, id(registry))``.

    :func:`build_canvas_elements_dag` is called on every state mutation,
    and each consumer block re-resolves its aux ports' accept lists from
    the operation registry. The walk costs ~12 microseconds per port but
    is redundant — the answer is invariant for a given (class, port)
    pair on a given registry instance. Caching collapses repeated
    renders into a single resolution per (class, port).

    ``registry_id`` is :func:`id(registry)` so a registry-instance swap
    (vanishingly rare — only happens in tests via monkeypatch) bypasses
    the cache automatically. The cache returns an immutable tuple so
    callers cannot mutate the cached value.

    Returns:
        Sorted tuple of accepted class names, or ``None`` when the
        caller's registry diverges from the live registry (signals the
        caller should fall back to the uncached path). ``None`` is used
        instead of ``()`` because a real port may legitimately resolve
        to zero accepts (unresolved forward reference); the caller
        still wants that empty-answer case cached.
    """

    # Import lazily so the module's top-level import graph stays Dash-only.
    from phenotypic.gui._operation_registry import get_registry

    live = get_registry()
    if id(live) != registry_id:
        # ``id(registry)`` mismatch — the caller passed a stand-in
        # registry (the test suite monkeypatches ``get_registry``).
        # Signal a miss with ``None`` so the caller falls back to a
        # direct ``_resolve_dag_accepts`` call.
        return None

    info = live.get(class_name)
    if info is None:
        return ()
    param_info = info.parameters.get(port_name)
    if param_info is None:
        return ()
    return tuple(_resolve_dag_accepts(param_info, live))


def _resolve_dag_accepts(
    param_info: Any,
    registry: "OperationRegistry",
) -> List[str]:
    """Compute the list of registry class names compatible with an aux port.

    The DAG canvas attaches this list as ``data.accepts`` on every aux-
    port sub-node so the clientside ``wire_drawing.js`` can decide which
    ports glow vs. dim during a drag.  The algorithm
    follows spec §5.5 ``accepts`` resolution rules:

    * **``is_operation`` AND ``is_pipeline``** (the annotation is
      ``ImageOperation`` itself, satisfied by every op and by
      ``ImagePipeline``) → emit every registered op class plus the
      ``PIPELINE_CLASS_NAME`` sentinel.
    * **``is_pipeline``** alone → emit ``PIPELINE_CLASS_NAME``.
    * **``is_operation``** alone → walk the registry for every class
      whose ``cls`` is a subclass of the resolved annotation type.  The
      registry caches ``type_hint``; we walk the resolved-class subset
      to keep the function self-contained.

    The result list is sorted for deterministic test snapshots.

    Args:
        param_info: ``ParamInfo`` for the aux-eligible parameter.
        registry: Operation registry consulted for the candidate
            class universe.

    Returns:
        Sorted list of class-name strings (lower-case alphabetical).
        Empty when the annotation fails to resolve (advisory
        ``unknown_class`` surface).
    """

    from phenotypic import ImagePipeline

    # Forward reference / unresolved type → empty accepts list.
    if not (param_info.is_operation or param_info.is_pipeline):
        return []

    names: List[str] = []
    # Both flags True → the annotation is ImageOperation itself (or a
    # union that includes both branches); emit every registered op
    # plus the pipeline sentinel.
    if param_info.is_operation and param_info.is_pipeline:
        for category in registry.get_categories():
            for op_info in registry.get_by_category(category):
                names.append(op_info.name)
        names.append(PIPELINE_CLASS_NAME)
        return sorted(set(names))

    # Pipeline-only annotation.
    if param_info.is_pipeline and not param_info.is_operation:
        return [PIPELINE_CLASS_NAME]

    # is_operation only — resolve the type hint to one or more classes
    # and walk the registry for subclasses.  ``_unwrap_to_classes``
    # handles Union/Optional/List/Annotated layers and yields every
    # candidate base class.
    type_hint = param_info.type_hint
    target_classes = _unwrap_to_classes(type_hint)
    if not target_classes:
        # Couldn't resolve a class — empty accepts so all sources dim.
        return []

    for category in registry.get_categories():
        for op_info in registry.get_by_category(category):
            cls = op_info.cls
            if not isinstance(cls, type):
                continue
            for target in target_classes:
                try:
                    if issubclass(cls, target):
                        names.append(op_info.name)
                        break
                except TypeError:
                    # Defensive: cls may not be a type (e.g. metaclass)
                    continue
    # ImagePipeline IS an ImageOperation so include it when the target is
    # ImageOperation or a base it satisfies.
    for target in target_classes:
        try:
            if issubclass(ImagePipeline, target):
                names.append(PIPELINE_CLASS_NAME)
                break
        except TypeError:
            pass
    return sorted(set(names))


def _unwrap_to_classes(hint: Any) -> List[Any]:
    """Strip ``Annotated[...]`` / ``Union[..., None]`` / ``List[T]`` wrappers.

    Spec §5.5 — ``Union[A, B]`` accepts the union of per-arm accept
    sets; this helper returns every class hint discovered after
    unwrapping the typing-construct sandwich.  Returns an empty list
    when no class is recoverable (forward references / unresolved
    annotations).

    Args:
        hint: Type hint pulled from ``ParamInfo.type_hint``.

    Returns:
        List of class candidates (may contain duplicates; caller
        deduplicates).
    """

    import types as types_mod
    import typing as t
    from typing import Union

    # Annotated[T, ...] → recurse into T
    origin = t.get_origin(hint)
    args = t.get_args(hint)
    annotated_metadata = getattr(hint, "__metadata__", None)
    if annotated_metadata is not None and args:
        return _unwrap_to_classes(args[0])

    # Union / Optional → recurse into every non-None arm
    if origin is Union or origin is types_mod.UnionType:
        out: List[Any] = []
        for arg in args:
            if arg is type(None):
                continue
            out.extend(_unwrap_to_classes(arg))
        return out

    # list[T] / List[T] — recurse into T (list-ness handled by is_list flag)
    if origin is list or hint is list:
        if args:
            return _unwrap_to_classes(args[0])
        return []

    # Plain class
    if isinstance(hint, type):
        return [hint]

    return []


def _dag_block_classes(
    block: BlockNode,
    *,
    is_aux_consumed: bool,
    has_issue: bool,
    issue_severity: str,
    has_stub_issue: bool,
) -> List[str]:
    """Compute the cytoscape class list for a DAG block.

    Border styling rules per spec §4.2:

    * 1px stage-coloured border for main-flow ops (default).
    * 1.5px purple border for aux-consumed blocks.
    * 1.5px yellow border for advisory issues (stage_order_hint / unknown).
    * 2.5px solid red border for blocking issues (Rules 1-6).
    * 2.5px dashed red border for the stub case of Rule 2 (unreachable
      from Input Image).

    Args:
        block: The :class:`BlockNode` being rendered.
        is_aux_consumed: ``True`` when this block's image-out wires to an
            aux port (so it lives outside the main spine).
        has_issue: ``True`` when validation reports any issue against
            this block.
        issue_severity: ``"error"`` for blocking issues, ``"advisory"``
            for hints; only consulted when ``has_issue`` is ``True``.
        has_stub_issue: ``True`` for the specific Rule 2 stub case —
            renders as a dashed red border.

    Returns:
        List of cytoscape classes (joined by the caller).
    """

    stage = _safe_stage(block.class_name)
    classes: List[str] = ["dag-block", f"stage--{stage}"]
    if block.class_name == INPUT_IMAGE_CLASS_NAME:
        classes.append("dag-block--input-image")
    if block.class_name == PIPELINE_CLASS_NAME:
        classes.append("dag-block--container")
    if is_aux_consumed:
        classes.append("dag-block--aux-consumed")
    if has_issue:
        if issue_severity == "advisory":
            classes.append("dag-block--advisory")
        elif has_stub_issue:
            classes.append("dag-block--stub")
        else:
            classes.append("dag-block--error")
    return classes


def _aux_port_classes(
    *,
    wired: bool,
    required: bool,
    is_list: bool,
) -> List[str]:
    """Compute the cytoscape class list for an aux-port sub-node."""

    classes = ["dag-port", "dag-port--aux"]
    if wired:
        classes.append("dag-port--wired")
    elif required:
        classes.append("dag-port--required")
    if is_list:
        classes.append("dag-port--list")
    return classes


def _build_image_port_subnode(
    block_id: str, port: str, port_kind: str, *, css_class: str
) -> dict:
    """Return the cytoscape element for an image-in / image-out port.

    Centralises the dict shape so the per-block emission loop in
    :func:`build_canvas_elements_dag` stays focussed on layout logic.
    The ``is_port`` flag is read by ``viewport_ops.js``'s dagre walker
    to skip port sub-nodes during rank assignment.
    """

    return {
        "data": {
            "id": ids.block_port_id(block_id, port),
            "parent": block_id,
            "block_id": block_id,
            "port": port,
            "port_kind": port_kind,
            "is_port": True,
        },
        "classes": f"dag-port {css_class}",
        "selectable": False,
        "grabbable": False,
    }


def build_canvas_elements_dag(
    scope: "_DagBuilderScope",
    *,
    selected_block_id: Optional[str] = None,
    selected_edge_id: Optional[str] = None,
    breadcrumb: Optional[List[str]] = None,
    issues: Optional[List[Any]] = None,
) -> List[dict]:
    """Compute the cytoscape ``elements`` list for a DAG-shaped scope.

    See spec §5.5.  Renders every :class:`BlockNode` as a cytoscape
    node, every aux/image port as a cytoscape compound child of its
    parent block, and every :class:`Edge` as a cytoscape edge.
    Container blocks become compound parents whose inner blocks
    visually nest inside the container's bounding box.

    Key design notes:

    * **No positions emitted.** The clientside ``viewport_ops.js``
      runs a per-scope ``cytoscape-dagre`` pass at first paint and
      after every state mutation.  Returning positionless elements
      lets the layout owner stay clientside without server-side
      coordination.
    * **Aux ports carry ``data.accepts``.**  Each op-typed param on a
      consumer block emits one aux-port sub-node carrying
      ``data.accepts: List[str]`` — the list of registry class names
      whose annotation is compatible with that port (spec §5.5).
      ``wire_drawing.js`` reads this on dragstart to decide which
      ports glow vs. dim.
    * **Issue badges.** Per spec §4.6, each block with an issue gets a
      sub-node ``dag-issue`` carrying ``data: {rule_kind, severity,
      block_id, detail}``.  Click-to-pan wiring lives in the
      validation-badge callback.
    * **Main-path emphasis.**  Edges on the path from ``Input Image``
      to the chain's terminal carry ``data.is_main: True`` so the
      cytoscape stylesheet can render them at ``width: 3px``; aux + non-
      main edges land at ``width: 2px``.

    Args:
        scope: The :class:`_DagBuilderScope` currently in view (root or
            nested container).
        selected_block_id: ``BlockNode.block_id`` of the focused block
            (selection styling), or ``None``.
        selected_edge_id: ``Edge.edge_id`` of the focused wire, or
            ``None``.
        breadcrumb: Container ``block_id``s walked from the root to the
            current scope (reserved for cross-scope rendering; Phase 2
            renders only the active scope).
        issues: List of :class:`~phenotypic.gui.builder._validation.Issue`
            instances (or anything with ``.block_id``, ``.kind``,
            ``.severity``, ``.detail`` attributes) — drives the border
            decoration and issue-badge sub-nodes.

    Returns:
        Ordered list of cytoscape element dicts (compound parents +
        blocks + port sub-nodes + edges + issue badges).
    """

    # Local import keeps the module's top-level import graph dash-only.
    from phenotypic.gui._operation_registry import get_registry

    registry = get_registry()
    # ``registry_key`` is the cache discriminator for
    # ``_resolve_dag_accepts_for_class_port``.  Captured once per render
    # so the inner aux-port loop avoids repeated ``id(registry)`` calls.
    registry_key = id(registry)
    elements: List[dict] = []
    issues = list(issues or [])

    # Build a per-block issue index so the renderer can decorate each
    # block once.  Issues outside the active scope are silently dropped
    # (spec §4.6: nested-scope issues surface as the container's
    # aggregate badge).
    issue_by_block: Dict[str, List[Any]] = {}
    for iss in issues:
        bid = getattr(iss, "block_id", None)
        if bid is None:
            continue
        issue_by_block.setdefault(bid, []).append(iss)

    # Compute the set of blocks consumed as aux (output wires to a
    # purple aux port).  Used by the border-decoration rule.
    aux_consumed_block_ids: set[str] = set()
    # Compute the main-path edge set for the width: 3px emphasis.
    main_path_edges: set[str] = set()

    # Bucket aux wired counts by (target_block_id, target_port) up-front
    # so the port-emission loop below stays O(V + E) rather than
    # re-walking ``scope.edges`` once per block (O(V × E)).  Indexed by
    # ``target_block_id`` -> {port_name: count}.
    aux_wired_counts: Dict[str, Dict[str, int]] = {}
    image_edges_by_source: Dict[str, List[Edge]] = {}
    for edge in scope.edges:
        if edge.kind == "aux":
            aux_consumed_block_ids.add(edge.source_block_id)
            block_ports = aux_wired_counts.setdefault(
                edge.target_block_id, {}
            )
            block_ports[edge.target_port] = (
                block_ports.get(edge.target_port, 0) + 1
            )
        elif edge.kind == "image":
            image_edges_by_source.setdefault(edge.source_block_id, []).append(edge)

    # Walk image-flow forward from the Input Image to populate the main-path.
    input_block = next(
        (b for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME),
        None,
    )
    if input_block is not None:
        frontier = [input_block.block_id]
        visited: set[str] = set()
        while frontier:
            curr = frontier.pop()
            if curr in visited:
                continue
            visited.add(curr)
            for out_edge in image_edges_by_source.get(curr, []):
                main_path_edges.add(out_edge.edge_id)
                frontier.append(out_edge.target_block_id)

    # ── 1. Emit one cytoscape node per block (including the InputImage
    #       sentinel).  Container blocks have no ``parent`` field — they ARE
    #       the parent; their inner blocks get a ``data.parent = <container
    #       block_id>`` reference when rendered.  Containers render
    #       always-expanded until the collapse interaction ships.
    for block in scope.blocks:
        block_issues = issue_by_block.get(block.block_id, [])
        has_issue = bool(block_issues)
        issue_severity = "error"
        has_stub_issue = False
        if has_issue:
            # Pick worst severity; stub gets the dashed-border treatment.
            severities = {getattr(iss, "severity", "error") for iss in block_issues}
            issue_severity = "advisory" if severities == {"advisory"} else "error"
            kinds = {getattr(iss, "kind", "") for iss in block_issues}
            has_stub_issue = "stub" in kinds

        is_aux_consumed = block.block_id in aux_consumed_block_ids
        classes = _dag_block_classes(
            block,
            is_aux_consumed=is_aux_consumed,
            has_issue=has_issue,
            issue_severity=issue_severity,
            has_stub_issue=has_stub_issue,
        )
        if block.block_id == selected_block_id:
            classes.append("selected")

        # Container blocks render an expand chevron + folder glyph in
        # their label; regular blocks fall back to their label/class.
        if block.class_name == PIPELINE_CLASS_NAME:
            base_label = block.label or block.class_name
            label = f"▼ Pipeline — {base_label}"
        elif block.class_name == INPUT_IMAGE_CLASS_NAME:
            label = block.label or _DAG_INPUT_IMAGE_LABEL
        else:
            label = block.label or block.class_name

        stage = _safe_stage(block.class_name)
        node_data: Dict[str, Any] = {
            "id": block.block_id,
            "label": label,
            "block_id": block.block_id,
            "class_name": block.class_name,
            "stage": stage,
            "bg": _STAGE_COLORS.get(stage, _STAGE_COLORS["ops"]),
            "parent": None,
        }
        elements.append(
            {
                "data": node_data,
                "classes": " ".join(classes),
                "grabbable": True,
                "selectable": True,
            }
        )

    # ── 2. Emit port sub-nodes per block.  Cytoscape compound children
    #       set ``data.parent = <parent_block_id>``.  Each port carries
    #       structured data so callbacks reading ``tapNodeData`` recover
    #       the block_id + port name.
    for block in scope.blocks:
        if block.class_name != INPUT_IMAGE_CLASS_NAME:
            elements.append(
                _build_image_port_subnode(
                    block.block_id, "in", "image-in",
                    css_class="dag-port--input",
                )
            )
        # The InputImage sentinel still gets an output port — it's the
        # source of the main spine.
        elements.append(
            _build_image_port_subnode(
                block.block_id, "out", "image-out",
                css_class="dag-port--output",
            )
        )

        # Aux ports: one per op-typed parameter on the consumer (block).
        if block.class_name in (INPUT_IMAGE_CLASS_NAME, PIPELINE_CLASS_NAME):
            continue
        info = registry.get(block.class_name)
        if info is None:
            continue
        # Wired counts were bucketed in the initial ``scope.edges`` walk
        # above (avoids the per-block O(E) re-scan that this loop used
        # to do).
        wired_count_by_port = aux_wired_counts.get(block.block_id, {})
        for param_name, param_info in info.parameters.items():
            if not (param_info.is_operation or param_info.is_pipeline):
                continue
            cached_accepts = _resolve_dag_accepts_for_class_port(
                block.class_name, param_name, registry_key
            )
            if cached_accepts is None:
                # Cache signalled a registry-id mismatch (tests
                # monkeypatch ``get_registry``); fall back to a direct
                # uncached resolve.
                accepts: List[str] = _resolve_dag_accepts(
                    param_info, registry
                )
            else:
                accepts = list(cached_accepts)
            wired_count = wired_count_by_port.get(param_name, 0)
            wired = wired_count > 0
            required = not param_info.has_default
            classes = _aux_port_classes(
                wired=wired,
                required=required,
                is_list=param_info.is_list,
            )
            elements.append(
                {
                    "data": {
                        "id": ids.block_port_id(block.block_id, param_name),
                        "parent": block.block_id,
                        "block_id": block.block_id,
                        "port": param_name,
                        "port_kind": "aux",
                        # ``is_port`` is read by ``viewport_ops.js`` to
                        # skip ports during the dagre rank assignment.
                        "is_port": True,
                        "is_list": param_info.is_list,
                        "is_required": required,
                        "accepts": accepts,
                        "wired_count": wired_count,
                    },
                    "classes": " ".join(classes),
                    "selectable": False,
                    "grabbable": False,
                }
            )

    # ── 3. Emit one cytoscape edge per :class:`Edge`.  Edges carry
    #       ``data.kind`` (image|aux) + ``data.target_slot`` so the
    #       stylesheet can pick blue-solid (image) vs purple-dashed
    #       (aux); ``wire_drawing.js`` reads the slot during drag-replace
    #       gestures.
    for edge in scope.edges:
        edge_classes = ["dag-wire"]
        if edge.kind == "image":
            edge_classes.append("dag-wire--image")
        else:
            edge_classes.append("dag-wire--aux")
        if edge.edge_id in main_path_edges:
            edge_classes.append("dag-wire--main")
        if edge.edge_id == selected_edge_id:
            edge_classes.append("selected")
        elements.append(
            {
                "data": {
                    "id": ids.edge_id(edge.edge_id),
                    "source": edge.source_block_id,
                    "target": edge.target_block_id,
                    "edge_id": edge.edge_id,
                    "kind": edge.kind,
                    "target_slot": edge.target_slot,
                    "target_port": edge.target_port,
                    "source_port": edge.source_port,
                    "is_main": edge.edge_id in main_path_edges,
                },
                "classes": " ".join(edge_classes),
                "selectable": True,
                "grabbable": False,
            }
        )

    # ── 4. Emit issue-badge sub-nodes.  Each block with one or more
    #       issues gets one compound child badge whose ``data`` lists
    #       the rule kind / severity / detail (badge collapses multiple
    #       issues into one row — the tooltip lists them all).
    for block_id, block_issues in issue_by_block.items():
        if not block_issues:
            continue
        first = block_issues[0]
        severity = getattr(first, "severity", "error")
        kind = getattr(first, "kind", "")
        detail = getattr(first, "detail", "")
        elements.append(
            {
                "data": {
                    "id": f"issue__{block_id}",
                    "parent": block_id,
                    "block_id": block_id,
                    "rule_kind": kind,
                    "severity": severity,
                    "detail": detail,
                    "count": len(block_issues),
                },
                "classes": (
                    "dag-issue dag-issue--advisory"
                    if severity == "advisory"
                    else "dag-issue dag-issue--error"
                ),
                "selectable": True,
                "grabbable": False,
            }
        )

    return elements


def build_asset_status_banner() -> html.Div:
    """Thin row above the canvas; surfaces missing-asset messages.

    Subscribes to :data:`STORE_ASSET_STATUS` via Phase 2 callbacks.  Each
    missing JS asset renders one row inside the banner:

    * ``wire_drawing.js`` → *"Wire drawing offline"*
    * ``palette_dnd.js`` → *"Block creation offline — drag from the
      palette is unavailable"*
    * ``viewport_ops.js`` → *"Layout offline"*
    * ``cytoscape-dagre`` extension missing → *"Layout extension
      missing"*

    The row is hidden when all assets report ready.  Phase 2 mounts the
    container with ``children=[]`` and the ``asset_status_disables``
    callback populates / hides it within ~1500ms of page load (the
    asset-readiness poll interval is 500ms; three poll cycles cover the
    expected JS download window).

    Returns:
        A :class:`dash.html.Div` carrying id :data:`BANNER_ASSET_STATUS`,
        hidden by default and replaced wholesale by the
        ``asset_status_disables`` callback when assets are missing.
    """

    return html.Div(
        id=ids.BANNER_ASSET_STATUS,
        className="builder-asset-status",
        style={"display": "none"},
        children=[],
    )


def build_confirm_delete_modal() -> dbc.Modal:
    """Lightweight confirm-delete modal for non-empty container blocks.

    Mounted once at app boot inside ``build_app_layout``; visibility is
    driven by ``STORE_BUILDER_STATE.pending_delete_block_id`` — setting
    the field on a non-empty container opens the modal, Confirm
    dispatches ``block_delete_confirm``, and Cancel clears the field.

    This builder emits the modal scaffold only; the body label + the
    inner-block count are filled in by the open-modal callback when
    ``pending_delete_block_id`` resolves.  The Confirm / Cancel
    buttons carry stable ids :data:`BTN_CONFIRM_DELETE` /
    :data:`BTN_CANCEL_DELETE`.

    Returns:
        A :class:`dash_bootstrap_components.Modal` ready to embed in
        the app's modal mount.
    """

    return dbc.Modal(
        id=ids.CONFIRM_DELETE_MODAL_ID,
        is_open=False,
        size="sm",
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Confirm delete")),
            dbc.ModalBody(
                # The open-modal callback rewrites the children to:
                #   "Delete container '<label>' and its N inner block(s)?"
                # at dispatch time.  Empty default keeps the modal
                # renderable as a layout scaffold.
                html.Div(id=f"{ids.CONFIRM_DELETE_MODAL_ID}-body"),
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id=ids.BTN_CANCEL_DELETE,
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Delete",
                        id=ids.BTN_CONFIRM_DELETE,
                        color="danger",
                        n_clicks=0,
                    ),
                ]
            ),
        ],
    )


def build_canvas(
    scope: BuilderScope,
    selected_node_id: Optional[str],
) -> cyto.Cytoscape:
    """Mount the ``Cytoscape`` component with initial elements + stable props.

    Only used by :func:`build_canvas_section` at initial paint — every
    subsequent mutation updates ``Output(ids.CANVAS_CYTOSCAPE, "elements")``
    directly, so the same cytoscape.js instance persists for the session.
    Layout / stylesheet / interaction flags are set once here.

    ``_canvas_stylesheet()`` is parameterless and takes no state input;
    if a future change makes it state-dependent (theme switch, view modes),
    introduce a dedicated ``Output(ids.CANVAS_CYTOSCAPE, "stylesheet",
    allow_duplicate=True)`` callback rather than rebuilding this component.
    """

    return cyto.Cytoscape(
        id=ids.CANVAS_CYTOSCAPE,
        elements=build_canvas_elements(scope, selected_node_id),
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

    # ``Re-layout`` re-runs the dagre pass via ``viewport_ops.js``.
    # The button is mounted on every render path regardless of the
    # ``PHENOTYPIC_GUI_DAG`` flag so ``asset_status_disables``'s output
    # always resolves.  When the flag is off the button stays visually
    # inert (clicks no-op because ``viewport_ops.js`` only binds its
    # listener when ``window.phenotypicGuiDag`` is true).
    relayout_btn = dbc.Button(
        "Re-layout",
        id=ids.BTN_RELAYOUT,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Re-run the dagre layout pass and fit to viewport",
    )

    header = html.Div(
        [
            html.H6("Canvas", className="mb-0"),
            html.Div(
                [controls, delete_btn, relayout_btn],
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
    # The slot has a stable id so ``window.phenoGetCy()`` can anchor its
    # React fiber walk to find the live cytoscape.js instance. The
    # ``Cytoscape`` component is mounted once at initial paint and
    # persists for the session; mutation callbacks update its
    # ``elements`` prop directly via
    # ``Output(ids.CANVAS_CYTOSCAPE, "elements", allow_duplicate=True)``,
    # with a clientside callback in ``_callbacks.py`` mirroring the new
    # list into the live cytoscape via ``cy.json({elements})``.
    #
    # The popover container is a SIBLING of the cytoscape slot inside the
    # same relative-positioned outer Div so ``cytoscape-popper`` (via
    # popper.js) can position it relative to the tapped aux port and
    # pan / zoom along with the canvas. The container is hidden by
    # default; ``aux_popover.js`` toggles its ``display`` on tap and
    # writes the structured tap data to :data:`PORT_CLICK_STORE` so
    # server-side callbacks fill in its children based on
    # ``state.inspector_focus_aux`` + the active port click.
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
    # Asset-status banner sits between the header and the canvas slot.
    # Mounted on every render path so the ``asset_status_disables``
    # callback's Output resolves; visibility is driven by the callback
    # (hidden when all assets ready).
    banner = build_asset_status_banner()
    return html.Div(
        [header, banner, cytoscape_slot, popover_container],
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


# ---------------------------------------------------------------------------
# DAG inspector helpers (Phase 4: wire card + aux ports section)
# ---------------------------------------------------------------------------


def _dag_block_display_label(block: "BlockNode") -> str:
    """Return the user-facing label for a DAG block.

    Mirrors the rendering rule in :func:`build_canvas_elements_dag` (label
    falls back to ``class_name``) so the inspector and the canvas show the
    same identifier per block.

    Args:
        block: The :class:`BlockNode` to label.

    Returns:
        Label string.  Defaults to the class name when ``block.label`` is
        ``None`` or empty.
    """

    return block.label if block.label else block.class_name


def _find_dag_block_in_scope(
    scope: "_DagBuilderScope", block_id: str
) -> Optional["BlockNode"]:
    """Return the block in *scope* with a matching ``block_id``.

    Args:
        scope: The :class:`_DagBuilderScope` to search.
        block_id: Identifier to match.

    Returns:
        The matching :class:`BlockNode`, or ``None`` if no block in the
        scope has that ``block_id``.
    """

    return next((b for b in scope.blocks if b.block_id == block_id), None)


def _wire_target_port_label(edge: "Edge") -> str:
    """Format the target-port label for the wire-card header.

    Image-flow edges always land on ``"in"``; aux edges land on the
    consumer's parameter name (with a ``[i]`` suffix for list-aux slots
    so the user can spot which slot the wire targets at a glance).

    Args:
        edge: The :class:`Edge` whose target port to render.

    Returns:
        Port label string ready to slot into ``"<target>.<port>"``.
    """

    if edge.target_slot is None:
        return edge.target_port
    return f"{edge.target_port}[{edge.target_slot}]"


def _build_wire_card(
    scope: "_DagBuilderScope", edge: "Edge"
) -> dbc.Card:
    """Render the inspector wire card for a selected :class:`Edge`.

    Spec §4.5 calls for: source label → target.port label, edge kind
    badge (image flow vs aux assignment), and a ``Disconnect`` button.
    The label resolution uses :func:`_dag_block_display_label` so users
    see the same identifier as on the canvas.

    Args:
        scope: The active :class:`_DagBuilderScope`.  Used to resolve the
            edge's source/target ``block_id``s to their ``BlockNode``
            instances.
        edge: The :class:`Edge` to render.  Caller is responsible for
            ensuring ``edge`` belongs to *scope*.

    Returns:
        :class:`dbc.Card` wrapping the wire-card body.  The wrapper carries
        :data:`INSPECTOR_WIRE_CARD` so callbacks can refresh its contents
        on selection change; the ``Disconnect`` button carries the
        pattern-matching id returned by
        :func:`phenotypic.gui.builder._ids.inspector_disconnect_id`.
    """

    source_block = _find_dag_block_in_scope(scope, edge.source_block_id)
    target_block = _find_dag_block_in_scope(scope, edge.target_block_id)
    source_label = (
        _dag_block_display_label(source_block)
        if source_block is not None
        else edge.source_block_id
    )
    target_label = (
        _dag_block_display_label(target_block)
        if target_block is not None
        else edge.target_block_id
    )
    target_port_label = _wire_target_port_label(edge)
    kind_label = "image flow" if edge.kind == "image" else "aux assignment"

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Wire", className="card-title mb-2"),
                html.Div(
                    f"{source_label} → {target_label}.{target_port_label}",
                    className="inspector-wire-summary mb-2",
                    style={
                        "fontFamily": FONT_FAMILY_MONO,
                        "fontSize": FONT_SIZE_LABEL,
                    },
                ),
                dbc.Badge(
                    kind_label,
                    color="primary" if edge.kind == "image" else "secondary",
                    className="mb-3",
                ),
                dbc.Button(
                    "Disconnect",
                    id=ids.inspector_disconnect_id(edge.edge_id),
                    color="danger",
                    outline=True,
                    n_clicks=0,
                ),
            ]
        ),
        id=ids.INSPECTOR_WIRE_CARD,
        className="h-100",
    )


def _aux_typed_params(
    info: Any,
) -> List[Tuple[str, Any]]:
    """Return the ``(name, ParamInfo)`` pairs of op-typed parameters.

    Walks ``info.parameters`` and keeps only entries whose annotation
    accepts an :class:`~phenotypic.abc_.ImageOperation` subclass or an
    :class:`~phenotypic.ImagePipeline` instance — the same predicate
    used by :func:`build_canvas_elements_dag` to decide which params get
    aux-port sub-nodes.  Order matches the dict iteration order so the
    inspector renders params in their signature order.

    Args:
        info: Either an :class:`OperationInfo` instance or ``None``.

    Returns:
        List of ``(name, ParamInfo)`` tuples; empty when *info* is
        ``None`` or has no aux-typed params.
    """

    if info is None:
        return []
    out: List[Tuple[str, Any]] = []
    for name, param_info in info.parameters.items():
        if param_info.is_operation or param_info.is_pipeline:
            out.append((name, param_info))
    return out


def _aux_port_required_tag(required: bool) -> dbc.Badge:
    """Render the small required/optional badge next to an aux-port name.

    Args:
        required: ``True`` when the underlying param has no default.

    Returns:
        :class:`dbc.Badge` styled as ``danger`` for required ports and
        ``secondary`` for optional ones.
    """

    label = "required" if required else "optional"
    color = "danger" if required else "secondary"
    return dbc.Badge(label, color=color, className="ms-2", pill=True)


def _format_param_type_hint(param_info: Any) -> str:
    """Format an op-typed param's annotation for the aux-port row header.

    The full ``typing`` repr is verbose (e.g.
    ``typing.Optional[typing.List[phenotypic.abc_.ImageOperation]]``);
    the inspector instead shows the friendlier ``list[op]`` /
    ``op | None`` shorthand so the section stays scannable.

    Args:
        param_info: A :class:`ParamInfo` instance.

    Returns:
        Short annotation label.  Returns ``""`` when nothing useful can
        be derived (defensive).
    """

    base = "op" if param_info.is_operation else ""
    if param_info.is_pipeline:
        base = "pipeline" if not base else f"{base} | pipeline"
    if not base:
        return ""
    if param_info.is_list:
        base = f"list[{base}]"
    if param_info.is_optional:
        base = f"{base} | None"
    return base


def _block_is_container(block: Optional["BlockNode"]) -> bool:
    """Return ``True`` when *block* is a :data:`PIPELINE_CLASS_NAME` container.

    Used by the aux ports section to decide whether to show a ``Drill in
    →`` affordance next to a wired source.  Container blocks own a
    nested scope and can be drilled into; ordinary op sources cannot.
    """

    return (
        block is not None
        and block.class_name == PIPELINE_CLASS_NAME
        and block.nested is not None
    )


def _build_aux_scalar_row(
    *,
    param_name: str,
    param_info: Any,
    wired_edges: List["Edge"],
    scope: "_DagBuilderScope",
) -> html.Div:
    """Render one scalar aux-port row inside the aux ports section.

    The scalar variant shows either:

    * **Empty** placeholder when the param has no incoming edge.
    * **Wired** row: the source block's class label, a ``Disconnect``
      button (matching ``BTN_INSPECTOR_DISCONNECT``), and a ``Drill in
      →`` button when the source is a container.

    Args:
        param_name: Parameter name (also the edge ``target_port``).
        param_info: :class:`ParamInfo` for *param_name*.
        wired_edges: Edges in *scope* targeting this scalar port (length
            0 or 1; multiple wires on a scalar port are invalid per
            Rule 1 and surface as red borders, not as multiple rows).
        scope: The active :class:`_DagBuilderScope`.

    Returns:
        :class:`html.Div` carrying a row header + the wired/empty body.
    """

    required = not param_info.has_default
    type_label = _format_param_type_hint(param_info)
    header = html.Div(
        [
            html.Strong(param_name),
            html.Span(
                f"  ·  {type_label}" if type_label else "",
                className="text-muted ms-1",
                style={"fontSize": FONT_SIZE_LABEL},
            ),
            _aux_port_required_tag(required),
        ],
        className="d-flex align-items-center mb-1",
    )

    if not wired_edges:
        body: Any = html.Div(
            "Empty",
            className="text-muted fst-italic small",
            style={"padding": "0.25rem 0.5rem"},
        )
    else:
        edge = wired_edges[0]
        source_block = _find_dag_block_in_scope(scope, edge.source_block_id)
        source_label = (
            _dag_block_display_label(source_block)
            if source_block is not None
            else edge.source_block_id
        )
        action_buttons: List[Any] = [
            dbc.Button(
                "Disconnect",
                id=ids.inspector_disconnect_id(edge.edge_id),
                color="danger",
                outline=True,
                size="sm",
                n_clicks=0,
                className="ms-2",
            )
        ]
        if source_block is not None and _block_is_container(source_block):
            # Drill-in is implemented in Phase 5 (container expand/
            # collapse + drill-in).  Mount the button so the affordance
            # is visible but disable it until the Phase 5 callback
            # lands, with a tooltip explaining the gating.
            action_buttons.append(
                dbc.Button(
                    "Drill in →",
                    id={
                        "type": "btn-inspector-drill-in-aux",
                        "block_id": source_block.block_id,
                    },
                    color="primary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className="ms-2",
                    disabled=True,
                    title="Container drill-in lands in Phase 5",
                )
            )
        body = html.Div(
            [
                html.Span(source_label, className="me-2"),
                *action_buttons,
            ],
            className="d-flex align-items-center",
        )

    return html.Div(
        [header, body],
        className="inspector-aux-row mb-3",
    )


def _build_aux_list_row(
    *,
    block: "BlockNode",
    param_name: str,
    param_info: Any,
    wired_edges: List["Edge"],
    scope: "_DagBuilderScope",
) -> html.Div:
    """Render one list-aux row inside the aux ports section.

    Spec §4.5 calls for: drag-handles, badge numbers, source class
    labels, ``✕`` remove buttons per row, plus a ``+ Add empty slot``
    affordance.  Phase 4 ships the row with ``▲`` / ``▼`` arrow
    buttons as a drag-handle fallback (drag glue lands in a follow-up
    phase per the prompt); the hidden reorder ``dcc.Store`` is mounted
    so the future drag handlers don't churn the inspector callback
    surface.

    Empty slots are tracked on ``block.list_slot_counts``; the row
    renders ``[0, count)`` positions and fills the wired-edge slots
    first, then pads with "Empty" placeholders.

    Args:
        block: The selected :class:`BlockNode` (carries
            ``list_slot_counts``).
        param_name: List-typed parameter name.
        param_info: :class:`ParamInfo` for *param_name*.
        wired_edges: Edges in *scope* targeting ``(block.block_id,
            param_name)``.  Order may be arbitrary; this helper sorts by
            ``target_slot`` to render in slot order.
        scope: The active :class:`_DagBuilderScope`.

    Returns:
        :class:`html.Div` carrying the row header + ordered list body +
        ``+ Add empty slot`` button + hidden reorder store.
    """

    required = not param_info.has_default
    type_label = _format_param_type_hint(param_info)

    header = html.Div(
        [
            html.Strong(param_name),
            html.Span(
                f"  ·  {type_label}" if type_label else "",
                className="text-muted ms-1",
                style={"fontSize": FONT_SIZE_LABEL},
            ),
            _aux_port_required_tag(required),
        ],
        className="d-flex align-items-center mb-1",
    )

    # Order edges by their declared slot index so the row order matches
    # the canvas badge numbering.
    sorted_edges = sorted(
        wired_edges,
        key=lambda e: (e.target_slot if e.target_slot is not None else 0),
    )
    slot_count = max(
        int(block.list_slot_counts.get(param_name, 0)),
        len(sorted_edges),
    )

    # Build slot occupancy: slots[i] is either an Edge or None for empty.
    slots: List[Optional["Edge"]] = [None] * slot_count
    for edge in sorted_edges:
        idx = edge.target_slot if edge.target_slot is not None else 0
        if 0 <= idx < slot_count:
            slots[idx] = edge

    edge_id_order = [e.edge_id if e is not None else None for e in slots]

    row_children: List[Any] = []
    for i, slot_entry in enumerate(slots):
        is_first = i == 0
        is_last = i == len(slots) - 1
        # Up/down arrow buttons gate themselves by position; both share
        # the same pattern-matching id family so a single callback can
        # handle reorder.  Empty slots use a synthetic id sentinel so
        # the button still mounts (we disable it via ``disabled`` so
        # the user can't move an empty row).
        slot_edge_id = (
            slot_entry.edge_id if slot_entry is not None else f"empty:{i}"
        )
        up_btn = dbc.Button(
            "▲",
            id=ids.inspector_list_move_id(slot_edge_id, "up"),
            color="link",
            size="sm",
            n_clicks=0,
            disabled=is_first or slot_entry is None,
            className="p-0 me-1",
            title="Move up",
        )
        down_btn = dbc.Button(
            "▼",
            id=ids.inspector_list_move_id(slot_edge_id, "down"),
            color="link",
            size="sm",
            n_clicks=0,
            disabled=is_last or slot_entry is None,
            className="p-0 me-2",
            title="Move down",
        )
        # Drag-handle placeholder (spec §4.5 calls for one).  Phase 4
        # ships the arrow-button fallback; the handle carries the
        # ``inspector-drag-handle`` class so a follow-up phase can
        # attach HTML5 dnd glue without touching the inspector layout.
        # ``data-edge-id`` is set via ``data_attributes`` keyword which
        # Dash forwards as an ``html-data-*`` attribute on the DOM
        # element.  Bare ``data-edge-id=`` would type-check fine but
        # ``html.Span`` only accepts dict-keyed kwargs through ``**``,
        # which loses mypy narrowing — so we stash the id on the
        # className/title instead and let a future drag-glue phase
        # add a proper data-* attribute.
        drag_handle = html.Span(
            "☰",
            className="inspector-drag-handle me-2",
            style={"cursor": "grab", "color": COLOR_MUTED},
            title=(
                f"Row {i}"
                + (
                    f" (edge {slot_entry.edge_id})"
                    if slot_entry is not None
                    else ""
                )
            ),
        )
        badge = dbc.Badge(
            str(i),
            color="light",
            text_color="dark",
            className="me-2",
        )
        if slot_entry is None:
            label_node: Any = html.Span(
                "Empty",
                className="text-muted fst-italic",
            )
            action_buttons: List[Any] = []
        else:
            source_block = _find_dag_block_in_scope(
                scope, slot_entry.source_block_id
            )
            source_label = (
                _dag_block_display_label(source_block)
                if source_block is not None
                else slot_entry.source_block_id
            )
            label_node = html.Span(source_label, className="me-2")
            action_buttons = [
                dbc.Button(
                    "✕",
                    id=ids.inspector_list_remove_id(slot_entry.edge_id),
                    color="danger",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className="ms-1",
                    title="Remove",
                )
            ]
        row_children.append(
            html.Div(
                [
                    drag_handle,
                    up_btn,
                    down_btn,
                    badge,
                    label_node,
                    *action_buttons,
                ],
                className=(
                    f"inspector-aux-list-row inspector-aux-slot-{i} "
                    "d-flex align-items-center py-1 px-2 mb-1"
                ),
                style={
                    "border": f"1px solid {COLOR_BORDER}",
                    "borderRadius": "4px",
                    "background": "#fff",
                },
            )
        )

    add_slot_btn = dbc.Button(
        "+ Add empty slot",
        id=ids.inspector_add_empty_slot_id(block.block_id, param_name),
        color="primary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="mt-1",
    )

    # Hidden reorder sink — written by future drag glue, consumed by the
    # ``list_aux_reorder`` dispatcher.  Initial data carries the current
    # ordering so the consumer can diff cheaply.
    reorder_store = dcc.Store(
        id=ids.inspector_list_reorder_store_id(block.block_id, param_name),
        data={"edge_id_order": edge_id_order},
        storage_type="memory",
    )

    return html.Div(
        [
            header,
            html.Div(row_children, className="inspector-aux-list mb-1"),
            add_slot_btn,
            reorder_store,
        ],
        className="inspector-aux-row mb-3",
    )


def _build_aux_ports_section(
    *,
    block: "BlockNode",
    scope: "_DagBuilderScope",
    registry: "OperationRegistry",
) -> Optional[html.Div]:
    """Render the per-block aux ports section.

    Returns ``None`` when the block has no aux-typed parameters (either
    because it's the ``InputImage`` sentinel, a container, or an op
    whose signature has no op/pipeline-typed params).  The caller
    suppresses the wrapping ``html.Div`` in that case so the inspector
    layout doesn't include an empty section.

    Args:
        block: The selected :class:`BlockNode`.
        scope: The active :class:`_DagBuilderScope` (used to enumerate
            edges targeting this block).
        registry: The :class:`OperationRegistry`.

    Returns:
        :class:`html.Div` keyed by :data:`INSPECTOR_AUX_SECTION`, or
        ``None`` when nothing aux-related applies.
    """

    if block.class_name in (INPUT_IMAGE_CLASS_NAME, PIPELINE_CLASS_NAME):
        return None
    info = registry.get(block.class_name)
    if info is None:
        return None
    aux_params = _aux_typed_params(info)
    if not aux_params:
        return None

    # Bucket aux edges targeting (block.block_id, param) once so each
    # row's enumeration stays O(E_param) instead of re-walking
    # ``scope.edges`` per row.
    edges_by_port: Dict[str, List[Edge]] = {}
    for edge in scope.edges:
        if edge.kind != "aux" or edge.target_block_id != block.block_id:
            continue
        edges_by_port.setdefault(edge.target_port, []).append(edge)

    rows: List[Any] = []
    for param_name, param_info in aux_params:
        wired = edges_by_port.get(param_name, [])
        if param_info.is_list:
            rows.append(
                _build_aux_list_row(
                    block=block,
                    param_name=param_name,
                    param_info=param_info,
                    wired_edges=wired,
                    scope=scope,
                )
            )
        else:
            rows.append(
                _build_aux_scalar_row(
                    param_name=param_name,
                    param_info=param_info,
                    wired_edges=wired,
                    scope=scope,
                )
            )

    return html.Div(
        [
            html.H6("Aux ports", className="mt-3 mb-2"),
            *rows,
        ],
        id=ids.INSPECTOR_AUX_SECTION,
        className="inspector-aux-section",
    )


def _find_dag_edge_in_scope(
    scope: "_DagBuilderScope", edge_id: str
) -> Optional["Edge"]:
    """Locate an :class:`Edge` by id within a single scope (no recursion).

    The inspector only renders the active scope's wire card, so this
    helper deliberately does not recurse into nested container scopes —
    if the selected edge belongs to a different scope, the caller falls
    back to the empty-state placeholder (mirrors the behaviour of
    container drill-in once the breadcrumb has shifted).
    """

    return next((e for e in scope.edges if e.edge_id == edge_id), None)


def _empty_state_card_for_dag() -> dbc.Card:
    """Friendly placeholder for the DAG inspector empty state.

    Spec §4.5 calls for a small intro card describing the validation
    badge alongside the "drag from palette" hint.  Used when no
    block / wire is selected so a user opening a fresh canvas can
    orient themselves without exploring the toolbar first.
    """

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Inspector", className="card-title"),
                html.P(
                    "Drag an operation from the palette to begin.",
                    className="mb-1",
                ),
                html.P(
                    "The toolbar issue badge tells you when the canvas "
                    "is ready to run — it shows “0 issues” "
                    "for a clean pipeline.",
                    className="text-muted mb-0",
                    style={"fontSize": FONT_SIZE_LABEL},
                ),
            ]
        ),
        className="h-100",
    )


def _build_dag_inspector(
    state: "_DagBuilderState",
    registry: "OperationRegistry",
) -> html.Div:
    """Render the inspector for a DAG-shaped :class:`BuilderState`.

    Selection mutual exclusion (spec §4.5): at most one of ``block``,
    ``wire``, or ``container`` is selected at a time.  The dispatchers
    in ``_callbacks.py`` enforce this server-side; this renderer trusts
    that invariant and short-circuits on whichever id is set:

    * ``selected_edge_id`` set → render :func:`_build_wire_card`.
    * ``selected_block_id`` set → render block label + param form +
      aux ports section (when the block has aux-typed params).
    * Neither set → render :func:`_empty_state_card_for_dag`.

    The function always returns an :class:`html.Div` keyed by
    :data:`INSPECTOR_CONTAINER` plus the hidden inspector widgets so
    the existing fan-in callback's :data:`INPUT_NODE_LABEL` /
    :data:`BTN_DRILL_IN` / :data:`INSPECTOR_DOC_TOGGLE` inputs always
    resolve.

    Args:
        state: A DAG-shaped :class:`_DagBuilderState`.
        registry: The :class:`OperationRegistry`.

    Returns:
        :class:`html.Div` wrapping the inspector card.
    """

    # Resolve active scope.  Stale breadcrumb (e.g. the container the
    # user drilled into got deleted by another callback) collapses to
    # the empty state.  ``current_scope`` is typed for the legacy
    # state shape but works structurally on DAG state too — the
    # ``# type: ignore`` annotations below acknowledge that the
    # dispatch above guarantees a DAG-shaped scope here.
    try:
        scope: "_DagBuilderScope" = current_scope(state)  # type: ignore[arg-type,assignment]
    except KeyError:
        return html.Div(
            [_empty_state_card_for_dag(), *_hidden_inspector_widgets()],
            id=ids.INSPECTOR_CONTAINER,
        )

    # ── Wire selection takes precedence; ``selected_edge_id`` and
    #    ``selected_block_id`` are mutually exclusive per spec §4.5,
    #    but defense in depth: if both happen to be set (e.g. via a
    #    hand-crafted state dict in tests), prefer the wire card so
    #    the user can still disconnect.
    edge = (
        _find_dag_edge_in_scope(scope, state.selected_edge_id)
        if state.selected_edge_id
        else None
    )
    if edge is not None:
        return html.Div(
            [_build_wire_card(scope, edge), *_hidden_inspector_widgets()],
            id=ids.INSPECTOR_CONTAINER,
        )

    block = (
        _find_dag_block_in_scope(scope, state.selected_block_id)
        if state.selected_block_id
        else None
    )
    if block is None:
        return html.Div(
            [_empty_state_card_for_dag(), *_hidden_inspector_widgets()],
            id=ids.INSPECTOR_CONTAINER,
        )

    # ── Block selection: label + (optional) param form + aux section.
    label_value = block.label or block.class_name
    header_children: List[Any] = [
        html.H5(block.class_name, className="card-title mb-3"),
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

    # Input Image gets a dedicated info card (spec §4.5; the Re-layout
    # / Re-anchor buttons are deferred to Phase 6, flagged below).
    if block.class_name == INPUT_IMAGE_CLASS_NAME:
        body_children: List[Any] = [
            *header_children,
            html.P(
                "Every op chain starts here. The image flowing out of "
                "this block is whatever your run-time loader provides.",
                className="text-muted small",
            ),
            # Re-layout / Re-anchor buttons are Phase 6 work; hidden
            # placeholders keep the existing fan-in callbacks (BTN_DRILL_IN
            # etc.) wired up.
            *_doc_section_widgets(None),
            dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
        ]
        return html.Div(
            dbc.Card(dbc.CardBody(body_children), className="h-100"),
            id=ids.INSPECTOR_CONTAINER,
        )

    # Container blocks render a drill-in affordance + nested-scope
    # summary; param form is suppressed (containers have no scalar params).
    if block.class_name == PIPELINE_CLASS_NAME:
        nested_len = (
            len(block.nested.blocks) if block.nested is not None else 0
        )
        body_children = [
            *header_children,
            html.P(
                f"Nested scope: {nested_len} block(s).",
                className="text-muted small",
            ),
            dbc.Button(
                "Drill in →",
                id=ids.BTN_DRILL_IN,
                color="primary",
                outline=True,
                n_clicks=0,
            ),
            html.Hr(),
            html.Div(id=ids.INSPECTOR_PARAM_FORM),
            html.Div(id=ids.INSPECTOR_PREVIEW, className="mt-3"),
            *_doc_section_widgets(None),
        ]
        return html.Div(
            dbc.Card(dbc.CardBody(body_children), className="h-100"),
            id=ids.INSPECTOR_CONTAINER,
        )

    # Ordinary op block: param form + aux ports section + hidden
    # placeholders for the doc-section / drill-in widgets so the
    # existing fan-in callback always resolves its inputs.
    op_info = registry.get(block.class_name)
    if op_info is None:
        form: Any = html.Div(
            f"Unknown operation '{block.class_name}'. "
            "It may have been removed from the registry.",
            className="text-warning",
        )
    else:
        form = html.Div(
            param_form(
                op_info,
                current_values=block.params,
                form_id_prefix=block.block_id,
            ),
            id=ids.INSPECTOR_PARAM_FORM,
        )

    aux_section = _build_aux_ports_section(
        block=block, scope=scope, registry=registry
    )

    body_children = [
        *header_children,
        *_doc_section_widgets(op_info.docstring if op_info else None),
        form,
    ]
    if aux_section is not None:
        body_children.append(aux_section)
    body_children.extend(
        [
            html.Hr(),
            html.Div(
                "(Run preview to populate)",
                id=ids.INSPECTOR_PREVIEW,
                className="text-muted small fst-italic",
            ),
            dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
        ]
    )
    return html.Div(
        dbc.Card(dbc.CardBody(body_children), className="h-100"),
        id=ids.INSPECTOR_CONTAINER,
    )


def build_inspector(
    state: BuilderState,
    registry: "OperationRegistry",
) -> html.Div:
    """Render the inspector pane for the current selection.

    Dispatches by state shape:

    * **DAG state** (``hasattr(state, "selected_block_id")``) →
      :func:`_build_dag_inspector` renders the wire card (when
      ``selected_edge_id`` is set), the block param form + aux ports
      section (when ``selected_block_id`` is set), or the empty-state
      placeholder when neither is set (spec §4.5).
    * **Legacy state** (the popover-anchored model) → the original
      consumer / aux-focus dispatch is preserved verbatim below so the
      ``PHENOTYPIC_GUI_DAG`` flag can stay off in production until the
      DAG path is fully shipped.

    Args:
        state: The full builder state (legacy or DAG schema).
        registry: Operation registry consulted for parameter metadata.

    Returns:
        A :class:`dash.html.Div` wrapping the inspector card. Always carries
        the :data:`INSPECTOR_CONTAINER` id so callbacks can swap children.
    """

    # Duck-typed dispatch — same pattern as ``state_to_json`` in _state.py
    # to stay resilient against importlib.reload in tests.  The
    # ``# type: ignore[arg-type]`` is needed because mypy resolves
    # ``BuilderState`` to the legacy class when ``PHENOTYPIC_GUI_DAG=0``
    # (the static default); the duck-typed branch is only reachable
    # when the runtime state IS a ``_DagBuilderState``.
    if hasattr(state, "selected_block_id"):
        return _build_dag_inspector(state, registry)  # type: ignore[arg-type]

    if state.selected_node_id is None:  # type: ignore[attr-defined]
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
        className="mb-3",
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

    Four vertical sections wrapped in a ``dbc.Container(fluid=True)``:

    * Header chrome (logo + pipeline I/O).
    * Footer card with image source + I/O controls (sits above the
      breadcrumb so the active-image selector and Run preview button are
      anchored near the top of the page).
    * Breadcrumb nav (full width).
    * Three-column body — palette, canvas, inspector — sized 3/6/3.

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
            # DAG-redesign stores (spec §6).  Mounted on every render
            # path regardless of ``PHENOTYPIC_GUI_DAG`` so the new
            # callbacks never error on missing inputs.  Until the flag
            # is on, the stores stay at their initial values and the
            # downstream callbacks no-op.
            dcc.Store(
                id=ids.STORE_VIEWPORT_OP,
                data=None,
            ),
            dcc.Store(
                id=ids.STORE_ISSUES,
                data=[],
            ),
            dcc.Store(
                id=ids.STORE_ASSET_STATUS,
                # Default "everything ready" so the banner stays hidden
                # until the clientside asset-poll loop knocks one of
                # the fields to ``False``.
                data={
                    "wire_drawing": True,
                    "palette_dnd": True,
                    "viewport_ops": True,
                    "dagre_missing": False,
                },
            ),
            # Palette drag-and-drop store: written by
            # ``assets/palette_dnd.js`` on drop / keyboard fallback.
            # See spec §5.5 (clientside event contract) and §5.6
            # (``block_create`` dispatch).
            dcc.Store(
                id=ids.STORE_PALETTE_DROP,
                data=None,
            ),
            # Phase 4 wire-drawing store: written by
            # ``assets/wire_drawing.js`` on edge gestures + by the
            # inspector wire / aux cards (Agent 4C) for keyboard /
            # button-driven mutations.  Carries a discriminated-union
            # payload routed by ``payload["kind"]`` to the appropriate
            # ``edge_*`` / ``list_aux_*`` / ``wire_select`` /
            # ``block_select`` dispatch.  Mounted unconditionally so the
            # Phase 4 callbacks never error on a missing input; until
            # the feature flag is on, the store stays at ``None`` and
            # downstream dispatches no-op.
            dcc.Store(
                id=ids.STORE_EDGE_EVENT,
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
            # Confirm-delete modal mounted once at app boot; visibility
            # driven by ``STORE_BUILDER_STATE.pending_delete_block_id``.
            # The block-delete dispatcher wires the open / close
            # behaviour; mounting here keeps
            # ``BTN_CONFIRM_DELETE`` / ``BTN_CANCEL_DELETE`` resolvable.
            build_confirm_delete_modal(),
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
                [header, build_footer(image_root), build_breadcrumb(state), body_row],
                fluid=True,
            ),
        ]
    )


__all__ = [
    "build_palette",
    "build_canvas",
    "build_canvas_elements",
    "build_canvas_elements_dag",
    "build_canvas_section",
    "build_inspector",
    "build_breadcrumb",
    "build_footer",
    "build_app_layout",
    "build_popover_contents",
    "build_asset_status_banner",
    "build_confirm_delete_modal",
    "INSPECTOR_FOCUS_AUX_BANNER_ID",
]

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


#: Background tint for aux-dock nodes. A washed-out lavender so aux nodes
#: read as "different family" from the main ribbon at a glance, while still
#: allowing the existing per-stage label color to show through. Hand-mixed
#: from the ``--oi-purple`` accent (``#CC79A7``) at ~12% opacity over white.
_AUX_NODE_BG: str = "#f6eaf2"


def _canvas_stylesheet() -> List[dict]:
    """Cytoscape stylesheet used by :func:`build_canvas`.

    Phase 3 may extend this list (e.g. to highlight nodes with hot
    intermediates), so we keep it as a function for easy reuse.

    The aux-port additions appended at the end (image-flow / aux-wire
    edges, port handles, aux dock nodes) live in the cytoscape stylesheet
    so they apply to actual cytoscape elements; the matching DOM-side
    classes in ``builder.css`` style the inspector mirrors. The cytoscape
    side intentionally references ``OI_PURPLE`` directly (the value behind
    ``--color-interactive`` / ``--oi-purple``) because cytoscape's
    canvas renderer cannot resolve CSS custom properties.
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
                # Fixed-width ribbon/aux nodes so port-handle placement is
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
        # Image-flow edges between consecutive main-ribbon nodes. Same
        # visual as the default edge (gray solid + arrow); kept as a
        # named class so future polish (different arrow head, animation)
        # can target it without touching aux wires.
        {
            "selector": "edge.image-flow",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": COLOR_MUTED,
                "line-color": COLOR_MUTED,
                "width": 1.5,
            },
        },
        # Aux wires: aux-dock node -> consumer port handle. Purple
        # dashed + no arrow so they read as "configuration flow" rather
        # than "image flow". The endpoint specs anchor the wire at the
        # aux's RIGHT edge (``+90px 0`` matches the fixed 180px width)
        # and the port handle's LEFT edge (``-7px 0`` matches the
        # 14px port width), so the wire visibly *exits* the aux on
        # its right and *enters* the port on its left — Galaxy-style
        # horizontal data-flow with a smooth bezier curve to bridge
        # the vertical offset between the aux dock and the main
        # ribbon. ``unbundled-bezier`` lets us define an explicit
        # control point so the curve bows AWAY from the consumer
        # body (positive distance pulls the midpoint to the right of
        # the source-target line).
        {
            "selector": "edge.aux-wire",
            "style": {
                "curve-style": "unbundled-bezier",
                "source-endpoint": "90px 0",
                "target-endpoint": "-7px 0",
                "control-point-distances": [-30],
                "control-point-weights": [0.5],
                "line-color": OI_PURPLE,
                "line-style": "dashed",
                "target-arrow-shape": "triangle",
                "target-arrow-color": OI_PURPLE,
                "arrow-scale": 0.8,
                "width": 1.5,
            },
        },
        # Port-handle sub-nodes positioned on the consumer's left edge.
        # Default appearance: small unfilled square/ellipse (per ``shape``
        # data field). Labels live in the inspector + hover tooltip
        # rather than on the canvas — Galaxy-style — so they never
        # overlap the consumer node's class-name label even for long
        # parameter names like ``inoculum_detector``.
        {
            "selector": "node.port-handle",
            "style": {
                "shape": "data(shape)",
                "label": "",  # canvas-clean; param name shown in inspector
                "background-color": COLOR_BORDER,
                "border-color": COLOR_MUTED,
                "border-width": 1.5,
                "width": 14,
                "height": 14,
                "padding": 0,
                "min-width": 14,
            },
        },
        # Wired modifier: solid purple fill and matching label color.
        {
            "selector": "node.port-handle.wired",
            "style": {
                "background-color": OI_PURPLE,
                "border-color": OI_PURPLE,
                "color": COLOR_NAVY,
            },
        },
        # Aux dock node: same shape as ribbon nodes but with a soft
        # lavender background so users can tell at a glance that it's
        # an aux source rather than part of the image flow.
        {
            "selector": "node.aux-node",
            "style": {
                "background-color": _AUX_NODE_BG,
                "border-color": OI_PURPLE,
                "border-style": "solid",
            },
        },
        # Orphan aux node: no consumer wires reference it. Dashed border
        # signals "will be dropped on save" so users know to wire it up
        # or delete it.
        {
            "selector": "node.aux-node.aux-orphan",
            "style": {
                "border-style": "dashed",
                "opacity": 0.7,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Preset-layout positioning
# ---------------------------------------------------------------------------
#
# Cytoscape's ``preset`` layout requires explicit ``(x, y)`` for every node;
# the constants below pin the geometry of the main ribbon, port handles,
# and aux dock so callbacks can stay layout-agnostic. Numbers are tuned
# against the ~700px canvas slot configured in ``build_app_layout``.

#: Horizontal step between consecutive main-ribbon nodes (px).
_RIBBON_X_STEP: int = 180

#: Left padding before the first ribbon node (px).
_RIBBON_X_OFFSET: int = 24

#: Y position of every main-ribbon node (px).
_RIBBON_Y: int = 80

#: Y position of every aux-dock node (px). Picked so wires from the dock
#: drop visibly below the main ribbon without competing with selection
#: handles.
_AUX_DOCK_Y: int = 240

#: Horizontal step between consecutive orphan aux nodes when they have no
#: consumer to anchor under (px).
_ORPHAN_X_STEP: int = 160

#: Horizontal offset of the port handle's CENTER from the consumer's
#: center. The consumer is 180px wide (half-width 90), the port handle
#: 14px wide (half-width 7). Setting this to ``-97`` puts the port
#: handle's center 97px left of consumer center → the port handle's
#: right edge sits exactly on the consumer's left edge, so the dot
#: reads as VISUALLY ATTACHED to the node (a small tab on the left
#: edge) rather than floating in space.
_PORT_HANDLE_DX: int = -97

#: Horizontal offset of an aux-dock node's center from its anchor
#: consumer's center. Aux nodes sit BELOW-AND-LEFT of their consumer
#: so the wire from the aux's RIGHT edge naturally curves UP-AND-RIGHT
#: into the port handle's LEFT edge — no overlap with the consumer
#: body, no taxi-vertical "from the top" path. Pairs with the bezier
#: curve + explicit source-endpoint/target-endpoint defined in the
#: ``edge.aux-wire`` stylesheet.
_AUX_DX: int = -240

#: Vertical offset between stacked aux nodes (when one consumer feeds
#: from multiple aux operations, e.g. a list-typed port).
_AUX_STACK_DY: int = 70

#: Vertical spacing between stacked port handles on the same consumer (px).
_PORT_HANDLE_DY: int = 14

#: Encoding scheme for cytoscape port-handle ids.
#:
#: Cytoscape elements need flat string ids (cytoscape rejects dict ids
#: outside the dash pattern-matching layer). We mangle the structured
#: components ``(node_id, param, slot)`` into a delimited string so
#: callbacks can recover the structured form via
#: :func:`_decode_port_handle_id` without a lookup table. Choice of
#: ``__`` as separator avoids collision with single underscores in
#: realistic class/param names.
_PORT_HANDLE_PREFIX: str = "port-handle"
_PORT_HANDLE_SEP: str = "__"


def _encode_port_handle_id(node_id: str, param: str, slot: int) -> str:
    """Mangle a (node_id, param, slot) triple into a flat cytoscape id.

    The cytoscape canvas needs a string id; the matching dict id is
    available via :func:`phenotypic.gui.builder._ids.port_handle_id` for
    Dash pattern-matched callbacks.

    Args:
        node_id: Consumer node identifier the port handle attaches to.
        param: Aux-port-eligible parameter name on the consumer.
        slot: Zero-based slot index. Always ``0`` for scalar ports.

    Returns:
        Flat string ``"port-handle__<node_id>__<param>__<slot>"`` suitable
        as a cytoscape element id.
    """

    return _PORT_HANDLE_SEP.join(
        [_PORT_HANDLE_PREFIX, node_id, param, str(slot)]
    )


def _decode_port_handle_id(
    encoded: str,
) -> Optional[tuple[str, str, int]]:
    """Reverse of :func:`_encode_port_handle_id`.

    Phase 4 callbacks that read the cytoscape ``tapNodeData`` payload
    can use this to recover the structured triple. Returns ``None`` for
    any string that doesn't match the encoding (e.g. a tap on an aux
    node or main-ribbon node).

    Args:
        encoded: Cytoscape element id string.

    Returns:
        ``(node_id, param, slot)`` tuple when *encoded* is a port-handle
        id, otherwise ``None``.
    """

    if not encoded.startswith(_PORT_HANDLE_PREFIX + _PORT_HANDLE_SEP):
        return None
    parts = encoded.split(_PORT_HANDLE_SEP)
    # Expected shape: [_PORT_HANDLE_PREFIX, node_id, param, slot]
    if len(parts) != 4:
        return None
    _, node_id, param, slot_str = parts
    try:
        slot = int(slot_str)
    except ValueError:
        return None
    return node_id, param, slot


def _aux_port_specs(
    node: StepNode,
    registry: "OperationRegistry",
) -> List[tuple[str, int, str]]:
    """Enumerate ``(param_name, slot_index, shape)`` for a consumer's ports.

    Walks ``node.aux_ports`` and looks up each parameter's metadata in the
    registry so callers can know whether to render the handle as a square
    (operation-typed) or ellipse (pipeline-typed). Parameters not in the
    registry are dropped silently — the node may have an out-of-date
    ``aux_ports`` map for a class that was renamed/removed, and the
    canvas should degrade gracefully rather than render orphan handles.

    The order of the returned list matches the parameter order in the
    registry's ``parameters`` mapping (which preserves declaration order
    via :class:`inspect.Signature`); within a parameter, slots are
    enumerated in their stored order.

    Args:
        node: A consumer :class:`StepNode` whose ``aux_ports`` map should
            be enumerated.
        registry: The operation registry for type-flag lookup.

    Returns:
        Ordered list of ``(param_name, slot_index, shape)`` triples where
        ``shape`` is ``"ellipse"`` for pipeline-typed ports and
        ``"square"`` otherwise.
    """

    info = registry.get(node.class_name)
    if info is None:
        return []

    specs: List[tuple[str, int, str]] = []
    for param_name in info.parameters:
        slots = node.aux_ports.get(param_name)
        if slots is None:
            continue
        param_info = info.parameters[param_name]
        shape = "ellipse" if param_info.is_pipeline else "square"
        for slot_idx, _ in enumerate(slots):
            specs.append((param_name, slot_idx, shape))
    return specs


def _aux_consumer_anchors(
    nodes: List[StepNode],
    aux_nodes: List[StepNode],
    ribbon_x_by_id: Dict[str, int],
) -> Dict[str, int]:
    """Pick the anchor x for each aux node based on its (leftmost) consumer.

    Aux dock nodes sit below the main ribbon at ``y = _AUX_DOCK_Y``; their
    x is the x of the leftmost main-ribbon consumer that wires into them.
    Orphans (no consumer references) get a placeholder x of ``-1`` so the
    caller can layout them separately at the right edge.

    Args:
        nodes: Main-ribbon nodes (used to discover wires).
        aux_nodes: Aux-dock nodes that need an anchor.
        ribbon_x_by_id: Map from main-ribbon ``node_id`` to its x position.

    Returns:
        Map from aux ``node_id`` to anchor x. ``-1`` for orphans (no
        consumer found).
    """

    anchors: Dict[str, int] = {n.node_id: -1 for n in aux_nodes}
    for consumer in nodes:
        consumer_x = ribbon_x_by_id.get(consumer.node_id)
        if consumer_x is None:
            continue
        for slots in consumer.aux_ports.values():
            for aux_id in slots:
                if aux_id is None:
                    continue
                if aux_id not in anchors:
                    continue
                # Take the LEFTMOST consumer's x (smaller wins) so the aux
                # appears under the consumer with smallest x. ``-1`` is
                # the "no consumer yet" sentinel; any real x replaces it.
                current = anchors[aux_id]
                if current == -1 or consumer_x < current:
                    anchors[aux_id] = consumer_x
    return anchors


def build_canvas(
    scope: BuilderScope,
    selected_node_id: Optional[str],
) -> cyto.Cytoscape:
    """Render the linear chain for *scope* as a cytoscape canvas.

    Each :class:`StepNode` becomes one cytoscape node; consecutive
    main-ribbon nodes are joined by ``image-flow`` edges. Nested
    ``ImagePipeline`` nodes get a folder glyph in their label so the
    user can tell drillable nodes apart.

    Galaxy-style aux dock additions:

    * Each aux-port-eligible parameter renders one small port-handle
      sub-node per slot, positioned just to the left of the consumer
      so incoming wires read as "feeding into" the consumer.
    * Aux nodes that no consumer wires to (orphans) are placed on the
      right edge with a dashed border (``aux-orphan`` class) so users
      can tell they'll be dropped on save.
    * Aux wires are emitted as edges with ``aux-wire`` class — purple
      dashed in the cytoscape stylesheet — and target the consumer's
      port-handle id rather than the consumer node directly.

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

    # ── 2. Image-flow edges (drawn first so they sit underneath nodes). ─
    prev_id: Optional[str] = None
    for node in scope.nodes:
        if prev_id is not None:
            elements.append(
                {
                    "data": {
                        "id": f"{prev_id}__{node.node_id}",
                        "source": prev_id,
                        "target": node.node_id,
                    },
                    "classes": "image-flow",
                }
            )
        prev_id = node.node_id

    # ── 3. Aux wires (target port handles, not consumers). ──────────────
    for consumer in scope.nodes:
        for param_name, slots in consumer.aux_ports.items():
            for slot_idx, aux_id in enumerate(slots):
                if aux_id is None:
                    continue
                handle_id = _encode_port_handle_id(
                    consumer.node_id, param_name, slot_idx
                )
                elements.append(
                    {
                        "data": {
                            "id": (
                                f"wire__{aux_id}__"
                                f"{consumer.node_id}__{param_name}__{slot_idx}"
                            ),
                            "source": aux_id,
                            "target": handle_id,
                        },
                        "classes": "aux-wire",
                    }
                )

    # ── 4. Main-ribbon nodes. ───────────────────────────────────────────
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

    # ── 5. Aux-dock nodes (orphans go to the right). ────────────────────
    referenced_aux_ids: set[str] = set()
    for consumer in scope.nodes:
        for slots in consumer.aux_ports.values():
            for aux_id in slots:
                if aux_id is not None:
                    referenced_aux_ids.add(aux_id)

    aux_anchors = _aux_consumer_anchors(
        scope.nodes, scope.aux_nodes, ribbon_x_by_id
    )

    # Walk consumers in left-to-right order; for each, place its aux
    # nodes in a vertical column to the LEFT of the consumer. Multiple
    # auxes for the same consumer stack down (slot 0 highest, slot N
    # lowest), so the wire from each aux's right edge curves up-and-
    # right cleanly into the matching port handle on the consumer.
    aux_index_within_consumer: Dict[str, int] = {}
    aux_positions: Dict[str, tuple[int, int]] = {}
    for consumer in scope.nodes:
        # Collect aux ids referenced by this consumer in slot order so
        # the topmost aux maps to slot 0, the next to slot 1, etc.
        consumer_aux_ids: List[str] = []
        for slots in consumer.aux_ports.values():
            for aux_id in slots:
                if aux_id is not None and aux_id not in consumer_aux_ids:
                    consumer_aux_ids.append(aux_id)
        cons_x = ribbon_x_by_id[consumer.node_id]
        for stack_idx, aux_id in enumerate(consumer_aux_ids):
            if aux_id in aux_positions:
                continue  # already placed under a leftmost consumer
            aux_positions[aux_id] = (
                cons_x + _AUX_DX,
                _RIBBON_Y + (stack_idx + 1) * _AUX_STACK_DY,
            )
            aux_index_within_consumer[aux_id] = stack_idx

    # Place orphans to the right of the rightmost ribbon node so they
    # don't compete with anchored aux nodes for x space.
    if ribbon_x_by_id:
        orphan_x_start = (
            max(ribbon_x_by_id.values()) + _RIBBON_X_STEP
        )
    else:
        orphan_x_start = _RIBBON_X_OFFSET
    orphan_index = 0

    for aux in scope.aux_nodes:
        stage = _safe_stage(aux.class_name)
        label = aux.label or aux.class_name
        if aux.class_name == PIPELINE_CLASS_NAME:
            label = f"\U0001F4C1 {label}"

        is_orphan = aux.node_id not in referenced_aux_ids
        if is_orphan or aux.node_id not in aux_positions:
            x = orphan_x_start + orphan_index * _ORPHAN_X_STEP
            y = _AUX_DOCK_Y
            orphan_index += 1
        else:
            x, y = aux_positions[aux.node_id]

        classes = "aux-node"
        if is_orphan:
            classes = f"{classes} aux-orphan"
        if aux.node_id == selected_node_id:
            classes = f"{classes} selected"

        elements.append(
            {
                "data": {
                    "id": aux.node_id,
                    "label": label,
                    "bg": _STAGE_COLORS.get(stage, _STAGE_COLORS["ops"]),
                    "stage": stage,
                    "class_name": aux.class_name,
                },
                "classes": classes,
                "selectable": True,
                "grabbable": True,
                "position": {"x": x, "y": y},
            }
        )

    # Suppress the unused-anchor warning while keeping the helper for
    # future v2 work (manual repositioning will need it back).
    _ = aux_anchors

    # ── 6. Port handles (rendered last so they sit on top). ─────────────
    for consumer in scope.nodes:
        specs = _aux_port_specs(consumer, registry)
        if not specs:
            continue
        # Center the stack of handles vertically around the consumer.
        # ``offset`` ranges from -(N-1)/2 to +(N-1)/2 so a 1-handle stack
        # lands exactly on the consumer's y, a 2-stack straddles it, etc.
        n_handles = len(specs)
        consumer_x = ribbon_x_by_id[consumer.node_id]
        for i, (param_name, slot_idx, shape) in enumerate(specs):
            offset = (i - (n_handles - 1) / 2) * _PORT_HANDLE_DY
            handle_x = consumer_x + _PORT_HANDLE_DX
            handle_y = int(_RIBBON_Y + offset)
            handle_id = _encode_port_handle_id(
                consumer.node_id, param_name, slot_idx
            )
            slots_for_param = consumer.aux_ports.get(param_name) or []
            slot_value = (
                slots_for_param[slot_idx]
                if slot_idx < len(slots_for_param)
                else None
            )
            wired = slot_value is not None
            handle_classes = "port-handle"
            if wired:
                handle_classes = f"{handle_classes} wired"
            elements.append(
                {
                    "data": {
                        "id": handle_id,
                        "label": param_name,
                        "shape": shape,
                        "consumer_id": consumer.node_id,
                        "param": param_name,
                        "slot": slot_idx,
                    },
                    "classes": handle_classes,
                    "selectable": True,
                    "grabbable": False,
                    "position": {"x": handle_x, "y": handle_y},
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
        # balloon the consumer to fill the viewport (which would hide
        # the lane chrome and obscure the swim-lane semantic). 1.0
        # keeps the absolute Python-computed positions intact; the
        # user can still zoom in further via the toolbar.
        maxZoom=1.0,
        minZoom=0.25,
    )


def build_lane_chrome() -> html.Div:
    """Static HTML overlay naming the canvas's two swim lanes.

    Lives outside the cytoscape element graph so it doesn't perturb
    cytoscape's ``fit`` bounding-box calc and doesn't get clipped /
    obscured when nodes auto-zoom. The wrapper is positioned absolute
    inside the ``canvas-cytoscape-wrapper`` flex slot via the
    ``.pheno-canvas-lane-chrome`` rule in ``builder/assets/builder.css``.

    Used by both :func:`build_canvas_section` (initial render) and the
    fan-in callback in ``_callbacks.py`` (every state mutation re-emits
    the wrapper's children, so the chrome must be re-included on each
    update).
    """

    return html.Div(
        [
            html.Span(
                "MAIN IMAGE FLOW",
                className="pheno-canvas-lane-label pheno-canvas-lane-label--main",
            ),
            html.Span(
                "AUX OPERATIONS · op-as-config",
                className="pheno-canvas-lane-label pheno-canvas-lane-label--aux",
            ),
        ],
        className="pheno-canvas-lane-chrome",
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

    lane_chrome = build_lane_chrome()
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
        [build_canvas(scope, selected_node_id), lane_chrome],
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

    Used by :func:`_build_aux_palette_section` to filter the inspector's
    drop-on-port palette down to ops/pipelines that the consumer's port
    will actually accept (the `wire_create` dispatch validates the same
    contract; pre-filtering here saves the user a useless click).

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


def _build_aux_palette_section(
    node: StepNode, registry: "OperationRegistry"
) -> Optional[Any]:
    """Render the inspector's drop-on-port aux palette.

    For each empty slot in *node*'s ``aux_ports``, surfaces a compact
    DropdownMenu of compatible classes — clicking an item dispatches
    ``aux_palette_add`` (creates the aux node and wires it to this slot
    in one shot). Returns ``None`` when the consumer has no aux-port-
    eligible parameters with empty slots.

    Args:
        node: Currently-selected consumer node whose aux ports are being
            offered.
        registry: Operation registry consulted for parameter metadata
            and the compatibility filter.

    Returns:
        A ``dbc.Card`` for insertion into the inspector body, or
        ``None`` when there are no empty slots to populate.
    """

    info = registry.get(node.class_name)
    if info is None:
        return None

    groups: List[Any] = []
    for param_name, p in info.parameters.items():
        if not (p.is_operation or p.is_pipeline):
            continue
        compatible = _compatible_classes_for_port(p, registry)
        if not compatible:
            continue
        # Walk the existing slot list (if any) to find empty slots.
        # Params that have never been touched have no ``aux_ports`` entry —
        # treat them as a single empty slot so the palette can offer the
        # initial wire.
        existing_slots = node.aux_ports.get(param_name)
        slot_iter: List[tuple[int, Optional[str]]]
        if existing_slots is None:
            slot_iter = [(0, None)]
        else:
            slot_iter = list(enumerate(existing_slots))
        for slot_idx, aux_id in slot_iter:
            if aux_id is not None:
                continue  # already wired
            heading = (
                f"+ Add aux for {param_name}[{slot_idx}]"
                if p.is_list
                else f"+ Add aux for {param_name}"
            )
            buttons = [
                dbc.Button(
                    cls_name,
                    id=ids.aux_palette_add_id(
                        cls_name, node.node_id, param_name, slot_idx
                    ),
                    color="primary",
                    size="sm",
                    outline=True,
                    n_clicks=0,
                    className="me-1 mb-1",
                )
                for cls_name in compatible
            ]
            groups.append(
                html.Div(
                    [
                        html.Div(
                            heading,
                            className="text-muted small mb-1",
                        ),
                        html.Div(buttons, className="d-flex flex-wrap"),
                    ],
                    className="mb-3 inspector-aux-palette-group",
                )
            )

    if not groups:
        return None

    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Aux palette", className="mb-2"),
                html.Div(groups, className="d-flex flex-wrap"),
            ]
        ),
        className="mt-3 inspector-aux-palette",
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
            # Hidden placeholders so the Documentation toggle callback's
            # Input/State ids resolve even on the pipeline-sentinel branch.
            *_doc_section_widgets(None),
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
        # Build the wired_slots map for ``param_form``. Each entry maps a
        # parameter name to a list of source class names (one per slot;
        # ``None`` for empty slots). Aux node IDs in ``node.aux_ports`` are
        # resolved against ``scope.aux_nodes`` to get the human-readable
        # class name shown in the inspector's wired rows. An empty dict
        # (``{}``) is preserved verbatim — the caller checks
        # ``wired_slots is None`` vs ``wired_slots == {}`` to decide
        # between the legacy drill-in shape and the wired-row shape.
        aux_class_by_id = {a.node_id: a.class_name for a in scope.aux_nodes}
        wired_slots: dict[str, list[str | None]] = {}
        for param_name, slots in node.aux_ports.items():
            wired_slots[param_name] = [
                aux_class_by_id.get(aux_id) if aux_id is not None else None
                for aux_id in slots
            ]
        form = html.Div(
            param_form(
                op_info,
                current_values=node.params,
                form_id_prefix=node.node_id,
                wired_slots=wired_slots,
            ),
            id=ids.INSPECTOR_PARAM_FORM,
        )

    aux_palette = (
        _build_aux_palette_section(node, registry) if op_info is not None else None
    )

    body_children = [
        *header_children,
        # Documentation section is collapsed by default; emits hidden
        # placeholders carrying the same ids when ``op_info.docstring`` is
        # empty so the toggle callback's Input/State always resolve.
        *_doc_section_widgets(op_info.docstring if op_info else None),
        form,
        *(([aux_palette] if aux_palette is not None else [])),
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
            # Click-then-click wire creation: holds the partially-specified
            # wire endpoint (either a port handle or an aux node) between the
            # first and second click. ``None`` when no wire is pending.
            # Shape: ``{"endpoint_kind": "port", "node_id": str, "param": str,
            # "slot": int}`` for port-handle endpoints, or
            # ``{"endpoint_kind": "aux", "aux_id": str}`` for aux-node
            # endpoints.
            dcc.Store(
                id=ids.STORE_PENDING_WIRE,
                data=None,
            ),
            # Aux palette filter target. ``None`` when the palette is in its
            # default unfiltered mode; ``{"node_id": ..., "param": ...,
            # "slot": ...}`` when it is filtering for a specific port.
            dcc.Store(
                id=ids.STORE_AUX_PALETTE_TARGET,
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
    "build_inspector",
    "build_breadcrumb",
    "build_footer",
    "build_app_layout",
    "_decode_port_handle_id",
]

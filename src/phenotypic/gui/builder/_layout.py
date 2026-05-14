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
    COLOR_GOLD,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    COLOR_WHITE,
    FONT_FAMILY_MONO,
    FONT_SIZE_LABEL,
    OI_GREEN,
    OI_PURPLE,
    OI_VERMILION,
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
    _DagBuilderScope,
    _DagBuilderState,
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
            # Param drill (legacy popover-era): the synthesized op-param
            # scope was retired in Phase 7. We surface a label
            # ``GaussianBlur.sub_op`` so older saved state still renders a
            # readable breadcrumb, but the walker stops here — the
            # synthesized scope is no longer materialised.
            base = node.label or node.class_name
            labels.append(f"{base}.{param}")
            break
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
            # drag-and-drop bridge in ``assets/palette_dnd.js``.  dbc
            # components reject unknown kwargs, so the HTML attributes
            # live on an ``html.Div`` wrapper; ``palette_dnd.js`` uses
            # ``closest("[data-palette-class]")`` so the ancestor lookup
            # finds the wrapper from any descendant element.
            buttons.append(
                html.Div(
                    dbc.Button(
                        button_children,
                        id=ids.palette_button_id(op_info.name),
                        color=_STAGE_BUTTON_OUTLINE_COLOR.get(stage, "primary"),
                        outline=True,
                        size="sm",
                        n_clicks=0,
                        className=button_class,
                    ),
                    draggable="true",
                    **{"data-palette-class": op_info.name},
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


def build_new_pipeline_palette_button() -> dbc.Button:
    """Build the ``+ New Pipeline`` palette button (DAG redesign, spec §4.4).

    The button is rendered above the per-category palette accordions and
    serves as the palette entry for the ``ImagePipeline`` container
    sentinel class.  It carries the same ``draggable="true"`` +
    ``data-palette-class="ImagePipeline"`` attributes as the regular
    palette buttons so the clientside ``palette_dnd.js`` glue (Phase 3)
    handles the drag → ``STORE_PALETTE_DROP`` write → ``block_create``
    dispatch end-to-end with no extra wiring.

    The button intentionally REUSES :data:`BTN_NEW_PIPELINE_NODE` as its
    Dash id so the existing keyboard-fallback callback (``add_pipeline``
    in the legacy schema, ``block_create`` with
    ``class_name="ImagePipeline"`` in the DAG schema) continues to
    resolve for users without drag-and-drop.

    Returns:
        A :class:`dbc.Button` ready to embed above the operations
        accordion in the palette column.
    """

    # ``draggable="true"`` + ``data-palette-class="ImagePipeline"`` are
    # read by ``assets/palette_dnd.js`` via ``closest("[data-palette-
    # class]")`` so the wrapping ``html.Div`` is the ancestor the event
    # delegate finds.  dbc components reject unknown kwargs, hence the
    # wrapper (matches the per-op palette buttons above).
    return html.Div(
        dbc.Button(
            [html.Span("⛓"), html.Span(" + New Pipeline", className="ms-1")],
            id=ids.BTN_NEW_PIPELINE_NODE,
            color="primary",
            outline=True,
            size="sm",
            n_clicks=0,
            className="w-100 mb-2 palette-button palette-button--pipeline",
            title=(
                "Drag onto the canvas to add a nested ImagePipeline "
                "container (or click to drop in the current scope)."
            ),
        ),
        draggable="true",
        **{"data-palette-class": PIPELINE_CLASS_NAME},
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
    """Cytoscape stylesheet for the DAG builder canvas (:func:`build_canvas`).

    Renders the spec §4.1–§4.6 visual language:

    * **Blocks** (``node.dag-block``) — stage-tinted 180×54 rounded
      rectangles carrying the op label.  ``min-width`` / ``min-height``
      pin the body size: every block is a cytoscape *compound parent*
      of its port sub-nodes, and cytoscape would otherwise shrink the
      body down to the bounding box of those tiny children.
    * **Ports** (``node.dag-port``) — small edge markers: a filled blue
      image-in circle, a hollow image-out circle, a purple aux square.
      ``viewport_ops.js`` snaps them onto the block's edges after each
      dagre pass (dagre itself skips ``is_port`` sub-nodes).
    * **Issue badges** (``node.dag-issue``) — a red ``!`` / amber ``?``
      chip pinned to the block's top-right corner (spec §4.6).
    * **Wires** (``edge.dag-wire``) — blue-solid image flow, purple-
      dashed aux; 3 px on the main spine, 2 px elsewhere (spec §4.3).
    * **Containers** — purple-bordered compound groups; see the
      ``dag-block--container`` family below.

    cytoscape's canvas renderer cannot resolve CSS custom properties, so
    the rules reference ``_design`` colour constants directly.
    """

    return [
        # ── Generic node base ──────────────────────────────────────────
        # Deliberately minimal: blocks, ports and issue badges are all
        # cytoscape ``node``s but want very different chrome, so the
        # base only carries font defaults shared by every node.  Note
        # there is *no* ``label: data(label)`` mapping here — ports /
        # badges have no ``label`` data field and a base mapper would
        # spam "no mapping for label" warnings on every render.
        {
            "selector": "node",
            "style": {
                "shape": "round-rectangle",
                "font-family": FONT_FAMILY_MONO,
                # Cytoscape canvas-renders labels and only accepts pixel
                # values for font-size; rem units silently fall back.
                "font-size": "12px",
                "color": COLOR_NAVY,
                "border-color": COLOR_BORDER,
                "border-width": 1,
                "background-color": COLOR_SURFACE,
            },
        },
        # ── Block body (spec §4.1) ─────────────────────────────────────
        # 180×54 stage-tinted card.  ``min-width`` / ``min-height`` are
        # the load-bearing properties: each block is a compound parent
        # of its port sub-nodes, so without a floor cytoscape collapses
        # the body to the ports' (tiny) bounding box.  ``padding: 0``
        # keeps the compound box flush with the ports that
        # ``viewport_ops.js`` snaps onto its edges.
        {
            "selector": "node.dag-block",
            "style": {
                "shape": "round-rectangle",
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "ellipsis",
                "text-max-width": 150,
                "background-color": "data(bg)",
                "border-color": COLOR_NAVY,
                "border-width": 1,
                "font-weight": "500",
                "width": 180,
                "height": 54,
                "min-width": 180,
                "min-height": 54,
                "padding": 0,
            },
        },
        # Stage-coloured borders (spec §4.2: "1px stage-coloured border
        # for main-flow ops").  The stage tint already lands via
        # ``data(bg)``; the border echoes it a shade darker.
        {
            "selector": "node.dag-block.stage--meas",
            "style": {"border-color": COLOR_GOLD},
        },
        {
            "selector": "node.dag-block.stage--post",
            "style": {"border-color": OI_GREEN},
        },
        # Input Image sentinel — green "source" tag (spec §4.1 chevron).
        {
            "selector": "node.dag-block--input-image",
            "style": {
                "shape": "round-tag",
                "background-color": "#d6efe4",
                "border-color": OI_GREEN,
                "border-width": 1.5,
            },
        },
        # Aux-consumed block — solid purple 1.5px border (spec §4.2):
        # this block's output feeds an aux port, so it lives off the
        # main spine.
        {
            "selector": "node.dag-block--aux-consumed",
            "style": {
                "border-color": OI_PURPLE,
                "border-width": 1.5,
                "border-style": "solid",
            },
        },
        # Advisory issue — yellow 1.5px border (spec §4.2 / Rule 7).
        {
            "selector": "node.dag-block--advisory",
            "style": {
                "border-color": COLOR_GOLD,
                "border-width": 1.5,
            },
        },
        # Blocking issue — solid red 2.5px border (spec §4.2 Rules 1-6).
        {
            "selector": "node.dag-block--error",
            "style": {
                "border-color": OI_VERMILION,
                "border-width": 2.5,
                "border-style": "solid",
            },
        },
        # Stub (unreachable from Input Image) — dashed red 2.5px border
        # (spec §4.2: reads as "draft" rather than "broken").
        {
            "selector": "node.dag-block--stub",
            "style": {
                "border-color": OI_VERMILION,
                "border-width": 2.5,
                "border-style": "dashed",
            },
        },
        {
            "selector": "node.selected",
            "style": {
                "border-color": COLOR_BLUE,
                "border-width": 3,
            },
        },
        # ── Ports (spec §4.2) ──────────────────────────────────────────
        # Small edge markers.  ``min-width`` / ``min-height`` are reset
        # here too: a port is a leaf node, but it shares the cascade
        # with ``node.dag-block`` only via the generic ``node`` base —
        # explicit small dimensions keep it a discrete dot.
        {
            "selector": "node.dag-port",
            "style": {
                "label": "",
                "width": 13,
                "height": 13,
                "min-width": 13,
                "min-height": 13,
                "padding": 0,
                "border-width": 1.5,
                "z-index": 10,
            },
        },
        # Image-input port — filled blue circle, left edge.
        {
            "selector": "node.dag-port--input",
            "style": {
                "shape": "ellipse",
                "background-color": COLOR_BLUE,
                "border-color": COLOR_BLUE,
            },
        },
        # Image-output port — hollow neutral circle, right edge (the
        # port itself is neutral; the *wire* colour follows the target).
        {
            "selector": "node.dag-port--output",
            "style": {
                "shape": "ellipse",
                "background-color": COLOR_SURFACE,
                "border-color": COLOR_MUTED,
            },
        },
        # Aux-input port — purple rounded square, bottom edge.  Hollow
        # when empty so a dense canvas still reads "needs an aux here".
        {
            "selector": "node.dag-port--aux",
            "style": {
                "shape": "round-rectangle",
                "background-color": COLOR_SURFACE,
                "border-color": OI_PURPLE,
            },
        },
        # Required + empty aux port — red ring (spec §4.2 / Rule 3).
        {
            "selector": "node.dag-port--aux.dag-port--required",
            "style": {"border-color": OI_VERMILION},
        },
        # Wired aux port — solid purple fill (any slot non-empty).
        {
            "selector": "node.dag-port--aux.dag-port--wired",
            "style": {
                "background-color": OI_PURPLE,
                "border-color": OI_PURPLE,
            },
        },
        # ── Issue badges (spec §4.6) ───────────────────────────────────
        # A small chip pinned to the block's top-right corner by
        # ``viewport_ops.js``.  Blocking issues read ``!`` on red;
        # advisory hints read ``?`` on amber.
        {
            "selector": "node.dag-issue",
            "style": {
                "shape": "ellipse",
                "width": 18,
                "height": 18,
                "min-width": 18,
                "min-height": 18,
                "padding": 0,
                "border-width": 1,
                "font-weight": "bold",
                "font-size": "13px",
                "text-valign": "center",
                "text-halign": "center",
                "z-index": 20,
            },
        },
        {
            "selector": "node.dag-issue--error",
            "style": {
                "label": "!",
                "background-color": OI_VERMILION,
                "border-color": OI_VERMILION,
                "color": COLOR_WHITE,
            },
        },
        {
            "selector": "node.dag-issue--advisory",
            "style": {
                "label": "?",
                "background-color": COLOR_GOLD,
                "border-color": COLOR_GOLD,
                "color": COLOR_NAVY,
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
        # DAG-redesign wires (spec §4.3).  Image-flow wires render blue
        # solid throughout the chain including past the measure boundary;
        # aux wires render purple dashed.  Main-path edges (InputImage →
        # terminal) get 3px width; aux + non-main edges stay at 2px.
        # Selected wire stays above blocks (default below) for visibility.
        {
            "selector": "edge.dag-wire",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "none",
                "line-color": COLOR_MUTED,
                "width": 2,
                "z-compound-depth": "bottom",
            },
        },
        {
            "selector": "edge.dag-wire--image",
            "style": {
                "line-color": COLOR_BLUE,
                "line-style": "solid",
            },
        },
        {
            "selector": "edge.dag-wire--aux",
            "style": {
                "line-color": OI_PURPLE,
                "line-style": "dashed",
            },
        },
        {
            "selector": "edge.dag-wire--main",
            "style": {
                "width": 3,
            },
        },
        {
            "selector": "edge.dag-wire--selected",
            "style": {
                "width": 4,
                "line-color": COLOR_NAVY,
                "z-compound-depth": "top",
            },
        },
        # ── Container compound chrome (spec §4.4 + §4.7) ────────────────
        # Compound parents drawn with a purple 1.5px border + a low-
        # opacity surface tint.  The title bar text lives in the
        # compound node's ``label`` data; ``text-valign: top`` plus the
        # negative ``text-margin-y`` lifts the label clear of the
        # container's body so child blocks render beneath it.  Padding
        # is leaf-first (spec §4.7) so the inner dagre pass sees a
        # bounding box wide enough for its children plus a comfortable
        # gutter.
        {
            "selector": "node.dag-block--container",
            "style": {
                "shape": "round-rectangle",
                "background-color": COLOR_SURFACE,
                "background-opacity": 0.2,
                "border-color": OI_PURPLE,
                "border-width": 1.5,
                "padding": 32,
                "label": "data(label)",
                "text-valign": "top",
                "text-halign": "center",
                "text-margin-y": -8,
                "color": COLOR_NAVY,
                "font-weight": "bold",
                "font-family": FONT_FAMILY_MONO,
                "font-size": "12px",
            },
        },
        # ── Collapsed container (spec §4.4) ────────────────────────────
        # When ``block.collapsed == True`` the container renders as a
        # 1-row block (same dimensions as a regular op block) so wires
        # are never visually orphaned; children are hidden by the
        # ``dag-block--container-hidden-child`` selector below.
        {
            "selector": "node.dag-block--container.dag-block--collapsed",
            "style": {
                "width": 180,
                "height": 54,
                "padding": 0,
                "text-valign": "center",
                "text-margin-y": 0,
            },
        },
        # Children of a collapsed container are hidden from the canvas.
        # The data-attribute pivot (``parent_collapsed``) lets cytoscape's
        # canvas renderer drop them without us re-emitting elements per
        # collapse toggle.
        {
            "selector": "node[?parent_collapsed]",
            "style": {
                "display": "none",
            },
        },
        # ── Consumer-fed dot (spec §4.1, §4.4) ─────────────────────────
        # The container scope's ``InputImage`` sentinel renders as a
        # small purple dot anchored to the container's inner-left edge,
        # NOT as a regular block.  The DOM-side cursor / outline lives
        # in ``builder.css``; the canvas rule below is what cytoscape's
        # renderer actually paints.
        {
            "selector": "node.dag-block__consumer-fed-dot",
            "style": {
                "shape": "ellipse",
                "label": "",
                "width": 12,
                "height": 12,
                "background-color": OI_PURPLE,
                "border-color": OI_PURPLE,
                "border-width": 1.5,
                "padding": 0,
            },
        },
        # ── Empty container placeholder (spec §4.8) ────────────────────
        # When a container's nested scope holds only the auto-seeded
        # ``InputImage`` (no real ops) the renderer emits a label-only
        # sub-node carrying the hint text.  The element is non-selectable
        # so a stray click on the placeholder doesn't fight the
        # container's title-bar click handler.
        {
            "selector": "node.dag-block__placeholder",
            "style": {
                "shape": "round-rectangle",
                "background-color": COLOR_SURFACE,
                "background-opacity": 0,
                "border-width": 0,
                "label": "data(label)",
                "color": COLOR_MUTED,
                "font-family": FONT_FAMILY_MONO,
                "font-size": "11px",
                "font-style": "italic",
                "text-valign": "center",
                "text-halign": "center",
                "width": 140,
                "height": 24,
                "padding": 0,
            },
        },
        # ── Empty container chrome (spec §4.8) ─────────────────────────
        # A container with only the auto-seeded ``InputImage`` inside
        # reads as a dashed-outline drop target.  The dashed border
        # advertises the empty state without competing with the
        # purple-solid border used for populated containers; the lower
        # opacity surface tint keeps the empty container visually
        # quieter than its populated siblings.
        {
            "selector": "node.dag-block--container.dag-block--container-empty",
            "style": {
                "border-style": "dashed",
                "border-color": OI_PURPLE,
                "border-width": 1.5,
                "background-opacity": 0.1,
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

    Main I/O ports: every ribbon node renders two main I/O ports as
    additional cytoscape elements — a blue circle on the LEFT edge
    (input) and one on the RIGHT edge (output). Image-flow edges connect
    upstream output to downstream input so the wire visibly enters and
    exits each operation.

    Args:
        scope: The :class:`BuilderScope` currently in view.
        selected_node_id: If set, the matching node gets the
            ``"selected"`` class so the stylesheet highlights it.

    Returns:
        Ordered list of cytoscape element dicts (nodes + edges + port
        markers). Layout is ``"preset"`` — Python computes ``(x, y)``
        for every element so callbacks can stay layout-agnostic.
    """

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

    # Bottom-edge aux ports were the canvas anchor for the popover wire
    # flow (Phase 1-6). Phase 7 retired the popover; the legacy canvas
    # no longer renders aux-port markers because there is no surface to
    # open. Aux wiring lives on the DAG canvas (block_port elements +
    # the inspector aux-ports section).
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


#: Glyph + format for container title-bar labels (spec §4.4).  Expanded
#: containers carry the down chevron (▼); collapsed containers swap to
#: the right chevron (▶) so users can toggle between the two visual
#: states without reading the title text.
_DAG_CONTAINER_TITLE_EXPANDED = "▼ Pipeline — {label}"
_DAG_CONTAINER_TITLE_COLLAPSED = "▶ Pipeline — {label}"


def _count_inner_ops(nested: Optional["_DagBuilderScope"]) -> int:
    """Return the number of non-InputImage blocks in a nested scope.

    Spec §4.4 — collapsed containers surface a chain-glyph indicator
    showing how many ops live inside.  The count excludes the auto-
    seeded ``InputImage`` sentinel so a fresh container reads as
    ``0 ops`` rather than ``1 op``.  Nested ``Pipeline`` containers
    count once each (the user thinks of the aux pipeline as a single
    composed unit, not as the sum of its inner blocks).
    """

    if nested is None:
        return 0
    return sum(1 for b in nested.blocks if b.class_name != INPUT_IMAGE_CLASS_NAME)


def _scope_has_only_input_image(scope: "_DagBuilderScope") -> bool:
    """Return ``True`` when *scope* holds only the auto-seeded InputImage.

    Spec §4.8 calls for an empty-container placeholder (``+ drop ops
    here``) when the user creates a fresh container from
    ``+ New Pipeline`` and has not yet dropped any real ops inside.  The
    inner scope is "empty" iff every block is the ``InputImage``
    sentinel.

    Args:
        scope: The container's nested :class:`_DagBuilderScope`.

    Returns:
        ``True`` when every block in *scope* is an ``InputImage``
        sentinel (typical case: exactly one block, the auto-seeded
        sentinel); ``False`` once any real op block has been added.
    """

    return all(
        b.class_name == INPUT_IMAGE_CLASS_NAME for b in scope.blocks
    )


def _build_consumer_fed_dot_subnode(
    block: "BlockNode", *, container_block_id: str
) -> dict:
    """Build the consumer-fed dot element for a nested-scope InputImage.

    Per spec §4.1 + §4.4, the container scope's ``InputImage`` sentinel
    is NOT rendered as the same big green chevron used at the root
    scope; instead it surfaces as a small purple dot anchored to the
    container's inner-left edge.  Cytoscape compound parenting positions
    the dot inside the container via ``data.parent``.

    Args:
        block: The nested ``InputImage`` :class:`BlockNode`.
        container_block_id: ``BlockNode.block_id`` of the enclosing
            container; assigned as the cytoscape ``parent`` so the dot
            renders inside the container's bounding box.

    Returns:
        Cytoscape element dict ready to append to the canvas elements.
    """

    return {
        "data": {
            "id": block.block_id,
            "block_id": block.block_id,
            "class_name": INPUT_IMAGE_CLASS_NAME,
            "label": "",
            "parent": container_block_id,
            "is_consumer_fed": True,
        },
        "classes": "dag-block dag-block__consumer-fed-dot",
        "selectable": True,
        "grabbable": False,
    }


def _build_container_placeholder_subnode(
    container_block_id: str,
) -> dict:
    """Build the ``+ drop ops here`` placeholder for an empty container.

    Spec §4.8: when a container's nested scope holds only the auto-
    seeded ``InputImage`` (no real ops), the canvas surfaces a label-
    only sub-node with the hint text so users know the container is a
    valid drop target.  The element is non-selectable so it doesn't
    fight the container's own title-bar click handler.

    Args:
        container_block_id: ``BlockNode.block_id`` of the enclosing
            empty container.

    Returns:
        Cytoscape element dict for the placeholder hint.
    """

    return {
        "data": {
            "id": f"placeholder__{container_block_id}",
            "block_id": container_block_id,
            "label": "+ drop ops here",
            "parent": container_block_id,
            "is_placeholder": True,
        },
        "classes": "dag-block__placeholder",
        "selectable": False,
        "grabbable": False,
    }


def _collect_container_issues(
    scope: "_DagBuilderScope",
    issue_by_block: Dict[str, List[Any]],
) -> Tuple[int, int]:
    """Return ``(error_count, hint_count)`` aggregated over *scope* + nested.

    Per spec §4.4 a collapsed container surfaces a single badge with
    the inner scope's error+hint counts.  The same aggregate also
    powers the inspector container card.  Walks the scope's blocks
    recursively so issues attached to any descendant block count.

    Args:
        scope: The container's nested :class:`_DagBuilderScope`.
        issue_by_block: Pre-built mapping of ``block_id`` → issues
            (already bucketed by the outer renderer; reused here for
            O(1) lookup per block instead of re-walking the issue list).

    Returns:
        ``(error_count, hint_count)`` tuple — severities other than
        ``"advisory"`` count as errors; ``"advisory"`` count as hints.
    """

    errors = 0
    hints = 0
    stack: List["_DagBuilderScope"] = [scope]
    while stack:
        current = stack.pop()
        for block in current.blocks:
            for iss in issue_by_block.get(block.block_id, []):
                if getattr(iss, "severity", "error") == "advisory":
                    hints += 1
                else:
                    errors += 1
            if block.nested is not None:
                stack.append(block.nested)
    return errors, hints


def _bucket_issues_recursively(
    scope: "_DagBuilderScope",
    issues: List[Any],
) -> Dict[str, List[Any]]:
    """Bucket every issue by ``block_id`` regardless of scope depth.

    Spec §4.6 validates recursively — each container's nested scope
    runs the same six blocking + one advisory rules.  The renderer
    needs the union of issues from every scope so badges decorate the
    right block at the right depth.  This helper builds a flat
    ``block_id → [Issue, ...]`` map that the per-scope renderer reuses.

    Args:
        scope: Root scope to walk.
        issues: Flat list of :class:`~phenotypic.gui.builder._validation.Issue`
            instances.

    Returns:
        Mapping from ``block_id`` to a list of issues whose
        ``Issue.block_id`` matches a block reachable from *scope*.
        Issues without a ``block_id`` (scope-level rules — Rule 6) or
        whose id is stale (no matching block in the tree) drop here;
        the toolbar badge surfaces them separately.
    """

    valid_ids: set[str] = set()
    stack: List["_DagBuilderScope"] = [scope]
    while stack:
        current = stack.pop()
        for block in current.blocks:
            valid_ids.add(block.block_id)
            if block.nested is not None:
                stack.append(block.nested)

    bucket: Dict[str, List[Any]] = {}
    for iss in issues:
        bid = getattr(iss, "block_id", None)
        if bid is None or bid not in valid_ids:
            continue
        bucket.setdefault(bid, []).append(iss)
    return bucket


def _emit_scope_elements(
    scope: "_DagBuilderScope",
    *,
    elements: List[dict],
    parent_container_block_id: Optional[str],
    parent_collapsed: bool,
    selected_block_id: Optional[str],
    selected_edge_id: Optional[str],
    issue_by_block: Dict[str, List[Any]],
    registry: Any,
    registry_key: int,
) -> None:
    """Emit cytoscape elements for one scope (root or nested).

    Recursively handles container blocks: a container emits itself as a
    compound parent + recurses into ``block.nested`` so child blocks
    carry ``data.parent`` pointing at the container's ``block_id``.

    The function mutates *elements* in place — callers create the list
    once and the helper appends to it.  This keeps the recursive
    traversal allocation-free per scope (avoids the cost of merging
    sub-lists for what is typically a 5-50 element pipeline).

    Args:
        scope: The :class:`_DagBuilderScope` being rendered.
        elements: Cytoscape elements accumulator (mutated in place).
        parent_container_block_id: ``BlockNode.block_id`` of the
            enclosing container, or ``None`` at the root scope.  Every
            block emitted by this helper carries ``data.parent =
            parent_container_block_id`` so cytoscape's compound layout
            nests children inside their container's bounding box.
        parent_collapsed: ``True`` when any ancestor container is
            collapsed; child elements get ``data.parent_collapsed`` so
            the stylesheet's ``[?parent_collapsed]`` selector hides
            them without re-emitting per-collapse toggle.
        selected_block_id / selected_edge_id: Selection ids (only the
            active scope highlights — defense in depth, since
            selection is mutually exclusive across scopes).
        issue_by_block: Pre-bucketed issue index for the entire tree.
        registry: Live operation registry; cached parameter metadata
            comes from here.
        registry_key: ``id(registry)`` discriminator for the cached
            accept-list lookups.
    """

    aux_consumed_block_ids: set[str] = set()
    main_path_edges: set[str] = set()
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
            image_edges_by_source.setdefault(
                edge.source_block_id, []
            ).append(edge)

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

    # ── 1. Emit one cytoscape node per block. ───────────────────────────
    for block in scope.blocks:
        # The container scope's InputImage sentinel renders as a small
        # purple consumer-fed dot, NOT as a regular block (spec §4.1 +
        # §4.4).  Root-scope InputImage keeps the big chevron treatment.
        if (
            block.class_name == INPUT_IMAGE_CLASS_NAME
            and parent_container_block_id is not None
        ):
            dot = _build_consumer_fed_dot_subnode(
                block, container_block_id=parent_container_block_id
            )
            if parent_collapsed:
                dot["data"]["parent_collapsed"] = True
            elements.append(dot)
            continue

        block_issues = issue_by_block.get(block.block_id, [])
        has_issue = bool(block_issues)
        issue_severity = "error"
        has_stub_issue = False
        if has_issue:
            severities = {
                getattr(iss, "severity", "error") for iss in block_issues
            }
            issue_severity = (
                "advisory" if severities == {"advisory"} else "error"
            )
            kinds = {getattr(iss, "kind", "") for iss in block_issues}
            has_stub_issue = "stub" in kinds

        # Compute container inner-issue aggregate ONCE per container —
        # the same (errors, hints) tuple feeds the outer-border severity
        # decision below, the collapsed-title chain-glyph suffix, and
        # the node_data ``inner_error_count`` / ``inner_hint_count``
        # fields.  ``_collect_container_issues`` walks the entire nested
        # subtree, so calling it more than once per container wastes work.
        inner_errors = 0
        inner_hints = 0
        if (
            block.class_name == PIPELINE_CLASS_NAME
            and block.nested is not None
        ):
            inner_errors, inner_hints = _collect_container_issues(
                block.nested, issue_by_block
            )
            if inner_errors > 0:
                has_issue = True
                issue_severity = "error"
            elif inner_hints > 0 and not has_issue:
                has_issue = True
                issue_severity = "advisory"

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
        if block.class_name == PIPELINE_CLASS_NAME and block.collapsed:
            classes.append("dag-block--collapsed")
        # Empty container (only the auto-seeded InputImage inside) gets a
        # marker class so the stylesheet can apply the dashed-outline
        # hint chrome (spec §4.8).  Only applies when expanded — collapsed
        # containers read as a compact 1-row block regardless of inner
        # content.
        if (
            block.class_name == PIPELINE_CLASS_NAME
            and not block.collapsed
            and block.nested is not None
            and _scope_has_only_input_image(block.nested)
        ):
            classes.append("dag-block--container-empty")

        if block.class_name == PIPELINE_CLASS_NAME:
            base_label = block.label or block.class_name
            template = (
                _DAG_CONTAINER_TITLE_COLLAPSED
                if block.collapsed
                else _DAG_CONTAINER_TITLE_EXPANDED
            )
            label = template.format(label=base_label)
            # Collapsed containers append a chain-glyph suffix showing the
            # inner-op count + aggregated issue count so the user reads
            # the inner state at a glance without expanding (spec §4.4).
            # Reuses the ``inner_errors`` / ``inner_hints`` aggregate
            # computed above — no second walk of the nested subtree.
            if block.collapsed and block.nested is not None:
                inner_op_count = _count_inner_ops(block.nested)
                suffix_parts: List[str] = []
                # Chain glyph + op count (always rendered so the user can
                # tell a 0-op collapsed container from a populated one).
                suffix_parts.append(
                    f"⬞ {inner_op_count} op"
                    f"{'s' if inner_op_count != 1 else ''}"
                )
                if inner_errors or inner_hints:
                    bits: List[str] = []
                    if inner_errors:
                        bits.append(
                            f"{inner_errors} issue"
                            f"{'s' if inner_errors != 1 else ''}"
                        )
                    if inner_hints:
                        bits.append(
                            f"{inner_hints} hint"
                            f"{'s' if inner_hints != 1 else ''}"
                        )
                    suffix_parts.append(", ".join(bits))
                label = f"{label}  ({' • '.join(suffix_parts)})"
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
            "parent": parent_container_block_id,
        }
        if block.class_name == PIPELINE_CLASS_NAME:
            node_data["is_container"] = True
            node_data["collapsed"] = block.collapsed
            # ``inner_errors`` / ``inner_hints`` were computed once at
            # the top of this branch (or default to 0 when nested is
            # absent); reuse them rather than re-walking the subtree.
            node_data["inner_error_count"] = inner_errors
            node_data["inner_hint_count"] = inner_hints
        if parent_collapsed:
            node_data["parent_collapsed"] = True
        elements.append(
            {
                "data": node_data,
                "classes": " ".join(classes),
                "grabbable": True,
                "selectable": True,
            }
        )

    # ── 2. Port sub-nodes per non-InputImage / non-container block. ────
    for block in scope.blocks:
        if (
            block.class_name == INPUT_IMAGE_CLASS_NAME
            and parent_container_block_id is not None
        ):
            # Consumer-fed dot already emitted with no ports.
            continue
        if block.class_name != INPUT_IMAGE_CLASS_NAME:
            port = _build_image_port_subnode(
                block.block_id,
                "in",
                "image-in",
                css_class="dag-port--input",
            )
            if parent_collapsed:
                port["data"]["parent_collapsed"] = True
            elements.append(port)
        port = _build_image_port_subnode(
            block.block_id,
            "out",
            "image-out",
            css_class="dag-port--output",
        )
        if parent_collapsed:
            port["data"]["parent_collapsed"] = True
        elements.append(port)

        if block.class_name in (
            INPUT_IMAGE_CLASS_NAME,
            PIPELINE_CLASS_NAME,
        ):
            continue
        info = registry.get(block.class_name)
        if info is None:
            continue
        wired_count_by_port = aux_wired_counts.get(block.block_id, {})
        for param_name, param_info in info.parameters.items():
            if not (param_info.is_operation or param_info.is_pipeline):
                continue
            cached_accepts = _resolve_dag_accepts_for_class_port(
                block.class_name, param_name, registry_key
            )
            if cached_accepts is None:
                accepts: List[str] = _resolve_dag_accepts(
                    param_info, registry
                )
            else:
                accepts = list(cached_accepts)
            wired_count = wired_count_by_port.get(param_name, 0)
            wired = wired_count > 0
            required = not param_info.has_default
            port_classes = _aux_port_classes(
                wired=wired,
                required=required,
                is_list=param_info.is_list,
            )
            port_data: Dict[str, Any] = {
                "id": ids.block_port_id(block.block_id, param_name),
                "parent": block.block_id,
                "block_id": block.block_id,
                "port": param_name,
                "port_kind": "aux",
                "is_port": True,
                "is_list": param_info.is_list,
                "is_required": required,
                "accepts": accepts,
                "wired_count": wired_count,
            }
            if parent_collapsed:
                port_data["parent_collapsed"] = True
            elements.append(
                {
                    "data": port_data,
                    "classes": " ".join(port_classes),
                    "selectable": False,
                    "grabbable": False,
                }
            )

    # ── 3. Edges within this scope. ────────────────────────────────────
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
        edge_data: Dict[str, Any] = {
            "id": ids.edge_id(edge.edge_id),
            "source": edge.source_block_id,
            "target": edge.target_block_id,
            "edge_id": edge.edge_id,
            "kind": edge.kind,
            "target_slot": edge.target_slot,
            "target_port": edge.target_port,
            "source_port": edge.source_port,
            "is_main": edge.edge_id in main_path_edges,
        }
        if parent_collapsed:
            edge_data["parent_collapsed"] = True
        elements.append(
            {
                "data": edge_data,
                "classes": " ".join(edge_classes),
                "selectable": True,
                "grabbable": False,
            }
        )

    # ── 4. Recurse into container blocks. ──────────────────────────────
    for block in scope.blocks:
        if (
            block.class_name != PIPELINE_CLASS_NAME
            or block.nested is None
        ):
            continue
        nested_scope = block.nested
        # Empty-container placeholder: render the hint inside the
        # container's body when the nested scope holds only the
        # auto-seeded InputImage sentinel (spec §4.8).
        if (
            _scope_has_only_input_image(nested_scope)
            and not block.collapsed
        ):
            placeholder = _build_container_placeholder_subnode(
                block.block_id
            )
            if parent_collapsed:
                placeholder["data"]["parent_collapsed"] = True
            elements.append(placeholder)
        _emit_scope_elements(
            nested_scope,
            elements=elements,
            parent_container_block_id=block.block_id,
            parent_collapsed=parent_collapsed or block.collapsed,
            selected_block_id=selected_block_id,
            selected_edge_id=selected_edge_id,
            issue_by_block=issue_by_block,
            registry=registry,
            registry_key=registry_key,
        )


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
    * **Container compound parents.** Container blocks
      (``class_name == PIPELINE_CLASS_NAME``) recurse into their
      ``nested`` scope; child elements carry ``data.parent =
      container_block_id`` so cytoscape's compound layout nests them
      visually.  Container scopes' ``InputImage`` sentinel renders as
      a small purple consumer-fed dot (NOT the big root-scope
      chevron).  Empty containers surface a ``+ drop ops here``
      placeholder.  Collapsed containers render as a 1-row block
      whose data carries ``inner_error_count`` / ``inner_hint_count``
      so the aggregated badge reads ``▣ N issues, M hints``.

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

    from phenotypic.gui._operation_registry import get_registry

    registry = get_registry()
    registry_key = id(registry)
    elements: List[dict] = []
    issues = list(issues or [])

    issue_by_block = _bucket_issues_recursively(scope, issues)

    _emit_scope_elements(
        scope,
        elements=elements,
        parent_container_block_id=None,
        parent_collapsed=False,
        selected_block_id=selected_block_id,
        selected_edge_id=selected_edge_id,
        issue_by_block=issue_by_block,
        registry=registry,
        registry_key=registry_key,
    )

    # Issue badges: one per block with one or more issues.  Walks the
    # full tree's bucket so nested-block badges appear too.  The
    # cytoscape stylesheet's ``[?parent_collapsed]`` hides badges that
    # belong to a hidden child; the aggregate count surfaces on the
    # container instead via ``inner_error_count`` / ``inner_hint_count``.
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
        elements=build_canvas_elements_dag(
            scope, selected_block_id=selected_node_id
        ),
        # ``build_canvas_elements_dag`` emits positionless elements;
        # ``viewport_ops.js`` runs ``cytoscape-dagre`` on first paint
        # when the extension is registered.  ``breadthfirst`` is the
        # core-cytoscape fallback so the canvas still lays blocks out
        # left-to-right even when the dagre asset fails to register.
        layout={
            "name": "breadthfirst",
            "directed": True,
            "fit": True,
            "padding": 24,
            "spacingFactor": 1.4,
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


# ---------------------------------------------------------------------------
# Issue badge (spec §4.6)
# ---------------------------------------------------------------------------
#
# The toolbar issue badge surfaces the aggregated validation findings in
# a single chip — clicking opens a popover listing one row per issue.
# The ``revalidate_on_state_change`` callback feeds STORE_ISSUES; the
# badge label and tooltip rows are computed from that store and drive
# the click → scroll_to dispatch chain (consumed by the clientside
# ``phenotypicScrollTo`` chain in ``viewport_ops.js``).

#: Display names for each :class:`Issue.kind` shown in tooltip rows.
#: Mirrors the spec §4.6 short-name table.  Defined as a module-level
#: constant so the tests can monkey-patch / introspect without
#: duplicating the mapping.
_ISSUE_RULE_SHORT_NAMES: Dict[str, str] = {
    "fork": "Fork",
    "stub": "Unreachable",
    "required_aux": "Missing aux",
    "cycle": "Cycle",
    "container_mode": "Container mode",
    "missing_input": "No Input Image",
    "duplicate_input": "Extra Input Image",
    "stage_order_hint": "Stage order",
    "unknown_class": "Unknown class",
}


def _format_issue_badge_label(n_issues: int, n_hints: int) -> str:
    """Render the toolbar badge label per spec §4.6.

    Examples (all keyed to the singular/plural rules called out in the
    spec):

    * ``(0, 0) -> "0 issues"``
    * ``(1, 0) -> "1 issue"``
    * ``(3, 0) -> "3 issues"``
    * ``(0, 1) -> "0 issues, 1 hint"``
    * ``(1, 1) -> "1 issue, 1 hint"``
    * ``(3, 2) -> "3 issues, 2 hints"``

    Args:
        n_issues: Count of ``severity == "error"`` issues.
        n_hints: Count of ``severity == "advisory"`` issues.

    Returns:
        Human-readable label suitable for the badge component.
    """

    issue_word = "issue" if n_issues == 1 else "issues"
    if n_hints == 0:
        return f"{n_issues} {issue_word}"
    hint_word = "hint" if n_hints == 1 else "hints"
    return f"{n_issues} {issue_word}, {n_hints} {hint_word}"


def _issue_row_block_label(
    issue: Dict[str, Any], state: Optional[BuilderState]
) -> str:
    """Resolve a human-readable block label for a tooltip row.

    Walks the DAG state to find the offender ``BlockNode`` so the row
    can show ``"GaussianBlur#abc12345"`` (label or class_name + short
    block_id suffix) rather than a raw 32-character UUID.  Falls back
    to the rule short name for scope-level issues
    (``missing_input`` has ``block_id == None``).

    Args:
        issue: Issue dict as published to :data:`STORE_ISSUES`.
        state: Live :class:`BuilderState` (DAG schema) — used to resolve
            ``block_id`` → ``BlockNode``.  ``None`` falls back to a
            shortened block_id literal.

    Returns:
        Label text rendered inside the tooltip row.
    """

    block_id = issue.get("block_id")
    if block_id is None:
        return "Scope"
    if state is None:
        return f"Block {str(block_id)[:8]}"
    scope_path = issue.get("scope_path") or []
    scope = state.root
    for parent_id in scope_path:
        parent = next(
            (b for b in getattr(scope, "blocks", []) if b.block_id == parent_id),
            None,
        )
        if parent is None or parent.nested is None:
            break
        scope = parent.nested
    block = next(
        (b for b in getattr(scope, "blocks", []) if b.block_id == block_id),
        None,
    )
    if block is None:
        return f"Block {str(block_id)[:8]}"
    if block.label:
        return block.label
    return block.class_name


def _sort_issues_for_badge(
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order issues for the tooltip per spec §4.6.

    Rules:

    * Issues (severity == ``"error"``) come before hints
      (severity == ``"advisory"``).
    * Within each severity bucket, sort alphabetically by ``kind``.
    * Original list order acts as a stable tiebreaker so reproducing the
      tooltip across two renders of the same state never permutes rows.

    Args:
        issues: Raw issue list from :data:`STORE_ISSUES`.

    Returns:
        New sorted list (input is not mutated).
    """

    enumerated = sorted(
        enumerate(issues),
        key=lambda pair: (
            0 if pair[1].get("severity", "error") == "error" else 1,
            pair[1].get("kind", ""),
            pair[0],
        ),
    )
    return [pair[1] for pair in enumerated]


def build_issue_badge(
    issues: Optional[List[Dict[str, Any]]] = None,
    state: Optional[BuilderState] = None,
) -> html.Span:
    """Render the toolbar issue badge + its popover-style tooltip.

    The badge is a count chip whose label follows the §4.6 grammar
    ("``N issues``" or "``N issues, M hints``"); the tooltip target is
    a :class:`dbc.Popover` listing one row per issue.  Each row carries
    a pattern-matched id from :func:`issue_row_id` so a server-side
    callback can dispatch ``scroll_to`` against the offender on click.

    Args:
        issues: Live issue list (matches :data:`STORE_ISSUES` payload
            schema — list of dicts with ``kind`` / ``block_id`` /
            ``detail`` / ``scope_path`` / ``severity`` keys).  Defaults
            to an empty list so first-paint renders with ``"0 issues"``.
        state: Optional :class:`BuilderState` used to resolve
            ``block_id`` → ``BlockNode.label`` / ``class_name`` for
            the row's left column.  Falls back to a short-uuid literal
            when ``None`` or when the block has been deleted since the
            issues snapshot.

    Returns:
        An :class:`html.Span` carrying the badge + the popover.  Mounted
        once per builder render into the canvas toolbar header next to
        the :data:`BTN_RELAYOUT` button.
    """

    issue_list = list(issues or [])
    sorted_issues = _sort_issues_for_badge(issue_list)
    # Single pass severity tally — avoids two extra O(N) sweeps after sort
    # (each ``sum`` would otherwise iterate the full sorted list).  The
    # default severity ``"error"`` matches the sort key so untagged
    # legacy issues keep counting as blocking.
    n_issues = 0
    n_hints = 0
    for issue in sorted_issues:
        severity = issue.get("severity", "error")
        if severity == "error":
            n_issues += 1
        elif severity == "advisory":
            n_hints += 1
    label = _format_issue_badge_label(n_issues, n_hints)

    # Colour signals severity: red badge when there are blocking issues,
    # warning when only hints remain, secondary (grey) when fully clean.
    if n_issues > 0:
        color = "danger"
    elif n_hints > 0:
        color = "warning"
    else:
        color = "secondary"

    # Memoise (scope_path, block_id) -> rendered block label.  Each
    # ``_issue_row_block_label`` call walks the scope tree O(depth × |blocks|);
    # repeated issues on the same block (typical for fork / cycle rules
    # which emit multiple findings per offender) would otherwise repeat
    # the same walk.  The cache key is the tuple form of ``scope_path``
    # (which is JSON-list shaped on the wire) plus ``block_id``.
    label_cache: Dict[Tuple[Tuple[str, ...], Optional[str]], str] = {}
    rows: List[Any] = []
    for idx, issue in enumerate(sorted_issues):
        kind = str(issue.get("kind", ""))
        rule_name = _ISSUE_RULE_SHORT_NAMES.get(kind, kind)
        cache_key = (
            tuple(issue.get("scope_path") or ()),
            issue.get("block_id"),
        )
        block_label = label_cache.get(cache_key)
        if block_label is None:
            block_label = _issue_row_block_label(issue, state)
            label_cache[cache_key] = block_label
        detail = str(issue.get("detail", ""))
        rows.append(
            html.Div(
                [
                    html.Span(
                        block_label,
                        className="issue-row-block fw-bold me-2",
                    ),
                    html.Span(
                        rule_name,
                        className="issue-row-rule text-muted me-2",
                    ),
                    html.Span(
                        detail,
                        className="issue-row-detail small",
                    ),
                ],
                id=ids.issue_row_id(issue.get("block_id"), kind, idx),
                n_clicks=0,
                className="issue-row d-flex align-items-baseline px-2 py-1",
                style={"cursor": "pointer"},
                **{  # type: ignore[arg-type]
                    "data-testid": "issue-row",
                    "data-rule": kind,
                },
            )
        )

    if not rows:
        rows = [
            html.Div(
                "No issues",
                className="issue-row-empty text-muted small px-2 py-1",
            )
        ]

    # Wrap the chip in a ``html.Span`` so the spec's
    # ``data-testid="issue-badge"`` attribute can hang off the
    # outermost clickable surface — ``dbc.Badge`` has a strict prop
    # allowlist and rejects unknown ``data-*`` kwargs.  The badge
    # itself still carries the static id so the row callback can
    # subscribe via ``Input(ids.ISSUE_BADGE, "n_clicks")`` if a future
    # phase wants a chip-level click target (e.g. to open a modal
    # listing every issue across every scope).
    badge_chip = dbc.Badge(
        label,
        id=ids.ISSUE_BADGE,
        color=color,
        className="issue-badge",
        style={"cursor": "pointer"},
        n_clicks=0,
    )
    badge_wrapper = html.Span(
        badge_chip,
        className="issue-badge-target d-inline-flex align-items-center",
        **{"data-testid": "issue-badge"},  # type: ignore[arg-type]
    )

    popover = dbc.Popover(
        [
            dbc.PopoverHeader("Validation issues"),
            dbc.PopoverBody(rows, className="p-1"),
        ],
        id=ids.ISSUE_BADGE_TOOLTIP,
        target=ids.ISSUE_BADGE,
        trigger="hover focus",
        placement="bottom",
    )

    return html.Span([badge_wrapper, popover], className="issue-badge-wrap")


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
    # The button is mounted on every render path so
    # ``asset_status_disables``'s output always resolves.
    relayout_btn = dbc.Button(
        "Re-layout",
        id=ids.BTN_RELAYOUT,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Re-run the dagre layout pass and fit to viewport",
    )

    # Issue badge sits at the rightmost end of the toolbar so it draws
    # the user's eye when validation flips red.  Mounted on every render
    # path; the live count + tooltip rows are wired by the
    # ``revalidate_on_state_change`` → ``update_issue_badge`` callback
    # against ``STORE_ISSUES``.  An initial empty list is rendered here so
    # the badge has stable chrome (``"0 issues"``, secondary colour) on
    # the first paint before any state has been published.
    issue_badge = build_issue_badge(issues=[], state=None)

    header = html.Div(
        [
            html.H6("Canvas", className="mb-0"),
            html.Div(
                [controls, delete_btn, relayout_btn, issue_badge],
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
        [header, banner, cytoscape_slot],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100%",
            "minHeight": 0,
            "position": "relative",
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


# ---------------------------------------------------------------------------
# DAG inspector helpers — wire card + aux ports section
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
            # Container drill-in from the wire card lands in a follow-up
            # phase.  Mount the button so the affordance is visible but
            # disable it for now, with a tooltip explaining the gating.
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
                    title="Container drill-in from wire card is not yet wired",
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
    affordance.  The row ships with ``▲`` / ``▼`` arrow buttons as a
    drag-handle fallback (drag glue lands in a follow-up phase); the
    hidden reorder ``dcc.Store`` is mounted so the future drag
    handlers don't churn the inspector callback surface.

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
        # Drag-handle placeholder (spec §4.5 calls for one).  The
        # arrow-button fallback ships now; the handle carries the
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

    The card carries :data:`ids.INSPECTOR_EMPTY_STATE` so integration
    tests can assert that the empty branch is rendering, and so a
    future callback can subscribe to the placeholder for runtime
    chrome updates (e.g. swapping the hint copy on first-time-user
    detection).
    """

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Inspector", className="card-title"),
                html.P(
                    "Drag an operation from the palette to begin.",
                    className="mb-1",
                ),
                html.Small(
                    "The badge in the toolbar shows validation issues. "
                    "Run preview and Save are disabled when any errors "
                    "exist.",
                    className="text-muted",
                    style={"fontSize": FONT_SIZE_LABEL},
                ),
            ]
        ),
        id=ids.INSPECTOR_EMPTY_STATE,
        className="h-100",
    )


def _build_input_image_card(
    state: "_DagBuilderState",
    block: "BlockNode",
) -> html.Div:
    """Render the inspector card for the auto-seeded Input Image source.

    Spec §4.5 — Input Image inspector card surface:

    * **Heading**: "Input Image — pipeline source".
    * **Description**: one-paragraph explanation of the loader contract.
    * **Re-layout button** — clicking forwards to the toolbar's
      canonical :data:`ids.BTN_RELAYOUT`.  The inspector copy carries
      no Dash id (Dash forbids duplicate ids) and uses the
      ``data-relayout-proxy`` attribute as a clientside hook.
    * **Re-anchor button** (:data:`ids.BTN_REANCHOR`) — clicking
      dispatches a ``reanchor`` payload that pans / zooms the
      cytoscape viewport to centre on this Input Image block.
    * **No param form**: InputImage has no user-editable parameters
      (the image is supplied by the runtime loader, not the user).
    * **No Delete button**: spec §4.1 — the Input Image sentinel
      cannot be deleted because every scope must contain exactly
      one (Rule 6 of the validation suite).
    * **Hidden placeholders** for :data:`ids.INPUT_NODE_LABEL` /
      :data:`ids.BTN_DRILL_IN` / doc-section widgets so the existing
      fan-in callback's ``Input`` ids always resolve regardless of
      which inspector branch the user is looking at.

    Args:
        state: The full :class:`_DagBuilderState`. Reserved for a
            future enhancement that disables the buttons when the
            scope is in a degraded state (currently unused; suppresses
            a lint warning).
        block: The selected :class:`BlockNode`. Must satisfy
            ``block.class_name == INPUT_IMAGE_CLASS_NAME``.

    Returns:
        :class:`html.Div` ready to drop into :data:`ids.INSPECTOR_CONTAINER`.
    """

    # ``state`` / ``block`` are accepted to match the signature shape of
    # the other inspector-card builders; Input Image has no user-editable
    # fields so neither is consulted here.
    del state, block

    # ``BTN_RELAYOUT`` is owned by the canvas toolbar (always visible,
    # single instance) so it cannot be duplicated inside this card —
    # Dash forbids duplicate component ids in the live DOM and the
    # toolbar button stays mounted while the inspector card renders.
    # Spec §4.5 lists "Re-layout" and "Re-anchor" as the two affordances
    # on the Input Image card; we surface a "Re-layout (toolbar)" label
    # button here for discoverability (no Dash id — purely cosmetic; it
    # carries a ``className`` test selector so the integration tests can
    # assert the affordance is present) and bind the dispatching click
    # to the toolbar's canonical :data:`ids.BTN_RELAYOUT`.  The
    # Re-anchor button uses :data:`ids.BTN_REANCHOR` which lives only
    # inside this card so the clientside callback can dispatch the
    # ``reanchor`` payload through a single binding.
    body_children: List[Any] = [
        dbc.CardHeader(
            html.H4(
                "Input Image — pipeline source",
                className="mb-0",
            )
        ),
        dbc.CardBody(
            [
                html.P(
                    "Every op chain starts here. The image flowing out "
                    "of this block is whatever your run-time loader "
                    "provides — typically the file you point Run "
                    "preview at, or each image in the batch under a "
                    "production run.",
                    className="text-muted small mb-3",
                ),
                dbc.Button(
                    "Re-layout",
                    # No ``id=`` — Dash forbids duplicating the toolbar's
                    # ``BTN_RELAYOUT`` (always-mounted) inside this card.
                    # The button is purely cosmetic / signposting; a
                    # clientside delegation listener (registered in
                    # ``register_callbacks``) intercepts clicks on the
                    # ``inspector-input-image-relayout-btn`` class and
                    # forwards them to the toolbar's canonical
                    # ``BTN_RELAYOUT`` so the spec §4.5 affordance is
                    # functional from the inspector pane.
                    color="secondary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    className="me-2 inspector-input-image-relayout-btn",
                ),
                dbc.Button(
                    "Re-anchor view to Input Image",
                    id=ids.BTN_REANCHOR,
                    color="secondary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                ),
            ]
        ),
        # Hidden placeholders so the fan-in callback's
        # ``Input(BTN_DRILL_IN)`` / ``Input(INPUT_NODE_LABEL)`` /
        # ``Input(INSPECTOR_DOC_TOGGLE)`` ids stay resolvable even
        # though InputImage doesn't surface them.  Reuses
        # ``_hidden_inspector_widgets`` so the placeholder set stays
        # in sync with the other inspector branches.
        *_hidden_inspector_widgets(),
    ]

    return html.Div(
        dbc.Card(
            body_children,
            id=ids.INSPECTOR_INPUT_IMAGE_CARD,
            className="h-100",
        ),
        id=ids.INSPECTOR_CONTAINER,
    )


def _summarise_nested_scope(
    nested: Optional["_DagBuilderScope"],
) -> str:
    """Render the inner-scope summary string for the container card.

    Spec §4.5 calls for ``"3 ops, 1 aux pipeline"``-style summaries.
    The breakdown is computed from the registry-inferred stage of
    each non-InputImage block: ops/meas/post counts, and a separate
    nested-container count so the user can spot aux-of-aux structures.

    Args:
        nested: The container's :attr:`BlockNode.nested` scope, or
            ``None`` when the container has not yet been materialised
            (degraded state — surfaced as ``"empty"``).

    Returns:
        Human-readable summary, e.g. ``"2 ops, 1 measurement, 1 aux
        pipeline"``.  Returns ``"empty"`` for a scope holding only the
        auto-seeded ``InputImage`` (typical fresh-container case).
    """

    if nested is None:
        return "empty"
    op_count = 0
    meas_count = 0
    post_count = 0
    container_count = 0
    for block in nested.blocks:
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            continue
        if block.class_name == PIPELINE_CLASS_NAME:
            container_count += 1
            continue
        stage = _safe_stage(block.class_name)
        if stage == "meas":
            meas_count += 1
        elif stage == "post":
            post_count += 1
        else:
            op_count += 1
    parts: List[str] = []
    if op_count:
        parts.append(f"{op_count} op{'s' if op_count != 1 else ''}")
    if meas_count:
        parts.append(
            f"{meas_count} measurement{'s' if meas_count != 1 else ''}"
        )
    if post_count:
        parts.append(
            f"{post_count} post-step{'s' if post_count != 1 else ''}"
        )
    if container_count:
        parts.append(
            f"{container_count} aux pipeline"
            f"{'s' if container_count != 1 else ''}"
        )
    if not parts:
        return "empty"
    return ", ".join(parts)


def _build_container_inspector_card(
    state: "_DagBuilderState",
    block: "BlockNode",
) -> html.Div:
    """Render the inspector card for a selected container block.

    Spec §4.5 — container selection card:

    * **Label edit** — the visible title-bar text (``BlockNode.label``).
    * **Pipeline name / desc** — bound to the nested scope's
      ``BuilderScope.name`` / ``.desc``.  Suppressed when the container
      is missing a nested scope (degraded state).
    * **Inner scope summary** — e.g. ``"3 ops, 1 aux pipeline"``.
    * **Aggregated inner issue count** — sourced from the live
      :data:`STORE_ISSUES` (passed through to the rendered card by the
      enclosing callback).
    * **``Drill in →`` button** carrying :data:`BTN_DRILL_IN_CONTAINER`
      — wired in :func:`register_callbacks` to dispatch
      ``drill_into_container``.
    * **No ``nrows`` / ``ncols`` fields** — those only make sense at
      the root scope (spec §4.5).  Hidden ``INPUT_NROWS`` / ``INPUT_NCOLS``
      placeholders are emitted by the param form for non-container
      branches; on the container branch the inspector's enclosing
      build_inspector path is responsible.

    Args:
        state: The full :class:`_DagBuilderState`.  Needed in addition
            to *block* so the issue-count aggregate can be computed
            against the live store (Phase 6 will plumb the store into
            this card; for now the count is computed locally from
            ``state.toast_queue`` shape).
        block: The selected container :class:`BlockNode`.  Must satisfy
            ``block.class_name == PIPELINE_CLASS_NAME``.

    Returns:
        :class:`html.Div` ready to drop into ``INSPECTOR_CONTAINER``.
    """

    nested = block.nested
    nested_name = nested.name if nested is not None else ""
    nested_desc = nested.desc if nested is not None else ""
    summary = _summarise_nested_scope(nested)
    # ``state.toast_queue`` is unused here today; the aggregate issue
    # count is owned by Phase 6's STORE_ISSUES plumbing.  We emit a
    # placeholder row keyed for a future callback to fill.
    _unused_state = state  # noqa: F841 — reserved for Phase 6 wiring.

    label_value = block.label if block.label else block.class_name
    inner_block_count = _count_inner_ops(nested)

    header_children: List[Any] = [
        html.H5("Pipeline container", className="card-title mb-3"),
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
            className="mb-2",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Name"),
                dbc.Input(
                    id=ids.INPUT_CONTAINER_NAME,
                    type="text",
                    value=nested_name,
                    debounce=True,
                ),
            ],
            className="mb-2",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Description"),
                dbc.Textarea(
                    id=ids.INPUT_CONTAINER_DESC,
                    value=nested_desc,
                    rows=2,
                    style={"fontSize": FONT_SIZE_LABEL},
                ),
            ],
            className="mb-3",
        ),
    ]

    body_children: List[Any] = [
        *header_children,
        html.Div(
            [
                html.Span("Inner scope: ", className="text-muted small"),
                html.Strong(summary, className="small"),
            ],
            className="mb-1 inspector-container-summary",
        ),
        html.Div(
            [
                html.Span(
                    f"{inner_block_count} inner block"
                    f"{'s' if inner_block_count != 1 else ''}",
                    className="text-muted small",
                ),
            ],
            className="mb-2",
        ),
        # Aggregated inner-issue count placeholder.  A future phase will
        # wire STORE_ISSUES → this row; today we render the zero-state
        # text so the row is part of the visible chrome.
        html.Div(
            "0 inner issues",
            id=f"inspector-container-issues-{block.block_id}",
            className="inspector-container-issues text-muted small mb-3",
        ),
        dbc.Button(
            "Drill in →",
            id=ids.BTN_DRILL_IN_CONTAINER,
            color="primary",
            outline=True,
            n_clicks=0,
            className="me-2",
        ),
        # Hidden placeholder so the legacy fan-in callback's
        # ``Input(BTN_DRILL_IN)`` resolves even on the container card
        # branch; the dispatch path uses BTN_DRILL_IN_CONTAINER while
        # the legacy id stays mounted as a hidden anchor.
        dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
        html.Hr(),
        html.Div(id=ids.INSPECTOR_PARAM_FORM),
        html.Div(id=ids.INSPECTOR_PREVIEW, className="mt-3"),
        # Documentation section: containers have no docstring; emit
        # the hidden placeholders so the doc-toggle callback resolves.
        *_doc_section_widgets(None),
    ]

    return html.Div(
        dbc.Card(
            dbc.CardBody(body_children),
            className="h-100 inspector-container-card",
        ),
        id=ids.INSPECTOR_CONTAINER,
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

    # Input Image gets a dedicated info card (spec §4.5).  The card
    # carries Re-layout (label-only, proxied to the toolbar button)
    # and Re-anchor affordances + the loader-source description; no
    # param form (InputImage has no parameters) and no Delete button
    # (the Input Image sentinel cannot be deleted — spec §4.1).
    if block.class_name == INPUT_IMAGE_CLASS_NAME:
        return _build_input_image_card(state, block)

    # Container blocks render the dedicated container inspector card
    # (label + name + desc + summary + drill-in).  Spec §4.5 suppresses
    # ``nrows`` / ``ncols`` on container scopes — the
    # :func:`_build_container_inspector_card` helper emits only the
    # surface relevant to a Pipeline container.
    if block.class_name == PIPELINE_CLASS_NAME:
        return _build_container_inspector_card(state, block)

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
      ``test_legacy_pipeline_json`` migration tests still hit a working
      legacy inspector code path.  No active runtime callback feeds
      legacy state into this branch since Phase 8.

    Args:
        state: The full builder state (legacy or DAG schema).
        registry: Operation registry consulted for parameter metadata.

    Returns:
        A :class:`dash.html.Div` wrapping the inspector card. Always carries
        the :data:`INSPECTOR_CONTAINER` id so callbacks can swap children.
    """

    # Duck-typed dispatch — same pattern as ``state_to_json`` in _state.py
    # to stay resilient against importlib.reload in tests.  The
    # ``# type: ignore[arg-type]`` is needed because mypy can't narrow
    # the runtime branch; only the legacy fixture tests ever reach the
    # ``selected_node_id`` block below.
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

    # The legacy popover-era ``inspector_focus_aux`` override was retired
    # in Phase 7; the inspector always mirrors the canvas-selected
    # consumer's params now.
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

    if render_node.class_name == PIPELINE_CLASS_NAME:
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

    op_info = registry.get(render_node.class_name)
    if op_info is None:
        form: Any = html.Div(
            f"Unknown operation '{render_node.class_name}'. "
            "It may have been removed from the registry.",
            className="text-warning",
        )
    else:
        form = html.Div(
            param_form(
                op_info,
                current_values=render_node.params,
                form_id_prefix=render_node.node_id,
            ),
            id=ids.INSPECTOR_PARAM_FORM,
        )

    body_children = [
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

    # ``+ New Pipeline`` adds a nested ImagePipeline container — conceptually
    # a palette item, but kept as a sticky button above the Operations
    # accordion so it's never hidden by a collapsed section.  The DAG
    # palette factory carries the ``draggable`` + ``data-palette-class``
    # attributes that ``palette_dnd.js`` looks for; the same ``id`` keeps
    # the keyboard-fallback callback working when the user clicks the
    # button rather than dragging it.
    operations_section = html.Div(
        [
            html.H6("Operations", className="mb-2"),
            build_new_pipeline_palette_button(),
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
    # Duck-type the selection field: DAG state carries ``selected_block_id``;
    # legacy state carries ``selected_node_id``.  Either works for the
    # initial-paint preset positioning (both fall through to dagre on the
    # next mutation anyway).
    initial_selection = getattr(state, "selected_node_id", None) or getattr(
        state, "selected_block_id", None
    )
    top_half = html.Div(
        build_canvas_section(state.root, initial_selection),
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
            # DAG-redesign stores (spec §6).  Mounted on every render
            # path so the DAG callbacks never error on missing inputs.
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
            # Wire-drawing store: written by ``assets/wire_drawing.js``
            # on edge gestures + by the inspector wire / aux cards for
            # keyboard / button-driven mutations.  Carries a
            # discriminated-union payload routed by ``payload["kind"]``
            # to the appropriate ``edge_*`` / ``list_aux_*`` /
            # ``wire_select`` / ``block_select`` dispatch.
            dcc.Store(
                id=ids.STORE_EDGE_EVENT,
                data=None,
            ),
            # dash-cytoscape #106 workaround: every mutation callback
            # writes its JSON-serialised elements list straight into this
            # hidden element's ``children``; a MutationObserver in
            # viewport_ops.js watches it and reconciles the live
            # cytoscape graph.  The ``CANVAS_CYTOSCAPE`` ``elements`` prop
            # is never written by a callback (its diff drops edges on a
            # wholly-new list), and the callbacks write this DOM element
            # directly rather than via an intermediate ``dcc.Store``
            # because dash-renderer coalesces rapid store-update ->
            # downstream-callback chains and drops updates, whereas
            # MutationObserver delivers every DOM mutation.
            html.Div(
                id=ids.CANVAS_ELEMENTS_BRIDGE,
                style={"display": "none"},
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
    "build_new_pipeline_palette_button",
    "build_canvas",
    "build_canvas_elements",
    "build_canvas_elements_dag",
    "build_canvas_section",
    "build_inspector",
    "build_breadcrumb",
    "build_footer",
    "build_app_layout",
    "build_issue_badge",
    "build_asset_status_banner",
    "build_confirm_delete_modal",
]

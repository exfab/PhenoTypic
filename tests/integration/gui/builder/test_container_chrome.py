"""Integration tests for the Pipeline container chrome (Phase 5A).

Covers the visual + structural contract owned by Agent 5A in spec
§4.4 + §4.5:

* The compound parent element emitted for a container block carries
  the ``dag-block--container`` cytoscape class, ``data.label`` set to
  the inner scope's pipeline name (via the title-bar template), and
  ``data.is_container`` so clientside JS can target it.
* Inner blocks rendered inside a container scope set ``data.parent``
  to the enclosing container's ``block_id`` so cytoscape's compound
  layout positions them inside the container's bounding box.
* When a container is collapsed, child elements either drop their
  ``data.parent`` reference OR carry ``data.parent_collapsed`` /
  ``display: none``-compatible state so they read as hidden.
* When a container's nested scope holds only the auto-seeded
  ``InputImage`` sentinel, a ``+ drop ops here`` placeholder
  surfaces (spec §4.8).
* Selecting a container block renders the container inspector card —
  pipeline name + desc + inner-summary + ``Drill in →`` button —
  but NOT the ``nrows`` / ``ncols`` fields (root-scope only per
  spec §4.5).

Server-side only — no browser needed.  These tests run alongside the
existing ``test_canvas_render.py`` + ``test_container_recursion.py``
suites and complement the chrome-side coverage they don't address.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import (
    _build_container_inspector_card,
    _build_dag_inspector,
    build_canvas_elements_dag,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_from_json,
)

# Component-tree walking helpers shared with the other inspector tests.
from .conftest import _collect_text, _find_by_id


FIXTURE_DIR = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "builder_dag"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_state(name: str) -> Any:
    """Load a fixture by stem name and return the BuilderState."""

    return state_from_json(
        json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    )


def _block_elements(elements: List[dict]) -> List[dict]:
    """Filter to cytoscape elements that represent blocks (not ports / edges)."""

    return [
        e for e in elements
        if "class_name" in (e.get("data") or {})
        and not (e.get("data") or {}).get("is_port")
    ]


def _classes_of(elem: dict) -> List[str]:
    """Split a cytoscape element's ``classes`` string into a list."""

    return (elem.get("classes") or "").split()


def _elements_by_id(elements: List[dict]) -> Dict[str, dict]:
    """Index cytoscape elements by ``data.id``."""

    out: Dict[str, dict] = {}
    for elem in elements:
        eid = (elem.get("data") or {}).get("id")
        if eid is not None:
            out[eid] = elem
    return out


# ---------------------------------------------------------------------------
# Container compound parent emission (spec §4.4)
# ---------------------------------------------------------------------------


def test_container_compound_parent_emits_with_container_class() -> None:
    """Container blocks carry ``dag-block--container`` + container metadata.

    Spec §4.4 — Pipeline containers render as cytoscape compound parents
    with the ``dag-block--container`` class.  The renderer also sets
    ``data.is_container`` so clientside JS (drag adoption, collapse
    toggling) can target containers without walking class lists.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)

    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    elem = idx[container.block_id]
    classes = _classes_of(elem)
    assert "dag-block--container" in classes, (
        f"container block {container.block_id} missing dag-block--container "
        f"class; got {classes!r}"
    )
    assert elem["data"].get("is_container") is True, (
        f"container element should expose data.is_container=True; "
        f"got {elem['data']!r}"
    )


def test_container_title_bar_label_carries_pipeline_name() -> None:
    """The container's cytoscape label reads as the title-bar template.

    Spec §4.4 — the expanded container's title bar is
    ``▼ Pipeline — <name>``.  The renderer wires the cytoscape
    ``data.label`` to that template + the block's user-editable
    ``label`` (which falls back to the class name when blank).
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)

    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    elem = idx[container.block_id]
    label = elem["data"]["label"]
    base = container.label or container.class_name
    # Expanded containers carry the down chevron.
    assert label == f"▼ Pipeline -- {base}", (
        f"unexpected container label {label!r}; expected the expanded "
        f"chevron template ▼ Pipeline -- {base}"
    )


def test_collapsed_container_title_bar_uses_right_chevron() -> None:
    """A collapsed container swaps the down chevron for a right chevron.

    Spec §4.4 — the collapse state is encoded in the title-bar
    glyph (``▼`` expanded vs. ``▶`` collapsed) so the user reads the
    state from the text without needing to inspect the bounding box.
    The collapsed label additionally carries a chain-glyph suffix with
    the inner-op count (e.g. ``⬞ 2 ops``) so the user reads the inner
    state at a glance.  The renderer also adds the
    ``dag-block--collapsed`` class so the stylesheet can apply the
    compact 1-row chrome.
    """

    block_id = _new_block_id()
    container = BlockNode(
        block_id=block_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="sub_pipeline",
        nested=_DagBuilderScope(name="sub_pipeline"),
        collapsed=True,
    )
    scope = _DagBuilderScope(blocks=[container])
    elements = build_canvas_elements_dag(scope)
    idx = _elements_by_id(elements)

    elem = idx[block_id]
    classes = _classes_of(elem)
    assert "dag-block--collapsed" in classes
    label = elem["data"]["label"]
    # The right chevron + base label is the stable prefix; collapsed
    # containers also append the chain-glyph suffix with the inner-op
    # count.  An empty container reads as ``⬞ 0 ops``.
    assert label.startswith("▶ Pipeline -- sub_pipeline"), (
        f"collapsed container label should start with the right "
        f"chevron template; got {label!r}"
    )
    assert "0 op" in label, (
        f"collapsed container label should include the chain-glyph "
        f"inner-op count; got {label!r}"
    )


def test_collapsed_container_chain_glyph_counts_inner_ops() -> None:
    """The collapsed chain glyph counts non-InputImage inner blocks.

    Spec §4.4 — the chain glyph reads as ``⬞ N ops`` where N is the
    number of non-InputImage blocks in the nested scope.  Nested
    ``Pipeline`` containers count once each (the user treats an aux
    pipeline as a single composed unit, not as the sum of its
    descendants).
    """

    inner_one = BlockNode(
        block_id=_new_block_id(),
        class_name="GaussianBlur",
        params={},
    )
    inner_two = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    nested = _DagBuilderScope(
        blocks=[inner_one, inner_two],
        name="2_op_pipe",
    )
    container = BlockNode(
        block_id=_new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="pipeline_two",
        nested=nested,
        collapsed=True,
    )
    scope = _DagBuilderScope(blocks=[container])
    elements = build_canvas_elements_dag(scope)
    idx = _elements_by_id(elements)

    elem = idx[container.block_id]
    label = elem["data"]["label"]
    assert "2 ops" in label, (
        f"collapsed container with 2 inner ops should report '2 ops'; "
        f"got {label!r}"
    )


# ---------------------------------------------------------------------------
# Inner-block parent reference (spec §4.4 + §5.5)
# ---------------------------------------------------------------------------


def test_inner_blocks_set_parent_to_container_block_id() -> None:
    """Inner blocks in a container's nested scope set ``data.parent``.

    Spec §5.5 — when the renderer recurses into ``container.nested.blocks``,
    every emitted inner block carries ``data.parent =
    container.block_id`` so cytoscape's compound layout groups it inside
    the container's bounding box.  Outer-scope blocks parent to
    ``None``.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)

    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    nested = container.nested
    assert nested is not None, (
        "container_main_flow fixture must declare a nested scope"
    )
    # Each inner block (including the auto-seeded InputImage) should
    # surface in the canvas with parent = container.block_id.
    for inner in nested.blocks:
        assert inner.block_id in idx, (
            f"inner block {inner.block_id} ({inner.class_name}) is "
            "missing from the recursive render"
        )
        elem = idx[inner.block_id]
        assert elem["data"].get("parent") == container.block_id, (
            f"inner block {inner.block_id} ({inner.class_name}) should "
            f"carry data.parent={container.block_id}; got "
            f"{elem['data'].get('parent')!r}"
        )


def test_outer_scope_blocks_have_no_parent() -> None:
    """Root-scope blocks emit ``data.parent = None`` (top-level).

    Cytoscape compound parents only group elements whose ``data.parent``
    matches another element's id; root-scope blocks have no enclosing
    container, so the parent reference is ``None``.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)

    # All blocks living in state.root.blocks should have data.parent == None.
    for block in state.root.blocks:
        elem = idx[block.block_id]
        assert elem["data"].get("parent") is None, (
            f"root-scope block {block.block_id} ({block.class_name}) "
            f"should have data.parent=None; got "
            f"{elem['data'].get('parent')!r}"
        )


# ---------------------------------------------------------------------------
# Collapsed container — children hidden via display: none mechanism
# ---------------------------------------------------------------------------


def test_collapsed_container_children_marked_parent_collapsed() -> None:
    """Collapsed containers propagate ``parent_collapsed=True`` to children.

    Spec §4.4 — when a container is collapsed, its children must be
    visually hidden (cytoscape stylesheet's ``[?parent_collapsed]``
    selector maps to ``display: none``).  The renderer drops the
    ``parent_collapsed`` data flag on every descendant element so the
    selector resolves at canvas paint time without needing a re-emit
    on every collapse toggle.
    """

    inner_block_id = _new_block_id()
    inner_block = BlockNode(
        block_id=inner_block_id,
        class_name="GaussianBlur",
        params={},
    )
    nested = _DagBuilderScope(blocks=[inner_block], name="inner_pipe")
    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="parent_pipe",
        nested=nested,
        collapsed=True,
    )
    scope = _DagBuilderScope(blocks=[container])
    elements = build_canvas_elements_dag(scope)
    idx = _elements_by_id(elements)

    # The inner block must surface with parent_collapsed = True.
    assert inner_block_id in idx, (
        "collapsed container's children must still render so wires "
        "stay valid; the stylesheet hides them via the "
        "[?parent_collapsed] selector"
    )
    inner_elem = idx[inner_block_id]
    assert inner_elem["data"].get("parent_collapsed") is True, (
        "child of a collapsed container must carry "
        "data.parent_collapsed=True so the stylesheet's "
        "[?parent_collapsed] selector hides it"
    )
    # Per spec §4.4 — collapsed containers KEEP their own image-in /
    # image-out ports visible so wires from the outer scope aren't
    # orphaned.  The container's outer ports therefore parent to the
    # container's block_id but do NOT carry parent_collapsed.  Only the
    # nested-scope descendants (auto-seeded InputImage + inner blocks +
    # their ports) propagate the flag.
    inner_descendant_ids = {b.block_id for b in nested.blocks}
    for elem in elements:
        data = elem.get("data") or {}
        bid = data.get("block_id")
        # Skip elements whose block_id is the container itself (its outer
        # chrome + visible ports) or that don't live in the nested scope.
        if bid not in inner_descendant_ids:
            continue
        assert data.get("parent_collapsed") is True, (
            f"descendant {data.get('id')!r} of a collapsed container "
            f"must propagate data.parent_collapsed=True; got "
            f"{data.get('parent_collapsed')!r}"
        )


# ---------------------------------------------------------------------------
# Empty container placeholder (spec §4.8)
# ---------------------------------------------------------------------------


def test_empty_container_emits_drop_ops_here_placeholder() -> None:
    """An empty container surfaces a ``+ drop ops here`` placeholder.

    Spec §4.8 — when a container's nested scope holds only the
    auto-seeded ``InputImage`` (no real ops yet), the renderer emits a
    label-only sub-node parented to the container so the user sees a
    valid drop target hint.
    """

    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="fresh_pipeline",
        # Empty nested scope auto-seeds the InputImage on construction.
        nested=_DagBuilderScope(name="fresh_pipeline"),
    )
    scope = _DagBuilderScope(blocks=[container])
    elements = build_canvas_elements_dag(scope)

    placeholders = [
        e for e in elements
        if "dag-block__placeholder" in _classes_of(e)
    ]
    assert len(placeholders) == 1, (
        f"empty container should surface exactly one placeholder element; "
        f"got {len(placeholders)}"
    )
    placeholder = placeholders[0]
    assert placeholder["data"]["parent"] == container_id, (
        "placeholder must parent to the empty container so cytoscape "
        "renders the hint inside the container's bounding box"
    )
    assert placeholder["data"]["label"] == "+ drop ops here", (
        f"placeholder label should be '+ drop ops here'; got "
        f"{placeholder['data']['label']!r}"
    )
    # Non-selectable so a stray click on the hint doesn't fight the
    # container's title-bar selection handler.
    assert placeholder.get("selectable") is False


def test_populated_container_skips_placeholder() -> None:
    """A populated container does NOT emit the placeholder hint.

    Once the user has dropped at least one non-InputImage block into the
    container's nested scope, the ``+ drop ops here`` hint should
    disappear so it doesn't clutter the canvas.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    placeholders = [
        e for e in elements
        if "dag-block__placeholder" in _classes_of(e)
    ]
    # container_main_flow fixture has real ops inside the container.
    assert placeholders == [], (
        f"populated container should not emit a placeholder; got "
        f"{len(placeholders)} placeholder(s)"
    )


def test_empty_container_carries_container_empty_class() -> None:
    """Expanded containers with no real ops get the empty-container class.

    Spec §4.8 — the ``dag-block--container-empty`` class is added on
    the container's own element so the stylesheet can apply the
    dashed-outline / muted-fill hint chrome.  Only applies when
    expanded; collapsed containers always read as a compact 1-row
    block regardless of inner content.
    """

    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="empty_pipe",
        nested=_DagBuilderScope(name="empty_pipe"),
        collapsed=False,
    )
    scope = _DagBuilderScope(blocks=[container])
    elements = build_canvas_elements_dag(scope)
    idx = _elements_by_id(elements)
    classes = _classes_of(idx[container_id])
    assert "dag-block--container-empty" in classes, (
        f"empty expanded container should carry "
        f"dag-block--container-empty; got {classes!r}"
    )


def test_populated_container_does_not_carry_container_empty_class() -> None:
    """Once a container holds real ops, the empty-container class drops.

    Once the user has dropped at least one non-InputImage block into the
    nested scope, the dashed-outline empty-state chrome should disappear
    so it doesn't keep advertising the container as a drop target.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)
    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    classes = _classes_of(idx[container.block_id])
    assert "dag-block--container-empty" not in classes, (
        f"populated container should not carry dag-block--container-empty; "
        f"got {classes!r}"
    )


# ---------------------------------------------------------------------------
# Container inspector card (spec §4.5)
# ---------------------------------------------------------------------------


def test_container_inspector_card_renders_pipeline_name_and_summary() -> None:
    """The container card renders pipeline name + inner-scope summary.

    Spec §4.5 — selecting a container renders the dedicated container
    inspector card with the pipeline name, description, and an inner
    summary string (e.g. ``"3 ops, 1 aux pipeline"``).
    """

    inner = BlockNode(
        block_id=_new_block_id(),
        class_name="GaussianBlur",
        params={"sigma": 1.0},
    )
    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="preproc_pipe",
        nested=_DagBuilderScope(
            blocks=[inner],
            name="preproc",
            desc="Pre-processing chain",
        ),
    )
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container], name="root"),
        selected_block_id=container_id,
    )

    card = _build_container_inspector_card(state, container)

    # The pipeline name input carries the nested scope's name.
    name_inputs = _find_by_id(card, ids.INPUT_CONTAINER_NAME)
    assert len(name_inputs) == 1, (
        f"container inspector card should expose one "
        f"INPUT_CONTAINER_NAME input; found {len(name_inputs)}"
    )
    assert name_inputs[0].value == "preproc"

    # The description textarea carries the nested scope's desc.
    desc_inputs = _find_by_id(card, ids.INPUT_CONTAINER_DESC)
    assert len(desc_inputs) == 1
    assert desc_inputs[0].value == "Pre-processing chain"

    # The summary text mentions the inner block count.
    text = _collect_text(card)
    assert "Inner scope:" in text, (
        f"container inspector card should expose an 'Inner scope:' "
        f"summary; got text {text!r}"
    )


def test_container_inspector_card_exposes_drill_in_button() -> None:
    """The container card shows a ``Drill in →`` button (spec §4.5).

    The button id is :data:`BTN_DRILL_IN_CONTAINER`; Agent 5B owns the
    dispatcher that consumes this id and pushes the container onto the
    breadcrumb.
    """

    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="drillable",
        nested=_DagBuilderScope(name="drillable"),
    )
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container], name="root"),
        selected_block_id=container_id,
    )

    card = _build_container_inspector_card(state, container)
    drill_btns = _find_by_id(card, ids.BTN_DRILL_IN_CONTAINER)
    assert len(drill_btns) == 1, (
        f"container inspector card should expose exactly one Drill in → "
        f"button; got {len(drill_btns)}"
    )
    # The label text reads as the drill-in affordance.
    text = _collect_text(drill_btns[0])
    assert "Drill in" in text, (
        f"Drill in → button should render its label; got {text!r}"
    )


def test_container_inspector_card_does_not_render_nrows_ncols() -> None:
    """The container card suppresses ``nrows`` / ``ncols`` fields.

    Spec §4.5 — only the root scope exposes grid presets.  Container
    scopes inherit the runtime grid from their parent scope and must
    NOT expose duplicate inputs that the user might think apply to the
    inner pipeline.
    """

    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="no_grid",
        nested=_DagBuilderScope(name="no_grid"),
    )
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container], name="root"),
        selected_block_id=container_id,
    )

    card = _build_container_inspector_card(state, container)
    nrows_inputs = _find_by_id(card, ids.INPUT_NROWS)
    ncols_inputs = _find_by_id(card, ids.INPUT_NCOLS)
    assert nrows_inputs == [], (
        "container inspector card must not render the nrows input "
        "(spec §4.5 — root scope only); got nrows input(s)"
    )
    assert ncols_inputs == [], (
        "container inspector card must not render the ncols input "
        "(spec §4.5 — root scope only); got ncols input(s)"
    )


def test_dag_inspector_routes_container_selection_to_container_card() -> None:
    """The DAG inspector dispatch lands on the container card for containers.

    When ``state.selected_block_id`` points at a container block,
    :func:`_build_dag_inspector` short-circuits to
    :func:`_build_container_inspector_card`.  The card carries the
    container's pipeline-name input + drill-in button so the user can
    edit container metadata without leaving the inspector pane.
    """

    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="routed",
        nested=_DagBuilderScope(name="routed"),
    )
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container], name="root"),
        selected_block_id=container_id,
    )

    # Registry is not consulted for container selection (no param form).
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]
    name_inputs = _find_by_id(inspector, ids.INPUT_CONTAINER_NAME)
    drill_btns = _find_by_id(inspector, ids.BTN_DRILL_IN_CONTAINER)
    assert len(name_inputs) == 1
    assert len(drill_btns) == 1


# ---------------------------------------------------------------------------
# Consumer-fed dot (spec §4.1, §4.4)
# ---------------------------------------------------------------------------


def test_container_inner_input_image_renders_as_consumer_fed_dot() -> None:
    """The container scope's InputImage surfaces as a small purple dot.

    Spec §4.1 + §4.4 — every container scope auto-seeds an
    ``InputImage`` sentinel but renders it as a small consumer-fed dot
    (NOT the big green chevron used at the root).  The dot carries the
    ``dag-block__consumer-fed-dot`` class so the cytoscape stylesheet
    paints a 12×12 purple circle on the container's inner-left edge.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)

    dots = [
        e for e in elements
        if "dag-block__consumer-fed-dot" in _classes_of(e)
    ]
    assert dots, (
        "container_main_flow fixture should expose at least one "
        "consumer-fed dot (the inner InputImage sentinel)"
    )
    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    for dot in dots:
        assert dot["data"]["parent"] == container.block_id, (
            "consumer-fed dot must parent to the enclosing container "
            "so cytoscape draws it inside the container's bounding box"
        )


def test_root_scope_input_image_is_not_a_consumer_fed_dot() -> None:
    """The root-scope InputImage keeps the big chevron treatment.

    Only nested-scope ``InputImage`` sentinels render as a consumer-fed
    dot; the root scope's InputImage stays a regular block so the user
    can click it to inspect/re-anchor the chain source.
    """

    state = _load_state("linear_chain")
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)
    input_block = next(
        b for b in state.root.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    )
    elem = idx[input_block.block_id]
    classes = _classes_of(elem)
    assert "dag-block__consumer-fed-dot" not in classes, (
        "root-scope InputImage must NOT render as a consumer-fed dot; "
        "the dot treatment is for nested-scope InputImage sentinels only"
    )


# ---------------------------------------------------------------------------
# Inner issues surface on the container (spec §4.4)
# ---------------------------------------------------------------------------


def test_container_data_carries_inner_issue_counts() -> None:
    """Container elements expose ``inner_error_count`` / ``inner_hint_count``.

    Spec §4.4 — a collapsed container's outer chrome aggregates the
    inner scope's issue count so users see the badge without expanding
    the container.  The renderer attaches ``inner_error_count`` +
    ``inner_hint_count`` to every container's ``data`` dict so any
    downstream selector (badge, inspector card, scroll-to-issue
    callback) can read it without re-walking the nested scope.
    """

    from phenotypic.gui.builder._validation import Issue

    inner_block_id = _new_block_id()
    inner_block = BlockNode(
        block_id=inner_block_id,
        class_name="GaussianBlur",
        params={},
    )
    nested = _DagBuilderScope(blocks=[inner_block], name="inner")
    container_id = _new_block_id()
    container = BlockNode(
        block_id=container_id,
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="parent",
        nested=nested,
    )
    scope = _DagBuilderScope(blocks=[container])

    issues = [
        Issue(
            kind="fork",
            block_id=inner_block_id,
            detail="test inner issue",
            scope_path=[container_id],
            severity="error",
        )
    ]
    elements = build_canvas_elements_dag(scope, issues=issues)
    idx = _elements_by_id(elements)

    elem = idx[container_id]
    assert elem["data"].get("inner_error_count") == 1, (
        f"container should expose inner_error_count=1; got "
        f"{elem['data'].get('inner_error_count')!r}"
    )
    assert elem["data"].get("inner_hint_count") == 0

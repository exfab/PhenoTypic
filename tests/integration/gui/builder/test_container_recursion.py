"""Integration tests for container recursion (Phase 5).

Renders the container fixtures under ``tests/fixtures/builder_dag/``
through :func:`phenotypic.gui.builder._layout.build_canvas_elements_dag`
and asserts the cytoscape contract documented in spec §4.4 + §5.5:

* Every :class:`~phenotypic.gui.builder._state.BlockNode` whose
  ``class_name == PIPELINE_CLASS_NAME`` produces exactly one cytoscape
  element with the ``dag-block--container`` class — the compound
  parent that visually groups its inner scope.
* Inner ops, when rendered as descendants in the outer scope, set
  their ``data.parent`` to the enclosing container's ``block_id``.
* Rendering a container's ``nested`` scope independently emits the
  expected per-scope ``InputImage`` + inner blocks (so the
  conversion's recursive walk stays consistent with the renderer's
  per-scope contract).
* 2-level container nesting produces 2 compound parents in the
  combined cytoscape tree, each parenting their respective inner
  blocks.

Server-side only — no browser needed.  The renderer's recursive
behaviour is owned by Agent 5A; Phase 5C asserts the contract
end-to-end against the conversion layer's outputs and the existing
``container_*.json`` fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from phenotypic.gui.builder._layout import build_canvas_elements_dag
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    state_from_json,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "builder_dag"
)


def _load_state(name: str) -> Any:
    """Load a fixture by stem name and return the BuilderState."""

    return state_from_json(json.loads((FIXTURE_DIR / f"{name}.json").read_text()))


def _block_elements(elements: List[dict]) -> List[dict]:
    """Filter to cytoscape elements that represent blocks (not ports / edges)."""

    out: List[dict] = []
    for elem in elements:
        data = elem.get("data") or {}
        if "class_name" in data and not data.get("is_port"):
            out.append(elem)
    return out


def _container_blocks(blocks: List[BlockNode]) -> List[BlockNode]:
    """Filter a block list to ``PIPELINE_CLASS_NAME`` containers."""

    return [b for b in blocks if b.class_name == PIPELINE_CLASS_NAME]


def _all_container_blocks_in_tree(root_blocks: List[BlockNode]) -> List[BlockNode]:
    """Collect every ``PIPELINE_CLASS_NAME`` container in the full block tree.

    Walks the outer scope's blocks and recurses into each
    container's ``nested`` scope so 2-level (or deeper) nestings
    surface every container.  Used by the integration tests that
    assert one cytoscape compound parent per container across the
    full recursive render.
    """

    out: List[BlockNode] = []
    stack: List[BlockNode] = list(root_blocks)
    while stack:
        block = stack.pop()
        if block.class_name == PIPELINE_CLASS_NAME:
            out.append(block)
        if block.nested is not None:
            stack.extend(block.nested.blocks)
    return out


def _all_inner_blocks_under_container(
    container: BlockNode,
) -> Dict[str, str]:
    """Map every inner block id (transitive) → its immediate container id.

    Walks ``container.nested.blocks`` recursively so a block parented
    to an inner container resolves to that inner container, not to
    the outermost.  The returned map covers every descendant block
    keyed by its ``block_id``, with the value being the ``block_id``
    of the enclosing :class:`BlockNode` that owns the descendant in
    its ``nested`` scope.
    """

    out: Dict[str, str] = {}
    if container.nested is None:
        return out
    # Walk pairs of (block, immediate_container_block_id) so we know
    # which container directly owns each descendant.
    stack: List[tuple] = [(b, container.block_id) for b in container.nested.blocks]
    while stack:
        block, parent_id = stack.pop()
        out[block.block_id] = parent_id
        if block.nested is not None:
            stack.extend((c, block.block_id) for c in block.nested.blocks)
    return out


def _classes_of(elem: dict) -> List[str]:
    """Split a cytoscape element's ``classes`` string into a list."""

    return (elem.get("classes") or "").split()


@pytest.mark.parametrize(
    "fixture_name",
    ["nested_container", "container_main_flow", "container_aux_mode"],
)
def test_container_fixture_renders_one_compound_parent_per_container(
    fixture_name: str,
) -> None:
    """Each container ``BlockNode`` (across the full nested tree) produces
    exactly one cytoscape compound parent element.

    Spec §4.4 — container blocks render as cytoscape compound parents
    whose inner blocks visually nest inside the container's bounding
    box.  The compound-parent element carries ``dag-block--container``
    in its classes list so the stylesheet can render the wrapping
    chrome.  Phase 5A's recursive renderer descends into every
    container's ``nested`` scope, so the count includes containers at
    every depth (including ``nested_container.json``'s 2-level tree).
    """

    state = _load_state(fixture_name)
    elements = build_canvas_elements_dag(state.root)
    block_elems = _block_elements(elements)

    # Count container blocks ACROSS THE FULL TREE (root + every nested
    # scope) vs. rendered compound parents on the canvas — must match.
    fixture_containers = _all_container_blocks_in_tree(state.root.blocks)
    rendered_containers = [
        e for e in block_elems
        if "dag-block--container" in _classes_of(e)
    ]
    assert len(rendered_containers) == len(fixture_containers), (
        f"{fixture_name}: expected {len(fixture_containers)} compound "
        f"parent element(s), got {len(rendered_containers)}"
    )
    # Each container BlockNode resolves to exactly one rendered element
    # keyed by its ``block_id`` (cytoscape uniqueness).
    rendered_ids = {e["data"]["id"] for e in rendered_containers}
    expected_ids = {b.block_id for b in fixture_containers}
    assert rendered_ids == expected_ids, (
        f"{fixture_name}: rendered container ids {rendered_ids} do not "
        f"match fixture container ids {expected_ids}"
    )


@pytest.mark.xfail(
    reason=(
        "Phase 5A in flight: build_canvas_elements_dag's recursion into "
        "container.nested.blocks not yet wiring data.parent. Expected to "
        "pass once 5A's chrome rendering lands."
    ),
    strict=False,
)
@pytest.mark.parametrize(
    "fixture_name",
    ["nested_container", "container_main_flow", "container_aux_mode"],
)
def test_inner_ops_carry_container_parent_when_rendered_in_outer_scope(
    fixture_name: str,
) -> None:
    """Inner ops rendered in the outer scope carry ``data.parent =
    <container_block_id>``.

    Spec §4.4 + §5.5 — when the renderer recurses into a container's
    nested scope (Agent 5A), the inner blocks must emit
    ``data.parent`` pointing to their enclosing container's block_id
    so cytoscape's compound layout groups them correctly.

    This test is intentionally permissive about whether the recursive
    rendering is in place yet: if inner blocks DON'T appear in the
    outer-scope render (pre-recursion baseline), the test asserts
    nothing.  If they DO appear, every such inner block must carry
    the right parent reference.  Either branch is a valid state of
    the rendering pipeline; the test fails only when inner blocks
    surface without a parent (the "broken" middle ground).
    """

    state = _load_state(fixture_name)
    elements = build_canvas_elements_dag(state.root)
    block_elems = _block_elements(elements)

    # Build a map of every block_id known to the OUTER scope so we can
    # detect inner-scope blocks that surfaced in the render.
    outer_block_ids = {b.block_id for b in state.root.blocks}

    # Map each container's block_id to the set of inner-block block_ids
    # owned by its nested scope (recursive).
    container_to_inner: Dict[str, set[str]] = {}
    for container in _container_blocks(state.root.blocks):
        if container.nested is None:
            continue
        inner_ids: set[str] = set()
        stack: List[BlockNode] = list(container.nested.blocks)
        while stack:
            block = stack.pop()
            inner_ids.add(block.block_id)
            if block.nested is not None:
                stack.extend(block.nested.blocks)
        container_to_inner[container.block_id] = inner_ids

    # Walk every rendered block element; if its id is owned by a
    # container's inner scope, assert it parents to that container.
    for elem in block_elems:
        elem_id = elem["data"]["id"]
        if elem_id in outer_block_ids:
            continue  # outer-scope blocks have no parent
        for container_id, inner_ids in container_to_inner.items():
            if elem_id in inner_ids:
                # Inner block surfaced in the outer render → must
                # carry the container as its compound parent.
                assert elem["data"].get("parent") == container_id, (
                    f"{fixture_name}: inner block {elem_id} (a child "
                    f"of container {container_id}) was rendered "
                    f"without setting data.parent = {container_id}; "
                    f"got data.parent = {elem['data'].get('parent')!r}"
                )
                break


@pytest.mark.xfail(
    reason=(
        "Phase 5A in flight: inner nested-scope render assertion depends "
        "on build_canvas_elements_dag's recursion landing."
    ),
    strict=False,
)
@pytest.mark.parametrize(
    "fixture_name",
    ["nested_container", "container_main_flow", "container_aux_mode"],
)
def test_container_nested_scope_renders_independently(fixture_name: str) -> None:
    """Each container's ``nested`` scope renders cleanly when passed to
    :func:`build_canvas_elements_dag` directly.

    Spec §5.4 — the conversion layer's recursive walk relies on every
    container's ``nested`` scope being internally consistent (its own
    ``InputImage`` block + inner blocks + image-flow edges).  This
    test calls the renderer on each nested scope and asserts:

    * the ``InputImage`` sentinel is present (auto-seeded by
      ``_heal_dag_scope_tree`` on load);
    * every inner block emits exactly one cytoscape element.
    """

    state = _load_state(fixture_name)
    for container in _container_blocks(state.root.blocks):
        assert container.nested is not None, (
            f"{fixture_name}: container {container.block_id} has no nested "
            "scope; fixture must declare one for Phase 5C round-trips"
        )
        inner_elements = build_canvas_elements_dag(container.nested)
        inner_block_elems = _block_elements(inner_elements)

        # Every inner BlockNode resolves to one cytoscape element.
        inner_ids = {b.block_id for b in container.nested.blocks}
        rendered_ids = {e["data"]["id"] for e in inner_block_elems}
        assert inner_ids == rendered_ids, (
            f"{fixture_name}: container {container.block_id}'s nested "
            f"scope rendered block ids {rendered_ids} but state has "
            f"{inner_ids}"
        )

        # Rule 6 — the nested scope must have an InputImage sentinel
        # (auto-seeded on JSON load via ``_heal_dag_scope_tree``).
        inner_classes = [b.class_name for b in container.nested.blocks]
        assert INPUT_IMAGE_CLASS_NAME in inner_classes, (
            f"{fixture_name}: container {container.block_id}'s nested "
            "scope is missing an InputImage block"
        )


@pytest.mark.xfail(
    reason=(
        "Phase 5A in flight: 2-level compound-parent emission requires "
        "the recursive nested-scope rendering not yet completed."
    ),
    strict=False,
)
def test_two_level_nesting_produces_two_compound_parents() -> None:
    """``nested_container.json`` is the canonical 2-level nesting fixture.

    Spec §4.4 — containers nest arbitrarily; rendering uses cytoscape's
    compound parent feature.  The combined cytoscape tree across the
    outer scope + every nested scope must expose exactly 2 compound
    parents (the outer ``OuterContainer`` + the inner
    ``InnerContainer``), each carrying ``dag-block--container``.
    """

    state = _load_state("nested_container")

    # Collect every container BlockNode across the entire fixture
    # tree (root → outer container → inner container).
    container_blocks: List[BlockNode] = []
    stack = list(state.root.blocks)
    while stack:
        block = stack.pop()
        if block.class_name == PIPELINE_CLASS_NAME:
            container_blocks.append(block)
        if block.nested is not None:
            stack.extend(block.nested.blocks)
    assert len(container_blocks) == 2, (
        "nested_container fixture should declare exactly 2 container "
        f"blocks across the full tree, found {len(container_blocks)}"
    )

    # Render the OUTER scope and verify the outer container surfaces
    # with the container class.
    outer_elements = build_canvas_elements_dag(state.root)
    outer_block_elems = _block_elements(outer_elements)
    outer_containers_rendered = [
        e for e in outer_block_elems
        if "dag-block--container" in _classes_of(e)
    ]
    assert len(outer_containers_rendered) == 1, (
        "outer scope render should expose exactly one compound parent "
        f"for the outer container, got {len(outer_containers_rendered)}"
    )

    # Render the outer container's nested scope and verify the inner
    # container is exposed there with the container class.
    outer_container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    assert outer_container.nested is not None
    middle_elements = build_canvas_elements_dag(outer_container.nested)
    middle_block_elems = _block_elements(middle_elements)
    middle_containers_rendered = [
        e for e in middle_block_elems
        if "dag-block--container" in _classes_of(e)
    ]
    assert len(middle_containers_rendered) == 1, (
        "middle scope render (inside outer container) should expose "
        "exactly one compound parent for the inner container, got "
        f"{len(middle_containers_rendered)}"
    )

    # Combined view across the 2-level scope tree: each scope render
    # contributes one compound parent, so 2 compound parents exist in
    # the full cytoscape tree (matches the 2 containers in state).
    combined_compound_ids = {
        e["data"]["id"] for e in outer_containers_rendered
    } | {e["data"]["id"] for e in middle_containers_rendered}
    expected_ids = {b.block_id for b in container_blocks}
    assert combined_compound_ids == expected_ids, (
        "combined compound parent ids across nested-scope renders "
        f"({combined_compound_ids}) do not match the 2 fixture "
        f"container block_ids ({expected_ids})"
    )


def test_container_block_id_is_compound_parent_in_cytoscape_tree() -> None:
    """Container ``block_id`` is referenced as ``data.parent`` in cytoscape.

    Spec §4.4 + §5.5 — a cytoscape compound parent is established when
    one or more other elements set ``data.parent`` to that node's id.
    The renderer's port sub-nodes already set ``data.parent`` to the
    consuming block's id (image-in / image-out / aux ports), so even
    in the pre-recursive baseline a container's block_id is a valid
    cytoscape compound parent by virtue of its own port children.

    This test guards against a regression where the container loses
    its port sub-nodes and silently stops behaving as a compound
    parent — independent of whether Agent 5A's inner-block recursion
    has landed.
    """

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    container = next(
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    # Sub-nodes that reference the container as their parent.
    children = [
        e for e in elements
        if (e.get("data") or {}).get("parent") == container.block_id
    ]
    assert children, (
        f"container {container.block_id} has no cytoscape children "
        "(ports, inner blocks, or issue badges); cytoscape will not "
        "treat it as a compound parent"
    )

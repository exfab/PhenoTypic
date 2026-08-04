"""Integration tests for ``build_canvas_elements_dag``.

Renders every fixture under ``tests/fixtures/builder_dag/`` through
:func:`phenotypic.gui.builder._layout.build_canvas_elements_dag` and
asserts the resulting cytoscape ``elements`` list shape.  Server-side
only — no browser needed.

The tests check that:

* Every :class:`~phenotypic.gui.builder._state.BlockNode` produces one
  cytoscape node element keyed by ``BlockNode.block_id``.
* Every :class:`~phenotypic.gui.builder._state.Edge` produces one
  cytoscape edge element keyed by the ``edge__<edge_id>`` prefix.
* Aux-eligible parameters render port sub-nodes with the
  ``dag-port--aux`` class and a ``data.accepts`` list of compatible
  source class names.
* Container blocks emit the ``dag-block--container`` class.
* The ``InputImage`` block is *not* rendered with an image-in port
  (only an output port).
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
    state_from_json,
)


FIXTURE_DIR = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "builder_dag"

# Some fixtures intentionally encode states that wouldn't survive a
# round-trip through the dispatcher (cycles, forks, duplicate inputs).
# The renderer must still produce a non-empty element list for these so
# the user can see what's broken — the validator surfaces issues on top.
#
# ``legacy_popover_pipeline.json`` is excluded because it's a legacy
# ImagePipeline.to_json() payload (not a DAG scope) — covered separately
# by ``test_legacy_pipeline_json.py``.
_EXCLUDED_FIXTURES = {"legacy_popover_pipeline"}
_ALL_FIXTURES = sorted(
    p.stem for p in FIXTURE_DIR.glob("*.json")
    if not p.stem.endswith(".expected_issues")
    and p.stem not in _EXCLUDED_FIXTURES
)


def _load_state(name: str) -> Any:
    """Load a fixture by stem name and return the BuilderState."""

    path = FIXTURE_DIR / f"{name}.json"
    return state_from_json(json.loads(path.read_text(encoding="utf-8")))


def _elements_by_id(elements: List[dict]) -> Dict[str, dict]:
    """Index the cytoscape elements by ``data.id`` for assertion lookups."""

    out: Dict[str, dict] = {}
    for elem in elements:
        data = elem.get("data") or {}
        eid = data.get("id")
        if eid is not None:
            out[eid] = elem
    return out


@pytest.mark.parametrize("fixture_name", _ALL_FIXTURES)
def test_every_fixture_renders_non_empty(fixture_name: str) -> None:
    """Every fixture under ``builder_dag/`` produces at least one element."""

    state = _load_state(fixture_name)
    elements = build_canvas_elements_dag(state.root)
    assert isinstance(elements, list)
    assert len(elements) > 0, f"{fixture_name} rendered empty element list"


@pytest.mark.parametrize("fixture_name", _ALL_FIXTURES)
def test_block_elements_match_state_blocks(fixture_name: str) -> None:
    """Every :class:`BlockNode` becomes one cytoscape node with the same id."""

    state = _load_state(fixture_name)
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)
    for block in state.root.blocks:
        assert block.block_id in idx, (
            f"Block {block.block_id} ({block.class_name}) is missing from "
            f"the rendered elements for {fixture_name}"
        )
        elem = idx[block.block_id]
        classes = (elem.get("classes") or "").split()
        assert "dag-block" in classes
        assert elem["data"]["class_name"] == block.class_name
        assert elem["data"]["block_id"] == block.block_id


@pytest.mark.parametrize("fixture_name", _ALL_FIXTURES)
def test_edge_elements_match_state_edges(fixture_name: str) -> None:
    """Every :class:`Edge` becomes one cytoscape edge with the prefixed id."""

    state = _load_state(fixture_name)
    elements = build_canvas_elements_dag(state.root)
    idx = _elements_by_id(elements)
    for edge in state.root.edges:
        expected_id = f"edge__{edge.edge_id}"
        assert expected_id in idx, (
            f"Edge {edge.edge_id} is missing from the rendered elements "
            f"for {fixture_name}"
        )
        elem = idx[expected_id]
        assert elem["data"]["source"] == edge.source_block_id
        assert elem["data"]["target"] == edge.target_block_id
        assert elem["data"]["kind"] == edge.kind
        classes = (elem.get("classes") or "").split()
        assert "dag-wire" in classes


def test_input_image_block_has_no_input_port() -> None:
    """The InputImage sentinel renders only the image-out port (no input)."""

    state = _load_state("linear_chain")
    elements = build_canvas_elements_dag(state.root)
    input_block = next(
        b for b in state.root.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    )
    idx = _elements_by_id(elements)
    # Output port must exist
    out_id = f"port__{input_block.block_id}__out"
    assert out_id in idx
    # Input port must NOT exist (per spec §4.1, InputImage has no input)
    in_id = f"port__{input_block.block_id}__in"
    assert in_id not in idx, (
        "InputImage block should not render an image-in port "
        "(no upstream blocks per spec §4.1)"
    )


def test_aux_ports_carry_accepts_list() -> None:
    """Aux ports emit a ``data.accepts`` list of compatible class names."""

    state = _load_state("scalar_aux")
    elements = build_canvas_elements_dag(state.root)
    # Find the consumer block (anything that's not InputImage)
    consumer = next(
        b for b in state.root.blocks
        if b.class_name not in (INPUT_IMAGE_CLASS_NAME, "BlurGauss")
        and b.class_name != PIPELINE_CLASS_NAME
    )
    # Find an aux port on this consumer
    aux_ports = [
        e for e in elements
        if (e.get("data") or {}).get("parent") == consumer.block_id
        and (e.get("data") or {}).get("port_kind") == "aux"
    ]
    assert aux_ports, (
        f"scalar_aux fixture should expose at least one aux port on "
        f"{consumer.class_name}"
    )
    for port_elem in aux_ports:
        assert "accepts" in port_elem["data"]
        assert isinstance(port_elem["data"]["accepts"], list)


def test_container_block_carries_container_class() -> None:
    """Pipeline-class blocks render with ``dag-block--container`` class."""

    state = _load_state("container_main_flow")
    elements = build_canvas_elements_dag(state.root)
    container_blocks = [
        b for b in state.root.blocks if b.class_name == PIPELINE_CLASS_NAME
    ]
    assert container_blocks, "container_main_flow fixture should have a container"
    idx = _elements_by_id(elements)
    for block in container_blocks:
        elem = idx[block.block_id]
        classes = (elem.get("classes") or "").split()
        assert "dag-block--container" in classes


def test_main_path_edges_carry_is_main_flag() -> None:
    """Edges on the path from InputImage to terminal carry ``data.is_main``."""

    state = _load_state("linear_chain")
    elements = build_canvas_elements_dag(state.root)
    edge_elems = [e for e in elements
                  if (e.get("classes") or "").split()
                  and "dag-wire" in (e.get("classes") or "").split()]
    assert edge_elems, "linear_chain should have at least one wire"
    # Every wire in linear_chain is on the main path
    for elem in edge_elems:
        if elem["data"]["kind"] == "image":
            assert elem["data"]["is_main"] is True, (
                f"linear_chain image edge should be on main path: "
                f"{elem['data']}"
            )


def test_selected_block_id_adds_selected_class() -> None:
    """Passing ``selected_block_id`` adds the ``selected`` class to that block."""

    state = _load_state("linear_chain")
    target_block = state.root.blocks[1]  # The first non-InputImage block
    elements = build_canvas_elements_dag(
        state.root, selected_block_id=target_block.block_id
    )
    idx = _elements_by_id(elements)
    classes = (idx[target_block.block_id].get("classes") or "").split()
    assert "selected" in classes


def test_selected_edge_id_adds_selected_class() -> None:
    """Passing ``selected_edge_id`` adds the ``selected`` class to that edge."""

    state = _load_state("linear_chain")
    target_edge = state.root.edges[0]
    elements = build_canvas_elements_dag(
        state.root, selected_edge_id=target_edge.edge_id
    )
    idx = _elements_by_id(elements)
    classes = (idx[f"edge__{target_edge.edge_id}"].get("classes") or "").split()
    assert "selected" in classes


def test_issues_decorate_offending_blocks() -> None:
    """Issues with ``block_id`` produce border classes + an issue-badge sub-node."""

    from phenotypic.gui.builder._validation import Issue

    state = _load_state("linear_chain")
    offender = state.root.blocks[1]
    issues = [
        Issue(
            kind="fork",
            block_id=offender.block_id,
            detail="test issue",
            scope_path=[],
            severity="error",
        )
    ]
    elements = build_canvas_elements_dag(state.root, issues=issues)
    idx = _elements_by_id(elements)
    # Block should have an error-decoration class
    classes = (idx[offender.block_id].get("classes") or "").split()
    assert "dag-block--error" in classes
    # An issue badge sub-node should be emitted
    badge_id = f"issue__{offender.block_id}"
    assert badge_id in idx
    badge_classes = (idx[badge_id].get("classes") or "").split()
    assert "dag-issue" in badge_classes
    assert idx[badge_id]["data"]["rule_kind"] == "fork"

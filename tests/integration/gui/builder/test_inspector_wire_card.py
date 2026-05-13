"""Integration tests for the inspector wire card (spec §4.5).

Exercises the new DAG branch of
:func:`phenotypic.gui.builder._layout.build_inspector` for the
"wire-selected" state.  When ``state.selected_edge_id`` resolves to an
:class:`~phenotypic.gui.builder._state.Edge` in the active scope, the
inspector renders a wire card carrying:

* Source-block label → ``target_block.port`` text.
* Edge kind badge ("image flow" vs "aux assignment").
* A ``Disconnect`` button whose pattern-matching id encodes the
  ``Edge.edge_id`` for dispatcher routing.

The tests walk the rendered :class:`dash.html.Div` tree to assert the
above invariants without booting Dash — keeping the suite fast and
independent of clientside JS.
"""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import (
    _build_dag_inspector,
    _build_wire_card,
)
from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_from_json,
)

# Component-tree walking helpers shared with ``test_inspector_aux_section.py``;
# see ``conftest.py`` in this directory.
from .conftest import _collect_text, _find_by_id


FIXTURE_DIR = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "builder_dag"
)


# ---------------------------------------------------------------------------
# Fixture-driven smoke tests
# ---------------------------------------------------------------------------


def test_wire_card_renders_when_image_edge_selected() -> None:
    """A selected image-flow edge renders the wire card with the right labels."""

    state = state_from_json(
        json.loads((FIXTURE_DIR / "linear_chain.json").read_text())
    )
    # Pick the first image-flow edge in the root scope.
    edge = next(e for e in state.root.edges if e.kind == "image")
    state.selected_edge_id = edge.edge_id
    state.selected_block_id = None

    # No registry needed for wire-card rendering (it inspects the edge
    # + block labels, not param metadata).
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    # The wire card must be in the tree, identified by INSPECTOR_WIRE_CARD.
    cards = _find_by_id(inspector, ids.INSPECTOR_WIRE_CARD)
    assert len(cards) == 1, (
        f"Expected one wire card; found {len(cards)}: "
        f"{[getattr(c, 'id', None) for c in cards]}"
    )

    # Disconnect button keyed by edge_id should be present.
    disconnect_btns = _find_by_id(
        inspector, ids.inspector_disconnect_id(edge.edge_id)
    )
    assert len(disconnect_btns) == 1


def test_wire_card_summary_uses_block_labels() -> None:
    """The wire card's summary line shows source/target labels and the port."""

    # Build a small state where we control the labels.
    src = BlockNode(
        block_id=_new_block_id(),
        class_name="GaussianBlur",
        params={},
        label="UpstreamBlur",
    )
    tgt = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="DownstreamDetector",
    )
    edge = Edge(
        edge_id="abc",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="in",
        kind="image",
    )
    scope = _DagBuilderScope(blocks=[src, tgt], edges=[edge])

    card = _build_wire_card(scope, edge)
    text = _collect_text(card)
    assert "UpstreamBlur" in text
    assert "DownstreamDetector" in text
    assert ".in" in text  # target port label
    assert "image flow" in text


def test_wire_card_aux_edge_renders_param_name_and_slot() -> None:
    """An aux edge with a target slot renders ``param[idx]`` in the summary."""

    src = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    tgt = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
        list_slot_counts={"detectors": 2},
    )
    edge = Edge(
        edge_id="aux1",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="detectors",
        target_slot=1,
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[src, tgt], edges=[edge])

    card = _build_wire_card(scope, edge)
    text = _collect_text(card)
    assert "detectors[1]" in text
    assert "aux assignment" in text


def test_wire_card_disconnect_button_carries_edge_id() -> None:
    """Disconnect button's pattern-match id encodes the wire's edge_id."""

    src = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    tgt = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    edge = Edge(
        edge_id="my-edge-id",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="in",
        kind="image",
    )
    scope = _DagBuilderScope(blocks=[src, tgt], edges=[edge])

    card = _build_wire_card(scope, edge)
    btns = _find_by_id(card, ids.inspector_disconnect_id("my-edge-id"))
    assert len(btns) == 1
    assert btns[0].n_clicks == 0


def test_wire_card_replaces_block_when_edge_selected_overrides_block() -> None:
    """Selecting both an edge AND a block prefers the edge (defense in depth)."""

    src = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    tgt = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    edge = Edge(
        edge_id="wire-1",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="in",
        kind="image",
    )
    scope = _DagBuilderScope(blocks=[src, tgt], edges=[edge])
    state = _DagBuilderState(root=scope)
    state.selected_edge_id = "wire-1"
    state.selected_block_id = tgt.block_id  # Both set; wire card should win.

    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]
    # Wire card should be present even though a block is also selected.
    assert _find_by_id(inspector, ids.INSPECTOR_WIRE_CARD)


def test_wire_card_falls_back_to_empty_when_edge_missing() -> None:
    """Stale ``selected_edge_id`` collapses to the empty-state placeholder."""

    scope = _DagBuilderScope()
    state = _DagBuilderState(root=scope)
    state.selected_edge_id = "ghost-edge-that-does-not-exist"

    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]
    # No wire card should render (the lookup misses).
    assert not _find_by_id(inspector, ids.INSPECTOR_WIRE_CARD)


def test_wire_card_id_present_for_every_kind() -> None:
    """Both image- and aux-flow edges produce a wire card with the same id."""

    src = BlockNode(
        block_id=_new_block_id(), class_name="OtsuDetector", params={}
    )
    tgt = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
    )
    image_edge = Edge(
        edge_id="e1",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="in",
        kind="image",
    )
    aux_edge = Edge(
        edge_id="e2",
        source_block_id=src.block_id,
        target_block_id=tgt.block_id,
        target_port="detectors",
        target_slot=0,
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[src, tgt], edges=[image_edge, aux_edge])
    for edge in (image_edge, aux_edge):
        card = _build_wire_card(scope, edge)
        assert getattr(card, "id", None) == ids.INSPECTOR_WIRE_CARD

"""Integration tests for the inspector aux ports section (Phase 4, spec §4.5).

Exercises the new DAG branch of
:func:`phenotypic.gui.builder._layout.build_inspector` for the
"block-selected" state.  Each op-typed parameter on the selected block
renders as either a scalar row (single source label + ``Disconnect``)
or a list row (drag handle + arrow reorder + per-row remove + ``+ Add
empty slot``).

The tests use a fake :class:`OperationRegistry` so assertions don't
depend on the live phenotypic registry; only the structure of the
emitted Dash component tree is checked, not the actual op behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


from phenotypic.gui._operation_registry import OperationInfo, ParamInfo
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import (
    _build_aux_ports_section,
    _build_dag_inspector,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
)

# Component-tree walking helpers shared with ``test_inspector_wire_card.py``;
# see ``conftest.py`` in this directory.
from .conftest import _collect_text, _find_by_id, _find_by_type_key


@dataclass
class _FakeRegistry:
    """Minimal registry stub for the aux-section renderer."""

    ops: dict = field(default_factory=dict)

    def get(self, name: str) -> Optional[OperationInfo]:
        return self.ops.get(name)


def _make_param(
    name: str,
    *,
    is_operation: bool = True,
    is_pipeline: bool = False,
    is_list: bool = False,
    has_default: bool = False,
) -> ParamInfo:
    return ParamInfo(
        name=name,
        type_hint=Any,
        default=None,
        has_default=has_default,
        is_operation=is_operation,
        is_pipeline=is_pipeline,
        is_optional=False,
        is_list=is_list,
    )


def _make_op_info(name: str, params: dict) -> OperationInfo:
    class _StubCls:
        pass

    _StubCls.__name__ = name
    return OperationInfo(
        cls=_StubCls,
        name=name,
        category="Detector",
        module="tests.fake",
        docstring="",
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_aux_section_skipped_for_input_image() -> None:
    """The ``InputImage`` sentinel has no aux-typed params → no section."""

    block = BlockNode(
        block_id=_new_block_id(),
        class_name=INPUT_IMAGE_CLASS_NAME,
        params={},
    )
    scope = _DagBuilderScope(blocks=[block])
    registry = _FakeRegistry()
    result = _build_aux_ports_section(
        block=block, scope=scope, registry=registry
    )
    assert result is None


def test_aux_section_skipped_when_no_aux_params() -> None:
    """An op with no op-typed parameters → no section."""

    block = BlockNode(
        block_id=_new_block_id(),
        class_name="ScalarOnly",
        params={"sigma": 1.0},
    )
    scope = _DagBuilderScope(blocks=[block])
    registry = _FakeRegistry(
        ops={
            "ScalarOnly": _make_op_info(
                "ScalarOnly",
                {"sigma": _make_param("sigma", is_operation=False)},
            )
        }
    )
    result = _build_aux_ports_section(
        block=block, scope=scope, registry=registry
    )
    assert result is None


def test_scalar_aux_row_renders_empty_state() -> None:
    """An unwired scalar aux port renders the "Empty" placeholder."""

    block = BlockNode(
        block_id=_new_block_id(),
        class_name="ConsumerOp",
        params={},
    )
    scope = _DagBuilderScope(blocks=[block])
    registry = _FakeRegistry(
        ops={
            "ConsumerOp": _make_op_info(
                "ConsumerOp",
                {
                    "inoculum_detector": _make_param(
                        "inoculum_detector", is_operation=True
                    )
                },
            )
        }
    )
    section = _build_aux_ports_section(
        block=block, scope=scope, registry=registry
    )
    assert section is not None
    text = _collect_text(section)
    # Header includes the param name
    assert "inoculum_detector" in text
    # Empty placeholder
    assert "Empty" in text
    # Required tag (no default)
    assert "required" in text


def test_scalar_aux_row_renders_wired_state_with_disconnect() -> None:
    """A wired scalar aux port renders the source label + Disconnect button."""

    source_block = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="MyDetector",
    )
    block = BlockNode(
        block_id=_new_block_id(),
        class_name="ConsumerOp",
        params={},
    )
    edge = Edge(
        edge_id="my-edge",
        source_block_id=source_block.block_id,
        target_block_id=block.block_id,
        target_port="inoculum_detector",
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[source_block, block], edges=[edge])
    registry = _FakeRegistry(
        ops={
            "ConsumerOp": _make_op_info(
                "ConsumerOp",
                {
                    "inoculum_detector": _make_param(
                        "inoculum_detector", is_operation=True
                    )
                },
            )
        }
    )
    section = _build_aux_ports_section(
        block=block, scope=scope, registry=registry
    )
    assert section is not None
    text = _collect_text(section)
    assert "MyDetector" in text  # source label
    # Disconnect button present
    btns = _find_by_id(
        section, ids.inspector_disconnect_id("my-edge")
    )
    assert len(btns) == 1


def test_list_aux_row_renders_ordered_slots_with_remove_and_arrows() -> None:
    """A list aux port enumerates slots in order + emits ✕ remove + arrows."""

    a = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="DetectorA",
    )
    b = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="DetectorB",
    )
    consumer = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
        list_slot_counts={"detectors": 3},  # Two wired + one empty
    )
    edge_a = Edge(
        edge_id="ea",
        source_block_id=a.block_id,
        target_block_id=consumer.block_id,
        target_port="detectors",
        target_slot=0,
        kind="aux",
    )
    edge_b = Edge(
        edge_id="eb",
        source_block_id=b.block_id,
        target_block_id=consumer.block_id,
        target_port="detectors",
        target_slot=1,
        kind="aux",
    )
    scope = _DagBuilderScope(
        blocks=[a, b, consumer], edges=[edge_a, edge_b]
    )
    registry = _FakeRegistry(
        ops={
            "CompositeDetector": _make_op_info(
                "CompositeDetector",
                {
                    "detectors": _make_param(
                        "detectors",
                        is_operation=True,
                        is_list=True,
                        has_default=False,
                    )
                },
            )
        }
    )
    section = _build_aux_ports_section(
        block=consumer, scope=scope, registry=registry
    )
    assert section is not None
    text = _collect_text(section)
    assert "DetectorA" in text
    assert "DetectorB" in text
    # Empty slot placeholder
    assert "Empty" in text
    # ✕ remove buttons for both wired edges
    assert _find_by_id(section, ids.inspector_list_remove_id("ea"))
    assert _find_by_id(section, ids.inspector_list_remove_id("eb"))
    # + Add empty slot button keyed by (block_id, param)
    add_btn_id = ids.inspector_add_empty_slot_id(
        consumer.block_id, "detectors"
    )
    assert _find_by_id(section, add_btn_id)
    # Up/down arrow buttons present for each row (pattern-match family).
    move_buttons = _find_by_type_key(section, ids.BTN_INSPECTOR_LIST_MOVE)
    # 3 slots × 2 directions = 6 buttons.
    assert len(move_buttons) == 6


def test_list_aux_row_includes_drag_handle_placeholder() -> None:
    """Each list-aux row carries a drag-handle Span (spec §4.5)."""

    a = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="DetectorA",
    )
    consumer = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
        list_slot_counts={"detectors": 1},
    )
    edge = Edge(
        edge_id="e",
        source_block_id=a.block_id,
        target_block_id=consumer.block_id,
        target_port="detectors",
        target_slot=0,
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[a, consumer], edges=[edge])
    registry = _FakeRegistry(
        ops={
            "CompositeDetector": _make_op_info(
                "CompositeDetector",
                {
                    "detectors": _make_param(
                        "detectors",
                        is_operation=True,
                        is_list=True,
                        has_default=True,  # optional
                    )
                },
            )
        }
    )
    section = _build_aux_ports_section(
        block=consumer, scope=scope, registry=registry
    )
    assert section is not None
    # Drag handle is a Span with the "inspector-drag-handle" class.
    handles = [
        node
        for node in _walk(section)
        if "inspector-drag-handle" in str(getattr(node, "className", ""))
    ]
    assert handles, "Expected at least one drag-handle span per slot"
    # Optional-tag rendered as 'optional' badge
    assert "optional" in _collect_text(section)


def test_list_aux_row_emits_reorder_store_per_param() -> None:
    """Each list-aux param emits one hidden reorder store keyed by (block, param)."""

    a = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    consumer = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
        list_slot_counts={"detectors": 1},
    )
    edge = Edge(
        edge_id="e",
        source_block_id=a.block_id,
        target_block_id=consumer.block_id,
        target_port="detectors",
        target_slot=0,
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[a, consumer], edges=[edge])
    registry = _FakeRegistry(
        ops={
            "CompositeDetector": _make_op_info(
                "CompositeDetector",
                {
                    "detectors": _make_param(
                        "detectors",
                        is_operation=True,
                        is_list=True,
                    )
                },
            )
        }
    )
    section = _build_aux_ports_section(
        block=consumer, scope=scope, registry=registry
    )
    assert section is not None
    store_id = ids.inspector_list_reorder_store_id(
        consumer.block_id, "detectors"
    )
    stores = _find_by_id(section, store_id)
    assert len(stores) == 1
    # Initial data carries the current ordering.
    assert stores[0].data == {"edge_id_order": ["e"]}


def test_aux_section_renders_inside_full_inspector() -> None:
    """End-to-end: ``_build_dag_inspector`` mounts the aux section inside the block view."""

    source = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
        label="SourceA",
    )
    target = BlockNode(
        block_id=_new_block_id(),
        class_name="CompositeDetector",
        params={},
        list_slot_counts={"detectors": 2},
    )
    edge = Edge(
        edge_id="e1",
        source_block_id=source.block_id,
        target_block_id=target.block_id,
        target_port="detectors",
        target_slot=0,
        kind="aux",
    )
    scope = _DagBuilderScope(blocks=[source, target], edges=[edge])
    state = _DagBuilderState(root=scope)
    state.selected_block_id = target.block_id

    registry = _FakeRegistry(
        ops={
            "CompositeDetector": _make_op_info(
                "CompositeDetector",
                {
                    "detectors": _make_param(
                        "detectors",
                        is_operation=True,
                        is_list=True,
                        has_default=False,
                    )
                },
            ),
            "OtsuDetector": _make_op_info(
                "OtsuDetector",
                {},
            ),
        }
    )
    inspector = _build_dag_inspector(state, registry)  # type: ignore[arg-type]
    assert _find_by_id(inspector, ids.INSPECTOR_AUX_SECTION)
    # The wire card should NOT be present when a block (not a wire) is selected.
    assert not _find_by_id(inspector, ids.INSPECTOR_WIRE_CARD)


def test_pipeline_container_block_renders_drill_in_button() -> None:
    """Selecting a container block renders Drill-in but no aux ports section."""

    inner_input = BlockNode(
        block_id=_new_block_id(),
        class_name=INPUT_IMAGE_CLASS_NAME,
        params={},
    )
    inner_op = BlockNode(
        block_id=_new_block_id(),
        class_name="OtsuDetector",
        params={},
    )
    nested = _DagBuilderScope(blocks=[inner_input, inner_op])
    container = BlockNode(
        block_id=_new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        nested=nested,
        label="MyContainer",
    )
    scope = _DagBuilderScope(blocks=[container])
    state = _DagBuilderState(root=scope)
    state.selected_block_id = container.block_id

    inspector = _build_dag_inspector(state, registry=_FakeRegistry())  # type: ignore[arg-type]
    # Container does not emit an aux ports section.
    assert not _find_by_id(inspector, ids.INSPECTOR_AUX_SECTION)
    # The Drill-in button is the visible (non-hidden) one.
    drill_btns = _find_by_id(inspector, ids.BTN_DRILL_IN)
    assert drill_btns


def test_empty_state_card_when_nothing_selected() -> None:
    """No block + no wire selected → empty placeholder, no wire card, no aux section."""

    state = _DagBuilderState(root=_DagBuilderScope())
    # Selection both null → empty state.
    inspector = _build_dag_inspector(state, registry=_FakeRegistry())  # type: ignore[arg-type]
    assert not _find_by_id(inspector, ids.INSPECTOR_WIRE_CARD)
    assert not _find_by_id(inspector, ids.INSPECTOR_AUX_SECTION)
    # Text should mention the palette hint per spec §4.5.
    assert "palette" in _collect_text(inspector).lower()

"""Rendering tests for the fixed linear builder port map."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Optional

from phenotypic.gui._operation_registry import OperationInfo, ParamInfo
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._state import (
    BlockNode,
    BuilderScope,
    BuilderState,
    Edge,
    _new_block_id,
)


@dataclass
class _FakeRegistry:
    ops: dict[str, OperationInfo] = field(default_factory=dict)

    def get(self, name: str) -> Optional[OperationInfo]:
        return self.ops.get(name)

    def get_categories(self) -> list[str]:
        return sorted({info.category for info in self.ops.values()})

    def get_by_category(self, category: str) -> list[OperationInfo]:
        return [info for info in self.ops.values() if info.category == category]


def _param(
    name: str,
    *,
    is_operation: bool = False,
    is_pipeline: bool = False,
    is_list: bool = False,
    has_default: bool = True,
    description: str | None = None,
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
        description=description,
    )


def _op_info(name: str, params: dict[str, ParamInfo]) -> OperationInfo:
    class _Stub:
        pass

    _Stub.__name__ = name
    return OperationInfo(
        cls=_Stub,
        name=name,
        category="Enhancer",
        module="tests.fake",
        docstring=f"{name} class documentation.",
        parameters=params,
    )


def _walk(component: Any):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find_by_id(component: Any, target_id: Any) -> list[Any]:
    return [c for c in _walk(component) if getattr(c, "id", None) == target_id]


def _find_by_class(component: Any, class_name: str) -> list[Any]:
    return [
        c
        for c in _walk(component)
        if class_name in str(getattr(c, "className", "") or "").split()
    ]


def _find_by_type(component: Any, type_key: str) -> list[Any]:
    return [
        c
        for c in _walk(component)
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == type_key
    ]


def _component_id_key(component_id: Any) -> str:
    if isinstance(component_id, dict):
        return json.dumps(component_id, sort_keys=True)
    return str(component_id)


def _state_with_consumer() -> BuilderState:
    scope = BuilderScope()
    input_block = scope.blocks[0]
    block = BlockNode(
        block_id=_new_block_id(),
        class_name="ConsumerOp",
        params={"sigma": 1.0},
    )
    scope.blocks.append(block)
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=input_block.block_id,
            target_block_id=block.block_id,
            target_port="in",
            kind="image",
        )
    )
    return BuilderState(root=scope, selected_block_id=block.block_id)


def _registry() -> _FakeRegistry:
    return _FakeRegistry(
        {
            "ConsumerOp": _op_info(
                "ConsumerOp",
                {
                    "sigma": _param("sigma"),
                    "detector": _param(
                        "detector",
                        is_operation=True,
                        has_default=False,
                        description="Detector operation docstring.",
                    ),
                },
            ),
            "SourceOp": _op_info("SourceOp", {}),
        }
    )


def test_linear_map_empty_scope_renders_container_and_floating_port():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    tree = build_linear_map_section(BuilderState(), _registry())

    assert _find_by_id(tree, ids.LINEAR_MAP_CONTAINER)
    assert _find_by_class(tree, "linear-floating-port")
    assert _find_by_type(tree, ids.LINEAR_PORT)


def test_linear_map_marks_selected_side_port_green():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    state = _state_with_consumer()
    block = state.root.blocks[1]
    state.selected_targets_by_scope = {
        "__root__": {
            "kind": "parameter",
            "scope_path": [],
            "block_id": block.block_id,
            "param": "detector",
            "slot": None,
        }
    }

    tree = build_linear_map_section(state, _registry())

    selected = _find_by_class(tree, "is-selected-target")
    assert selected
    assert any(getattr(port, "id", {}).get("param") == "detector" for port in selected)


def test_linear_map_renders_doc_help_buttons():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    tree = build_linear_map_section(_state_with_consumer(), _registry())

    help_buttons = [
        button
        for button in _find_by_type(tree, ids.LINEAR_NODE_ACTION)
        + _find_by_type(tree, ids.LINEAR_PARAM_ACTION)
        if button.id.get("action") == "help"
    ]
    assert help_buttons
    assert any("documentation" in str(getattr(button, "title", "")).lower() for button in help_buttons)
    assert _find_by_class(tree, "linear-help-popover")


def test_linear_map_unsupported_state_renders_panel():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    state = _state_with_consumer()
    extra = BlockNode(
        block_id=_new_block_id(),
        class_name="SourceOp",
        params={},
    )
    state.root.blocks.append(extra)
    state.root.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=state.root.blocks[0].block_id,
            target_block_id=extra.block_id,
            target_port="in",
            kind="image",
        )
    )

    tree = build_linear_map_section(state, _registry())

    assert _find_by_class(tree, "linear-unsupported-panel")


def test_side_loader_badge_precedes_title_and_port_is_left_aligned():
    from phenotypic.gui.builder._linear_layout import build_linear_side_loader

    tree = build_linear_side_loader(_state_with_consumer(), _registry())

    header = _find_by_class(tree, "linear-side-header")[0]
    assert "linear-side-badge" in str(header.children[0].className)
    assert "linear-side-title" in str(header.children[1].className)

    row = _find_by_class(tree, "linear-side-param-row")[0]
    assert isinstance(row.children[0].id, dict)
    assert row.children[0].id["type"] == ids.LINEAR_PORT


def test_app_layout_mounts_linear_map_instead_of_cytoscape():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(BuilderState(), _registry(), image_root=None)

    assert _find_by_id(tree, ids.LINEAR_MAP_CONTAINER)
    assert not _find_by_id(tree, ids.CANVAS_CYTOSCAPE)


def test_app_layout_has_no_duplicate_ids_with_selected_block():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(_state_with_consumer(), _registry(), image_root=None)

    seen: set[str] = set()
    duplicates: list[Any] = []
    for component in _walk(tree):
        component_id = getattr(component, "id", None)
        if component_id is None:
            continue
        key = _component_id_key(component_id)
        if key in seen:
            duplicates.append(component_id)
        seen.add(key)
    assert duplicates == []


def test_linear_ids_are_exported():
    assert "LINEAR_MAP_CONTAINER" in ids.__all__
    assert "linear_port_id" in ids.__all__
    assert "linear_param_action_id" in ids.__all__

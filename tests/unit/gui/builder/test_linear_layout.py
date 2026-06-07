"""Rendering tests for the fixed linear builder port map."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from phenotypic.gui._operation_registry import OperationInfo, ParamInfo
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder import _linear_layout as linear_layout
from phenotypic.gui.builder._state import (
    BlockNode,
    BuilderScope,
    BuilderState,
    Edge,
    PIPELINE_CLASS_NAME,
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
    assert not _find_by_class(tree, "linear-port-menu")


def test_input_image_card_reserves_left_port_grid_cell():
    """The source card keeps its body in the center grid column."""

    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    tree = build_linear_map_section(BuilderState(), _registry())
    input_cards = [
        card
        for card in _find_by_class(tree, "linear-node-card")
        if "is-input-node" in str(getattr(card, "className", ""))
    ]

    assert len(input_cards) == 1
    child_classes = [
        str(getattr(child, "className", ""))
        for child in input_cards[0].children
    ]
    assert child_classes == [
        "linear-node-port-left linear-node-port-placeholder",
        "linear-node-body",
        "linear-node-port-right",
    ]


def test_linear_map_renders_view_only_zoom_controls():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    tree = build_linear_map_section(BuilderState(), _registry())

    zoom_out = _find_by_id(tree, ids.LINEAR_ZOOM_OUT)
    zoom_in = _find_by_id(tree, ids.LINEAR_ZOOM_IN)
    zoom_reset = _find_by_id(tree, ids.LINEAR_ZOOM_RESET)
    zoom_fit = _find_by_id(tree, ids.LINEAR_ZOOM_FIT)
    assert len(zoom_out) == 1
    assert len(zoom_in) == 1
    assert len(zoom_reset) == 1
    assert len(zoom_fit) == 1
    assert zoom_out[0].to_plotly_json()["props"]["aria-label"] == "Zoom out"
    assert zoom_in[0].to_plotly_json()["props"]["aria-label"] == "Zoom in"
    assert (
        zoom_reset[0].to_plotly_json()["props"]["aria-label"]
        == "Reset zoom to 100%"
    )
    assert zoom_fit[0].to_plotly_json()["props"]["aria-label"] == "Fit full pipeline"
    assert _find_by_class(tree, "linear-map-zoom-controls")
    assert _find_by_class(tree, "linear-map-lucide-icon")


def test_linear_fit_icon_falls_back_when_scan_is_unavailable(monkeypatch):
    """The fit button uses Maximize2 when Scan is missing from lucide data."""

    called: list[str] = []

    def fake_lucide_icon(icon_name: str, **_: Any) -> str:
        called.append(icon_name)
        return f'<svg class="lucide lucide-{icon_name}"></svg>'

    monkeypatch.setattr(linear_layout, "get_icon_list", lambda: ["maximize-2"])
    monkeypatch.setattr(linear_layout, "lucide_icon", fake_lucide_icon)

    icon = linear_layout._fit_icon()

    assert called == ["maximize-2"]
    assert "linear-map-lucide-icon" in str(icon.className)


def test_linear_map_renders_port_menu_only_for_open_target():
    from phenotypic.gui.builder._linear_layout import build_linear_map_section

    state = _state_with_consumer()
    block = state.root.blocks[1]
    state.open_port_menu = {
        "kind": "image_output",
        "scope_path": [],
        "block_id": block.block_id,
        "param": None,
        "slot": None,
    }

    tree = build_linear_map_section(state, _registry())

    menus = _find_by_class(tree, "linear-port-menu")
    assert len(menus) == 1
    assert menus[0].id["block_id"] == block.block_id
    assert _find_by_class(menus[0], "linear-port-menu-close")


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
    export_buttons = [
        button
        for button in _find_by_type(tree, ids.LINEAR_NODE_ACTION)
        if button.id.get("action") == "export_raw_state"
    ]
    assert export_buttons
    assert "linear-unsupported-export-action" in export_buttons[0].className
    reset_buttons = [
        button
        for button in _find_by_type(tree, ids.LINEAR_NODE_ACTION)
        if button.id.get("action") == "start_new_state"
    ]
    assert reset_buttons
    assert "linear-unsupported-reset-action" in reset_buttons[0].className


def test_side_loader_badge_precedes_title_and_port_is_left_aligned():
    from phenotypic.gui.builder._linear_layout import build_linear_side_loader

    tree = build_linear_side_loader(_state_with_consumer(), _registry())

    header = _find_by_class(tree, "linear-side-header")[0]
    assert "linear-side-badge" in str(header.children[0].className)
    assert "linear-side-title" in str(header.children[1].className)

    row = _find_by_class(tree, "linear-side-param-row")[0]
    assert isinstance(row.children[0].id, dict)
    assert row.children[0].id["type"] == ids.LINEAR_PORT


def test_breadcrumb_renders_dag_nested_scope_labels():
    """DAG string breadcrumbs resolve through blocks, not legacy nodes."""

    from phenotypic.gui.builder._layout import build_breadcrumb

    nested = BuilderScope(name="Inoculum detector")
    container = BlockNode(
        block_id=_new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        label="Embedded pipeline",
        nested=nested,
    )
    root = BuilderScope()
    root.blocks.append(container)
    state = BuilderState(root=root, breadcrumb=[container.block_id])

    breadcrumb = build_breadcrumb(state)

    assert breadcrumb.children[0].children == "Pipeline"
    assert breadcrumb.children[-1].children == "Embedded pipeline"


def test_side_loader_renders_linear_move_and_delete_actions():
    from phenotypic.gui.builder._linear_layout import build_linear_side_loader

    tree = build_linear_side_loader(_state_with_consumer(), _registry())

    actions = [
        component.id
        for component in _find_by_type(tree, ids.LINEAR_NODE_ACTION)
        if component.id.get("surface") == "side"
    ]
    assert {"move_left", "move_right", "delete"} <= {
        action["action"] for action in actions
    }


def test_app_layout_mounts_linear_map_instead_of_cytoscape():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(BuilderState(), _registry(), image_root=None)

    assert _find_by_id(tree, ids.LINEAR_MAP_CONTAINER)
    assert not _find_by_id(tree, ids.CANVAS_CYTOSCAPE)


def test_app_layout_uses_desktop_three_column_linear_builder():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(BuilderState(), _registry(), image_root=None)

    assert _find_by_class(tree, "linear-builder-map-column")
    assert _find_by_class(tree, "linear-builder-side-column")


def test_app_layout_keeps_retired_viewport_controls_hidden_and_inert():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(BuilderState(), _registry(), image_root=None)

    banner = _find_by_id(tree, ids.BANNER_ASSET_STATUS)
    relayout = _find_by_id(tree, ids.BTN_RELAYOUT)
    reanchor = _find_by_id(tree, ids.BTN_REANCHOR)
    assert len(banner) == 1
    assert len(relayout) == 1
    assert len(reanchor) == 1
    assert _find_by_id(tree, ids.DOWNLOAD_RAW_STATE)
    assert banner[0].style == {"display": "none"}
    assert relayout[0].disabled is True
    assert reanchor[0].disabled is True
    assert relayout[0].style == {"display": "none"}
    assert reanchor[0].style == {"display": "none"}


def test_app_layout_palette_is_click_only_not_draggable():
    from phenotypic.gui.builder._layout import build_app_layout

    tree = build_app_layout(BuilderState(), _registry(), image_root=None)

    assert not any(getattr(component, "draggable", None) == "true" for component in _walk(tree))
    assert any(getattr(component, "draggable", None) == "false" for component in _walk(tree))
    assert not any(
        "data-palette-class" in getattr(component, "__dict__", {})
        for component in _walk(tree)
    )


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
    assert "LINEAR_ZOOM_OUT" in ids.__all__
    assert "LINEAR_ZOOM_IN" in ids.__all__
    assert "LINEAR_ZOOM_RESET" in ids.__all__
    assert "LINEAR_ZOOM_FIT" in ids.__all__


def test_mobile_limited_mode_css_keeps_help_and_drill_available():
    css_path = (
        Path(__file__).parents[4]
        / "src/phenotypic/gui/builder/assets/builder.css"
    )
    css = css_path.read_text()

    assert "@media (max-width: 768px)" in css
    assert ".linear-port-button" in css
    assert ".palette-button,\n    .linear-port-button" not in css
    assert ".linear-port-menu-action:not(.linear-port-menu-close)" in css
    assert ".linear-side-action:not(.linear-side-drill-action)" in css
    assert "#btn-save" in css
    assert ".linear-help-button" in css
    assert ".linear-unsupported-export-action" in css
    assert ".linear-port-menu-close" in css
    assert ".linear-map-zoom-control" in css
    assert ".linear-map-lucide-icon" in css
    assert ".linear-map-track" in css
    assert "gap: 0;" in css
    assert "width: 24px;" in css
    assert "height: 24px;" in css
    assert "grid-template-columns: 18px" not in css
    assert ".linear-side-drill-action" in css


def test_mobile_limited_mode_js_applies_real_disabled_and_readonly_attributes():
    js_path = (
        Path(__file__).parents[4]
        / "src/phenotypic/gui/builder/assets/builder.js"
    )
    js = js_path.read_text()

    assert 'const MOBILE_LIMITED_QUERY = "(max-width: 768px)"' in js
    assert '".linear-port-button"' in js
    assert '".palette-button",\n        ".linear-port-button"' not in js
    assert '".linear-port-menu-action:not(.linear-port-menu-close)"' in js
    assert '".linear-map-zoom-control"' in js
    assert '".linear-unsupported-export-action"' in js
    assert '".linear-unsupported-reset-action"' not in js
    assert "phenoLinearMapMounted" in js
    assert "el.disabled = limited" in js
    assert "el.readOnly = limited" in js
    assert 'el.setAttribute("aria-disabled", limited ? "true" : "false")' in js


def test_linear_zoom_js_is_ui_only_and_preserves_clickable_ports():
    js_path = (
        Path(__file__).parents[4]
        / "src/phenotypic/gui/builder/assets/builder.js"
    )
    js = js_path.read_text()

    assert "linear-map-zoom-viewport" in js
    assert "linear-map-zoom-content" in js
    assert "transformOrigin = \"left center\"" in js
    assert "scrollIntoView" not in js
    assert "store-builder-state" not in js[js.find("linear-map zoom") :]

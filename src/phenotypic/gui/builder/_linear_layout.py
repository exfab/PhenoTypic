"""Fixed linear port-map layout for the pipeline builder.

The visible builder constrains the DAG state to a single image spine with
side-loaded operation/pipeline parameters. This module renders that view with
regular Dash/HTML components so every visible port is an accessible button.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui._operation_registry import OperationInfo
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._linear_model import (
    LinearTarget,
    default_continuation_target,
    derive_linear_scope,
    resolve_selected_target,
    scope_at_path,
    target_from_dict,
    target_to_dict,
)
from phenotypic.gui.builder._param_form import param_form
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    BuilderState,
    Edge,
)

if TYPE_CHECKING:  # pragma: no cover
    from phenotypic.gui._operation_registry import OperationRegistry


_HIDDEN_STYLE = {"display": "none"}


def _target_matches(a: LinearTarget, b: LinearTarget) -> bool:
    return target_to_dict(a) == target_to_dict(b)


def _class_names(*parts: Optional[str]) -> str:
    return " ".join(part for part in parts if part)


def _port_label(target: LinearTarget) -> str:
    if target.kind == "continuation":
        return "Add next operation"
    if target.kind == "image_output":
        return "Image output"
    if target.kind == "image_input":
        return "Image input"
    if target.kind == "parameter_slot":
        suffix = f" slot {target.slot}" if target.slot is not None else ""
        return f"Fill {target.param}{suffix}"
    return f"Fill {target.param}"


def _port_button(
    target: LinearTarget,
    selected_target: LinearTarget,
    *,
    className: str,
    children: Any = "",
    title: Optional[str] = None,
) -> html.Button:
    selected = _target_matches(target, selected_target)
    return html.Button(
        children,
        id=ids.linear_port_id(
            kind=target.kind,
            scope_path=target.scope_path,
            block_id=target.block_id,
            param=target.param,
            slot=target.slot,
        ),
        type="button",
        n_clicks=0,
        title=title or _port_label(target),
        className=_class_names(
            "linear-port-button",
            className,
            "is-selected-target" if selected else None,
        ),
        **{"aria-label": title or _port_label(target)},  # type: ignore[arg-type]
    )


def _help_button(
    *,
    action_scope: str,
    scope_path: Iterable[str],
    block_id: Optional[str],
    title: str,
    param: Optional[str] = None,
    source_block_id: Optional[str] = None,
) -> html.Button:
    """Render a compact docstring help button."""

    if action_scope == "node":
        button_id = ids.linear_node_action_id(
            action="help",
            scope_path=list(scope_path),
            block_id=block_id,
        )
    else:
        button_id = ids.linear_param_action_id(
            action="help",
            scope_path=list(scope_path),
            block_id=block_id,
            param=param,
            source_block_id=source_block_id,
        )
    return html.Button(
        "?",
        id=button_id,
        type="button",
        n_clicks=0,
        title=title,
        className="linear-help-button",
        **{"aria-label": "Show documentation"},  # type: ignore[arg-type]
    )


def _hidden_inspector_widgets() -> List[Any]:
    """Keep legacy inspector callback ids resolvable in every side-loader branch."""

    return [
        dbc.Input(id=ids.INPUT_NODE_LABEL, type="text", style=_HIDDEN_STYLE),
        dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
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


def _op_doc(registry: "OperationRegistry", class_name: str) -> str:
    info = registry.get(class_name)
    doc = getattr(info, "docstring", None) if info is not None else None
    return str(doc or f"No documentation found for {class_name}.")


def _param_doc(param_info: Any) -> str:
    description = getattr(param_info, "description", None)
    return str(description or "No parameter documentation found.")


def _aux_params(info: Optional[OperationInfo]) -> List[tuple[str, Any]]:
    if info is None:
        return []
    return [
        (name, param)
        for name, param in info.parameters.items()
        if param.is_operation or param.is_pipeline
    ]


def _scalar_info(info: OperationInfo) -> OperationInfo:
    params = {
        name: param
        for name, param in info.parameters.items()
        if not param.is_operation and not param.is_pipeline
    }
    return replace(info, parameters=params)


def _block_label(block: BlockNode) -> str:
    return block.label or block.class_name


def _source_for_edge(scope_blocks: List[BlockNode], edge: Edge) -> Optional[BlockNode]:
    return next(
        (block for block in scope_blocks if block.block_id == edge.source_block_id),
        None,
    )


def _edges_for_param(scope_edges: List[Edge], block_id: str, param: str) -> List[Edge]:
    return [
        edge for edge in scope_edges
        if edge.kind == "aux"
        and edge.target_block_id == block_id
        and edge.target_port == param
    ]


def _stage_badge_label(block: BlockNode) -> str:
    if block.class_name == INPUT_IMAGE_CLASS_NAME:
        return "INPUT"
    if block.class_name == PIPELINE_CLASS_NAME:
        return "PIPELINE"
    return "OP"


def _active_target_strip(target: LinearTarget) -> html.Div:
    return html.Div(
        [
            html.Span("Active target", className="linear-target-strip-label"),
            html.Span(_port_label(target), className="linear-target-strip-value"),
        ],
        className="linear-target-strip",
    )


def _port_menu(target: LinearTarget) -> html.Div:
    actions: List[Any] = []
    if target.kind in {"continuation", "image_output"}:
        actions.append(
            html.Button(
                "Preview here",
                id=ids.linear_node_action_id(
                    action="preview_here",
                    scope_path=target.scope_path,
                    block_id=target.block_id,
                ),
                type="button",
                n_clicks=0,
                className="linear-port-menu-action",
            )
        )
    actions.append(
        html.Button(
            "Close",
            id=ids.linear_node_action_id(
                action="target_menu_close",
                scope_path=target.scope_path,
                block_id=target.block_id,
            ),
            type="button",
            n_clicks=0,
            className="linear-port-menu-action",
        )
    )
    return html.Div(
        actions,
        id=ids.linear_port_menu_id(
            kind=target.kind,
            scope_path=target.scope_path,
            block_id=target.block_id,
            param=target.param,
            slot=target.slot,
        ),
        className="linear-port-menu",
    )


def _maybe_port_menu(
    state: BuilderState, target: LinearTarget
) -> Optional[html.Div]:
    open_target = target_from_dict(state.open_port_menu, target.scope_path)
    if _target_matches(open_target, target):
        return _port_menu(target)
    return None


def _image_connector() -> html.Div:
    return html.Div(
        [html.Div(className="linear-connector-line")],
        className="linear-connector",
    )


def _block_card(
    *,
    state: BuilderState,
    registry: "OperationRegistry",
    block: BlockNode,
    selected_target: LinearTarget,
    unknown: bool,
) -> html.Div:
    scope_path = list(state.breadcrumb)
    info = registry.get(block.class_name)
    title = _block_label(block)

    left_port: Optional[Any] = None
    if block.class_name != INPUT_IMAGE_CLASS_NAME:
        input_target = LinearTarget(
            kind="image_input",
            scope_path=scope_path,
            block_id=block.block_id,
        )
        left_port = html.Div(
            [
                _port_button(
                    input_target,
                    selected_target,
                    className="linear-port-image-in",
                    title=f"Insert before {title}",
                ),
                _maybe_port_menu(state, input_target),
            ],
            className="linear-node-port-left",
        )

    output_target = LinearTarget(
        kind="image_output",
        scope_path=scope_path,
        block_id=block.block_id,
    )
    right_port = html.Div(
        [
            _port_button(
                output_target,
                selected_target,
                className="linear-port-image-out",
                title=f"Insert after {title}",
            ),
            _maybe_port_menu(state, output_target),
        ],
        className="linear-node-port-right",
    )

    param_rows: List[Any] = []
    for param_name, param_info in _aux_params(info):
        target = LinearTarget(
            kind="parameter",
            scope_path=scope_path,
            block_id=block.block_id,
            param=param_name,
        )
        param_rows.append(
            html.Div(
                [
                    _port_button(
                        target,
                        selected_target,
                        className="linear-port-param",
                        title=f"Fill {param_name}",
                    ),
                    html.Span(param_name, className="linear-node-param-name"),
                    _help_button(
                        action_scope="param",
                        scope_path=scope_path,
                        block_id=block.block_id,
                        param=param_name,
                        title=_param_doc(param_info),
                    ),
                    _maybe_port_menu(state, target),
                ],
                className="linear-node-param-row",
            )
        )

    badges: List[Any] = [
        html.Span(_stage_badge_label(block), className="linear-node-badge")
    ]
    if unknown:
        badges.append(html.Span("UNKNOWN", className="linear-node-badge-warning"))

    return html.Div(
        [
            left_port,
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(badges, className="linear-node-badges"),
                            html.Button(
                                title,
                                id=ids.linear_node_action_id(
                                    action="select",
                                    scope_path=scope_path,
                                    block_id=block.block_id,
                                ),
                                type="button",
                                n_clicks=0,
                                className="linear-node-title-button",
                            ),
                            _help_button(
                                action_scope="node",
                                scope_path=scope_path,
                                block_id=block.block_id,
                                title=_op_doc(registry, block.class_name),
                            ),
                        ],
                        className="linear-node-header",
                    ),
                    html.Div(param_rows, className="linear-node-params"),
                ],
                className="linear-node-body",
            ),
            right_port,
        ],
        className=_class_names(
            "linear-node-card",
            "is-selected-node" if state.selected_block_id == block.block_id else None,
            "is-input-node" if block.class_name == INPUT_IMAGE_CLASS_NAME else None,
        ),
    )


def _unsupported_panel(reason: str, detail: str) -> html.Div:
    return html.Div(
        [
            html.H6("Unsupported DAG shape", className="mb-2"),
            html.Div(reason, className="linear-unsupported-reason"),
            html.P(detail, className="mb-0 text-muted"),
        ],
        className="linear-unsupported-panel",
    )


def build_linear_map_section(
    state: BuilderState,
    registry: "OperationRegistry",
) -> html.Div:
    """Render the fixed linear map as the default builder canvas."""

    scope = scope_at_path(state.root, state.breadcrumb)
    selected_target = resolve_selected_target(state)

    header = html.Div(
        [
            html.H6("Pipeline map", className="mb-0"),
            html.Div("Fixed linear view", className="linear-map-kicker"),
        ],
        className="linear-map-header",
    )

    if scope is None:
        body: Any = _unsupported_panel(
            "stale_breadcrumb",
            "The selected nested scope is no longer available.",
        )
    else:
        model = derive_linear_scope(scope, scope_path=state.breadcrumb)
        if model.unsupported is not None:
            body = _unsupported_panel(model.unsupported.reason, model.unsupported.detail)
        else:
            track_children: List[Any] = []
            for idx, block in enumerate(model.spine_blocks):
                if idx > 0:
                    track_children.append(_image_connector())
                track_children.append(
                    _block_card(
                        state=state,
                        registry=registry,
                        block=block,
                        selected_target=selected_target,
                        unknown=block.block_id in model.unknown_block_ids,
                    )
                )
            floating_target = default_continuation_target(state.breadcrumb)
            track_children.extend(
                [
                    html.Div(
                        [html.Div(className="linear-terminal-line")],
                        className="linear-terminal",
                    ),
                    html.Div(
                        [
                            _port_button(
                                floating_target,
                                selected_target,
                                className="linear-floating-port",
                                children="+",
                                title="Add next operation",
                            ),
                            _maybe_port_menu(state, floating_target),
                        ],
                        className="linear-floating-port-wrap",
                    ),
                ]
            )
            body = html.Div(track_children, className="linear-map-track")

    return html.Div(
        [header, html.Div(body, id=ids.LINEAR_MAP_CONTAINER, className="linear-map")],
        className="linear-map-section",
    )


def _selected_block_and_scope(
    state: BuilderState,
) -> tuple[Optional[Any], Optional[BlockNode]]:
    scope = scope_at_path(state.root, state.breadcrumb)
    if scope is None or state.selected_block_id is None:
        return scope, None
    block = next(
        (candidate for candidate in scope.blocks if candidate.block_id == state.selected_block_id),
        None,
    )
    return scope, block


def _value_row(
    *,
    state: BuilderState,
    registry: "OperationRegistry",
    block: BlockNode,
    param_name: str,
    edge: Edge,
    slot: Optional[int],
    source: Optional[BlockNode],
) -> html.Div:
    scope_path = list(state.breadcrumb)
    source_label = _block_label(source) if source is not None else edge.source_block_id
    source_block_id = source.block_id if source is not None else edge.source_block_id
    actions: List[Any] = [
        html.Button(
            "Replace",
            id=ids.linear_param_action_id(
                action="replace",
                scope_path=scope_path,
                block_id=block.block_id,
                param=param_name,
                slot=slot,
                source_block_id=source_block_id,
            ),
            type="button",
            n_clicks=0,
            className="linear-side-action",
        ),
        html.Button(
            "Clear",
            id=ids.linear_param_action_id(
                action="clear",
                scope_path=scope_path,
                block_id=block.block_id,
                param=param_name,
                slot=slot,
                source_block_id=source_block_id,
            ),
            type="button",
            n_clicks=0,
            className="linear-side-action",
        ),
    ]
    if source is not None and source.class_name == PIPELINE_CLASS_NAME:
        actions.append(
            html.Button(
                "Edit",
                id=ids.linear_param_action_id(
                    action="drill",
                    scope_path=scope_path,
                    block_id=block.block_id,
                    param=param_name,
                    slot=slot,
                    source_block_id=source_block_id,
                ),
                type="button",
                n_clicks=0,
                className="linear-side-action",
            )
        )
    actions.append(
        _help_button(
            action_scope="param",
            scope_path=scope_path,
            block_id=block.block_id,
            param=param_name,
            source_block_id=source_block_id,
            title=_op_doc(registry, source.class_name) if source is not None else "",
        )
    )
    return html.Div(
        [
            html.Span(source_label, className="linear-side-value-label"),
            html.Div(actions, className="linear-side-value-actions"),
        ],
        className="linear-side-value-row",
    )


def _side_param_row(
    *,
    state: BuilderState,
    registry: "OperationRegistry",
    block: BlockNode,
    param_name: str,
    param_info: Any,
    edges: List[Edge],
    scope_blocks: List[BlockNode],
    selected_target: LinearTarget,
) -> html.Div:
    scope_path = list(state.breadcrumb)
    target = LinearTarget(
        kind="parameter",
        scope_path=scope_path,
        block_id=block.block_id,
        param=param_name,
    )
    ordered = sorted(
        edges,
        key=lambda edge: edge.target_slot if edge.target_slot is not None else 0,
    )
    values: List[Any] = []
    for edge in ordered:
        source = _source_for_edge(scope_blocks, edge)
        values.append(
            _value_row(
                state=state,
                registry=registry,
                block=block,
                param_name=param_name,
                edge=edge,
                slot=edge.target_slot,
                source=source,
            )
        )
    if not values:
        values.append(html.Div("Empty", className="linear-side-empty-value"))

    return html.Div(
        [
            _port_button(
                target,
                selected_target,
                className="linear-port-param linear-side-param-port",
                title=f"Fill {param_name}",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(param_name),
                            html.Span(
                                "required" if not param_info.has_default else "optional",
                                className="linear-side-param-meta",
                            ),
                            _help_button(
                                action_scope="param",
                                scope_path=scope_path,
                                block_id=block.block_id,
                                param=param_name,
                                title=_param_doc(param_info),
                            ),
                        ],
                        className="linear-side-param-heading",
                    ),
                    html.Div(values, className="linear-side-values"),
                ],
                className="linear-side-param-body",
            ),
        ],
        className="linear-side-param-row",
    )


def _side_loader_empty(state: BuilderState) -> html.Div:
    active_target = resolve_selected_target(state)
    return html.Div(
        [
            _active_target_strip(active_target),
            html.Div(
                [
                    html.Div("TARGET", className="linear-side-badge"),
                    html.H5("Side loader", className="linear-side-title"),
                ],
                className="linear-side-header",
            ),
            html.P("Select a node to edit parameters.", className="text-muted mb-0"),
            *_hidden_inspector_widgets(),
        ],
        id=ids.INSPECTOR_CONTAINER,
        className="linear-side-loader",
    )


def build_linear_side_loader(
    state: BuilderState,
    registry: "OperationRegistry",
) -> html.Div:
    """Render the right-side loader for scalar params and side values."""

    active_target = resolve_selected_target(state)
    scope, block = _selected_block_and_scope(state)
    if scope is None or block is None:
        return _side_loader_empty(state)

    title = _block_label(block)
    header = html.Div(
        [
            html.Div(_stage_badge_label(block), className="linear-side-badge"),
            html.H5(title, className="linear-side-title"),
            _help_button(
                action_scope="node",
                scope_path=state.breadcrumb,
                block_id=block.block_id,
                title=_op_doc(registry, block.class_name),
            ),
        ],
        className="linear-side-header",
    )

    label_input = dbc.InputGroup(
        [
            dbc.InputGroupText("Label"),
            dbc.Input(
                id=ids.INPUT_NODE_LABEL,
                type="text",
                value=title,
                debounce=True,
            ),
        ],
        className="mb-3",
    )

    body: List[Any] = [_active_target_strip(active_target), header, label_input]

    if block.class_name == INPUT_IMAGE_CLASS_NAME:
        body.append(
            html.P(
                "This is the runtime image source for the active scope.",
                className="text-muted small",
            )
        )
    elif block.class_name == PIPELINE_CLASS_NAME:
        body.append(
            dbc.Button(
                "Edit pipeline",
                id=ids.linear_node_action_id(
                    action="drill",
                    scope_path=state.breadcrumb,
                    block_id=block.block_id,
                ),
                color="primary",
                outline=True,
                size="sm",
                n_clicks=0,
            )
        )
    else:
        info = registry.get(block.class_name)
        if info is None:
            body.append(
                html.Div(
                    f"Unknown operation '{block.class_name}'.",
                    className="text-warning",
                )
            )
        else:
            scalar_info = _scalar_info(info)
            if scalar_info.parameters:
                body.append(
                    html.Div(
                        param_form(
                            scalar_info,
                            current_values=block.params,
                            form_id_prefix=block.block_id,
                        ),
                        id=ids.INSPECTOR_PARAM_FORM,
                        className="linear-side-param-form",
                    )
                )
            else:
                body.append(html.Div(id=ids.INSPECTOR_PARAM_FORM))

            aux_rows: List[Any] = []
            for param_name, param_info in _aux_params(info):
                aux_rows.append(
                    _side_param_row(
                        state=state,
                        registry=registry,
                        block=block,
                        param_name=param_name,
                        param_info=param_info,
                        edges=_edges_for_param(scope.edges, block.block_id, param_name),
                        scope_blocks=scope.blocks,
                        selected_target=active_target,
                    )
                )
            if aux_rows:
                body.append(
                    html.Div(
                        [
                            html.H6("Side values", className="mb-2"),
                            *aux_rows,
                        ],
                        className="linear-side-values-section",
                    )
                )

    body.extend(
        [
            html.Div(
                "(Run preview to populate)",
                id=ids.INSPECTOR_PREVIEW,
                className="text-muted small fst-italic mt-3",
            ),
            dbc.Button(id=ids.BTN_DRILL_IN, n_clicks=0, style=_HIDDEN_STYLE),
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
    )

    return html.Div(
        body,
        id=ids.INSPECTOR_CONTAINER,
        className="linear-side-loader",
    )


__all__ = [
    "build_linear_map_section",
    "build_linear_side_loader",
]

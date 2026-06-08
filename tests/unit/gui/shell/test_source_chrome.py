"""Unit guards for source-image-root chrome components."""
from __future__ import annotations

import json
from pathlib import Path

from dash import dcc

from phenotypic.gui.shell._layout import build_top_bar, wrap_in_chrome
from phenotypic.gui.shell._sandbox import SandboxRoot


def _walk_components(node: object) -> list[object]:
    nodes = [node]
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            nodes.extend(_walk_components(child))
    elif children is not None:
        nodes.extend(_walk_components(children))
    return nodes


def _component_with_id(node: object, component_id: str) -> object | None:
    for component in _walk_components(node):
        if getattr(component, "id", None) == component_id:
            return component
    return None


def test_top_bar_renders_source_label_and_clear_action(tmp_path: Path) -> None:
    from phenotypic.gui.shell._ids import (
        SHELL_SOURCE_IMAGE_ROOT_CLEAR,
        SHELL_SOURCE_IMAGE_ROOT_LABEL,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    top_bar = build_top_bar(active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    assert _component_with_id(top_bar, SHELL_SOURCE_IMAGE_ROOT_LABEL) is not None
    assert _component_with_id(top_bar, SHELL_SOURCE_IMAGE_ROOT_CLEAR) is not None


def test_wrap_in_chrome_mounts_local_source_store(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    store = _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_STORE)
    assert isinstance(store, dcc.Store)
    assert store.storage_type == "local"
    assert store.data is None


def test_clear_action_registers_source_store_writer(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_SOURCE_IMAGE_ROOT_CLEAR,
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    source_store_callbacks = [
        meta
        for callback_id, meta in app.callback_map.items()
        if callback_id.startswith(f"{SHELL_SOURCE_IMAGE_ROOT_STORE}.data")
    ]
    assert any(
        SHELL_SOURCE_IMAGE_ROOT_CLEAR in json.dumps(meta["inputs"])
        for meta in source_store_callbacks
    )

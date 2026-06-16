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


def test_top_bar_renders_settings_button_not_inline_source_controls(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._ids import (
        SHELL_SETTINGS_BUTTON,
        SHELL_SOURCE_IMAGE_ROOT_CLEAR,
        SHELL_SOURCE_IMAGE_ROOT_LABEL,
        SHELL_SOURCE_IMAGE_ROOT_MODAL,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    top_bar = build_top_bar(active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    assert _component_with_id(top_bar, SHELL_SETTINGS_BUTTON) is not None
    settings_button = _component_with_id(top_bar, SHELL_SETTINGS_BUTTON)
    children = getattr(settings_button, "children", None)
    assert type(children).__name__ == "Img"
    assert "lucide-settings" in getattr(children, "src", "")
    assert _component_with_id(top_bar, SHELL_SOURCE_IMAGE_ROOT_LABEL) is None
    assert _component_with_id(top_bar, SHELL_SOURCE_IMAGE_ROOT_CLEAR) is None
    assert _component_with_id(top_bar, SHELL_SOURCE_IMAGE_ROOT_MODAL) is None


def test_wrap_in_chrome_mounts_local_settings_stores(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_METADATA_CSV_STORE,
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_TAB_HOME,
        TUNE_PIPELINE_PATH_STORE,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    store = _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_STORE)
    assert isinstance(store, dcc.Store)
    assert store.storage_type == "local"
    assert store.data is None

    metadata_store = _component_with_id(app.layout, SHELL_METADATA_CSV_STORE)
    assert isinstance(metadata_store, dcc.Store)
    assert metadata_store.storage_type == "local"
    assert metadata_store.data is None

    tune_store = _component_with_id(app.layout, TUNE_PIPELINE_PATH_STORE)
    assert isinstance(tune_store, dcc.Store)
    assert tune_store.storage_type == "local"
    assert tune_store.data is None


def test_clear_action_registers_source_store_writer(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_SETTINGS_INPUT_FOLDER_CLEAR,
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
        SHELL_SETTINGS_INPUT_FOLDER_CLEAR in json.dumps(meta["inputs"])
        for meta in source_store_callbacks
    )


def test_wrap_in_chrome_mounts_source_picker_modal(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE,
        SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
        SHELL_SOURCE_IMAGE_ROOT_MODAL,
        SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    assert _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_MODAL) is not None
    assert (
        _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY)
        is not None
    )
    assert (
        _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_CONFIRM)
        is not None
    )
    store = _component_with_id(app.layout, SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE)
    assert isinstance(store, dcc.Store)
    assert store.data == str(tmp_path.resolve())


def test_wrap_in_chrome_mounts_settings_popover_and_metadata_picker(
    tmp_path: Path,
) -> None:
    import dash

    from phenotypic.gui.shell import _layout as shell_layout
    from phenotypic.gui.shell._ids import (
        SHELL_METADATA_CSV_BROWSE_STORE,
        SHELL_METADATA_CSV_CONFIRM,
        SHELL_METADATA_CSV_MODAL,
        SHELL_METADATA_CSV_MODAL_BODY,
        SHELL_SETTINGS_POPOVER,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    popover = _component_with_id(app.layout, SHELL_SETTINGS_POPOVER)
    assert popover is not None
    assert getattr(popover, "className", "") == "shell-settings-popover"
    assert ".shell-settings-popover" in shell_layout._SHELL_CSS
    assert "max-width: min(420px, calc(100vw - 32px));" in shell_layout._SHELL_CSS
    assert "grid-template-columns: minmax(96px, max-content) minmax(0, 1fr);" in (
        shell_layout._SHELL_CSS
    )
    assert "word-break: break-word;" in shell_layout._SHELL_CSS
    assert "white-space: normal;" in shell_layout._SHELL_CSS
    assert _component_with_id(app.layout, SHELL_METADATA_CSV_MODAL) is not None
    assert _component_with_id(app.layout, SHELL_METADATA_CSV_MODAL_BODY) is not None
    assert _component_with_id(app.layout, SHELL_METADATA_CSV_CONFIRM) is not None
    store = _component_with_id(app.layout, SHELL_METADATA_CSV_BROWSE_STORE)
    assert isinstance(store, dcc.Store)
    assert store.data == str(tmp_path.resolve())


def test_source_picker_registers_store_writer(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
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
        if f"{SHELL_SOURCE_IMAGE_ROOT_STORE}.data" in callback_id
    ]
    assert any(
        SHELL_SOURCE_IMAGE_ROOT_CONFIRM in json.dumps(meta["inputs"])
        for meta in source_store_callbacks
    )


def test_metadata_picker_registers_store_writer(tmp_path: Path) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_METADATA_CSV_CONFIRM,
        SHELL_METADATA_CSV_STORE,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    metadata_store_callbacks = [
        meta
        for callback_id, meta in app.callback_map.items()
        if f"{SHELL_METADATA_CSV_STORE}.data" in callback_id
    ]
    assert any(
        SHELL_METADATA_CSV_CONFIRM in json.dumps(meta["inputs"])
        for meta in metadata_store_callbacks
    )

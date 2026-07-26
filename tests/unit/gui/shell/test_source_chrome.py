"""Unit guards for source-image-root chrome components."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from urllib.parse import unquote

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
    settings_icon_src = getattr(children, "src", "")
    assert "lucide-settings" in settings_icon_src
    assert 'stroke="#f8fafc"' in unquote(settings_icon_src)
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


def test_shell_callback_map_shared_store_writers_are_explicit_actions_only(
    tmp_path: Path,
) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_METADATA_CSV_CONFIRM,
        SHELL_METADATA_CSV_STORE,
        SHELL_SETTINGS_INPUT_FOLDER_CLEAR,
        SHELL_SETTINGS_METADATA_CSV_CLEAR,
        SHELL_SIDEBAR_SELECTION_STORE,
        SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")
    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    def _writer_inputs(store_id: str) -> set[str]:
        inputs: set[str] = set()
        for callback_id, meta in app.callback_map.items():
            if f"{store_id}.data" not in callback_id:
                continue
            inputs.update(item["id"] for item in meta["inputs"])
        return inputs

    assert _writer_inputs(SHELL_SOURCE_IMAGE_ROOT_STORE) == {
        SHELL_SETTINGS_INPUT_FOLDER_CLEAR,
        SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
        SHELL_SIDEBAR_SELECTION_STORE,
    }
    assert _writer_inputs(SHELL_METADATA_CSV_STORE) == {
        SHELL_SETTINGS_METADATA_CSV_CLEAR,
        SHELL_METADATA_CSV_CONFIRM,
    }


def test_repository_shared_store_writer_inventory_is_explicit() -> None:
    """Audit every production ``Output`` declaration for shared-store writes.

    The Run and Tune entries are deliberately enumerated as deferred generic
    writers. This test must be updated when R0/T1 remove them; it does not
    mistake the Shell callback map for global shared-store authority.
    """
    source_root = Path(__file__).parents[4] / "src" / "phenotypic" / "gui"
    target_names = {
        "SHELL_SOURCE_IMAGE_ROOT_STORE",
        "SHELL_METADATA_CSV_STORE",
    }
    inventory: set[tuple[str, str, str]] = set()

    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class _OutputVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_name = "<module>"

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                previous = self.function_name
                self.function_name = node.name
                self.generic_visit(node)
                self.function_name = previous

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "Output"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in target_names
                ):
                    inventory.add(
                        (
                            path.relative_to(source_root).as_posix(),
                            self.function_name,
                            node.args[0].id,
                        )
                    )
                self.generic_visit(node)

        _OutputVisitor().visit(tree)

    assert inventory == {
        (
            "shell/_callbacks.py",
            "_clear_source_root",
            "SHELL_SOURCE_IMAGE_ROOT_STORE",
        ),
        (
            "shell/_callbacks.py",
            "_confirm_source_picker",
            "SHELL_SOURCE_IMAGE_ROOT_STORE",
        ),
        (
            "shell/_callbacks.py",
            "_source_from_sidebar_selection",
            "SHELL_SOURCE_IMAGE_ROOT_STORE",
        ),
        (
            "shell/_callbacks.py",
            "_clear_metadata_csv",
            "SHELL_METADATA_CSV_STORE",
        ),
        (
            "shell/_callbacks.py",
            "_confirm_metadata_picker",
            "SHELL_METADATA_CSV_STORE",
        ),
        (
            "run_console/_callbacks.py",
            "_action_control_outputs",
            "SHELL_METADATA_CSV_STORE",
        ),
        (
            "run_console/_callbacks.py",
            "_mirror_input_dir_to_shared_source",
            "SHELL_SOURCE_IMAGE_ROOT_STORE",
        ),
        (
            "tune/_callbacks.py",
            "_mirror_tune_image_source_to_shared",
            "SHELL_SOURCE_IMAGE_ROOT_STORE",
        ),
    }


def test_shared_refresh_revision_invalidates_labels_and_open_pickers(
    tmp_path: Path,
) -> None:
    import dash

    from phenotypic.gui.shell._ids import (
        SHELL_CLASSIFIER_CACHE_STORE,
        SHELL_METADATA_CSV_MODAL_BODY,
        SHELL_SETTINGS_METADATA_CSV_LABEL,
        SHELL_SOURCE_IMAGE_ROOT_LABEL,
        SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY,
        SHELL_TAB_HOME,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dcc.Markdown("body")

    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    expected_outputs = {
        SHELL_SOURCE_IMAGE_ROOT_LABEL,
        SHELL_SETTINGS_METADATA_CSV_LABEL,
        SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY,
        SHELL_METADATA_CSV_MODAL_BODY,
    }
    refresh_consumers: set[str] = set()
    for callback_id, meta in app.callback_map.items():
        input_ids = {item["id"] for item in meta["inputs"]}
        if SHELL_CLASSIFIER_CACHE_STORE not in input_ids:
            continue
        refresh_consumers.update(
            output_id
            for output_id in expected_outputs
            if output_id in callback_id
        )

    assert refresh_consumers == expected_outputs

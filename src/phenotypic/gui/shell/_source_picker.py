"""Shell source-image-root picker components."""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.builder._directory_browser import directory_tree
from phenotypic.gui.shell import _ids as ids
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = ["build_source_picker_modal", "render_source_picker_tree"]


def render_source_picker_tree(
    sandbox: SandboxRoot,
    current: Path | None = None,
) -> html.Div:
    """Render the folder-only source-image-root picker tree.

    Args:
        sandbox: Containment primitive; navigation cannot leave this root.
        current: Directory currently being viewed. Defaults to ``sandbox.root``.

    Returns:
        Directory tree rooted at the sandbox, with only directories selectable.
    """
    return directory_tree(
        sandbox.root,
        current=current,
        extensions=None,
        select_files=False,
        id_type=ids.SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE,
    )


def build_source_picker_modal(sandbox: SandboxRoot) -> dbc.Modal:
    """Build the top-bar shared source-image-root picker modal.

    Args:
        sandbox: Containment primitive used for the directory picker.

    Returns:
        Closed :class:`dbc.Modal` containing a sandbox-bounded directory tree.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick source image folder")),
            dbc.ModalBody(
                html.Div(
                    [
                        dcc.Store(
                            id=ids.SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE,
                            data=str(sandbox.root),
                        ),
                        html.Div(
                            id=ids.SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY,
                            children=[render_source_picker_tree(sandbox)],
                        ),
                    ]
                )
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id=ids.SHELL_SOURCE_IMAGE_ROOT_CANCEL,
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Use this folder",
                        id=ids.SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
                        color="primary",
                        n_clicks=0,
                    ),
                ]
            ),
        ],
        id=ids.SHELL_SOURCE_IMAGE_ROOT_MODAL,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )

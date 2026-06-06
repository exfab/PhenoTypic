"""Curate Image Source picker — sandbox-bounded plate-directory selection (B-IMG).

The Curate view loads each plate as a :class:`~phenotypic.GridImage` from
``<Image Source>/<plate_name>`` to render a candidate's segmentation overlay.
The run output directory holds no input images, so the user points the view at
the calibration-image directory via a **sandbox-bounded directory picker** —
the same security boundary the builder / run-console pickers enforce, so a plate
load can't escape the sandbox on a shared SSH tunnel.

This module is Tune-only (not the shell / builder). It reuses the builder's
:func:`~phenotypic.gui.builder._directory_browser.directory_tree` (folder-only
listing) inside a :class:`dbc.Modal`, mirroring the run-console input picker, and
exposes the two pure helpers the callbacks (and tests) target:

* :func:`resolve_image_source` — validate a candidate directory is in-sandbox
  and a real directory (``None`` on escape / non-directory).
* :func:`plate_image_path` — join ``<Image Source>/<plate_name>`` for the
  overlay loader.

Like the rest of :mod:`phenotypic.gui.tune`, importing this module must never
drag ``optuna`` into ``sys.modules`` (it imports only Dash + the sandbox / tree
primitives, none of which touch optuna).
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.builder._directory_browser import directory_tree
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.tune import _ids as ids

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_image_source",
    "plate_image_path",
    "render_image_source_tree",
    "build_image_source_modal",
]


def resolve_image_source(sandbox: SandboxRoot, candidate: str) -> Path | None:
    """Resolve ``candidate`` to an in-sandbox directory, or ``None``.

    The Image Source must be a real directory inside the sandbox boundary. Any
    path that escapes the sandbox (``..`` traversal or an out-of-root absolute
    path) or is not an existing directory is refused — the caller surfaces the
    refusal in a toast rather than loading plates from an unbounded location.

    Args:
        sandbox: The frozen-at-launch containment primitive.
        candidate: The path string from the picker (typed or browsed).

    Returns:
        The absolute, sandbox-contained directory path, or ``None`` when the
        candidate escapes the sandbox or is not an existing directory.
    """
    try:
        resolved = sandbox.resolve(candidate)
    except ValueError:
        logger.warning("Image Source escapes sandbox: %r", candidate)
        return None
    if not resolved.is_dir():
        logger.warning("Image Source is not a directory: %s", resolved)
        return None
    return resolved


def plate_image_path(image_source: str, plate_name: str) -> Path:
    """Join the Image Source directory and a plate name into a load path.

    The overlay loader reads ``<Image Source>/<plate_name>`` as a
    :class:`~phenotypic.GridImage`. A pure path expression — it does not touch
    disk (the caller decides whether the file exists).

    Args:
        image_source: The selected Image Source directory (absolute).
        plate_name: The plate file name (e.g. ``"plate_01.tif"``).

    Returns:
        ``Path(image_source) / plate_name``.
    """
    return Path(image_source) / plate_name


def render_image_source_tree(
    sandbox: SandboxRoot, current: Path | None = None
) -> html.Div:
    """Render the folder-only Image Source tree for ``current``.

    Reuses the builder's :func:`directory_tree` (no file selection — the user
    picks a *directory* of plate images) with a Tune-specific ``id_type`` so the
    pattern-matching navigation callback never collides with the builder /
    run-console trees on the unified hub.

    Args:
        sandbox: Containment primitive — the listing never crosses
            ``sandbox.root``.
        current: Directory currently being viewed (defaults to the sandbox root).

    Returns:
        The depth-1 directory listing as an :class:`html.Div`.
    """
    return directory_tree(
        sandbox.root,
        current=current,
        extensions=None,
        select_files=False,
        id_type=ids.TUNE_DIR_ENTRY_IMAGE_SOURCE,
    )


def build_image_source_modal(
    sandbox: SandboxRoot, *, initial_dir: Path | None = None
) -> dbc.Modal:
    """Build the sandbox-bounded Image Source picker modal.

    Folder-only tree (mirrors the run-console input picker). The selected
    directory is committed via the "Use this directory" footer button so an
    accidental click while exploring doesn't pin the wrong directory.

    Args:
        sandbox: Containment primitive used as the security boundary.
        initial_dir: The directory the tree opens on (the bound run's
            ``images_dir`` when known); defaults to the sandbox root.

    Returns:
        A :class:`dbc.Modal` with id :data:`ids.TUNE_IMAGE_SOURCE_MODAL`,
        starting closed.
    """
    start_dir = initial_dir if initial_dir is not None else sandbox.root
    body = html.Div(
        [
            dcc.Store(
                id=ids.TUNE_IMAGE_SOURCE_BROWSE_DIR,
                data=str(start_dir),
            ),
            html.Div(
                id=ids.TUNE_IMAGE_SOURCE_MODAL_BODY,
                children=[render_image_source_tree(sandbox, start_dir)],
            ),
        ]
    )
    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.TUNE_BTN_IMAGE_SOURCE_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Use this directory",
                id=ids.TUNE_BTN_IMAGE_SOURCE_CONFIRM,
                color="primary",
                n_clicks=0,
            ),
        ]
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick plate Image Source")),
            dbc.ModalBody(body),
            footer,
        ],
        id=ids.TUNE_IMAGE_SOURCE_MODAL,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )

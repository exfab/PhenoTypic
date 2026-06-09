"""Output-directory picker for the Run console.

Forked from :mod:`phenotypic.gui.builder._directory_browser` because the
output picker has different semantics:

    * Files are never selectable — the user picks a *directory* (which may
      not yet exist).
    * The text input is the source of truth: typing a non-existent path
      like ``output_<timestamp>`` is supported. The tree is a navigation
      aid, not a constraint.
    * Sandbox-relative containment is enforced via
      :meth:`SandboxRoot.resolve` rather than the builder's
      ``image-root``-based ``_is_within`` helper.

The :func:`build_output_picker_modal` factory mirrors the builder's
:class:`dbc.Modal` chrome for visual parity, but its IDs all carry the
``rc-`` prefix so the two coexist on the unified hub without collisions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.run_console import _ids as ids
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = [
    "build_output_picker_modal",
    "render_output_dir_tree",
    "ensure_output_dir",
]


def _safe_iterdir(path: Path) -> List[Path]:
    """List ``path`` defensively, returning ``[]`` on any OS error."""
    try:
        return list(path.iterdir())
    except (OSError, PermissionError):
        return []


def _entry_item(
    *,
    icon: str,
    label: str,
    kind: str,
    path: Path,
) -> dbc.ListGroupItem:
    """Build one clickable :class:`dbc.ListGroupItem` for the output tree.

    Each item carries a pattern-matching id of the form
    ``{"type": RC_DIR_ENTRY_TYPE_OUTPUT_DIR, "kind": kind, "path": str(path)}``
    so a single callback can subscribe via ``ALL`` and dispatch on ``kind``.
    """
    return dbc.ListGroupItem(
        [html.Span(f"{icon} "), html.Span(label)],
        id={
            "type": ids.RC_DIR_ENTRY_TYPE_OUTPUT_DIR,
            "kind": kind,
            "path": str(path),
        },
        action=True,
        n_clicks=0,
    )


def render_output_dir_tree(
    sandbox: SandboxRoot,
    current: Path | None = None,
) -> html.Div:
    """Render a depth-1 directory listing rooted at ``sandbox.root``.

    Files are always hidden (this is a directory picker). Hidden entries
    (names starting with ``.``) and out-of-sandbox symlinks are filtered.
    The header shows the absolute path of the directory currently being
    viewed so the user can confirm where they are about to write output.

    Args:
        sandbox: Containment primitive — the listing never crosses
            ``sandbox.root``.
        current: Directory currently being viewed. Defaults to
            ``sandbox.root`` when ``None``. Silently falls back to the
            sandbox root if ``current`` is outside the sandbox.

    Returns:
        :class:`dash.html.Div` containing a path header and a
        :class:`dbc.ListGroup` of subdirectories. Each item carries a
        pattern-matching id with type :data:`ids.RC_DIR_ENTRY_TYPE_OUTPUT_DIR`.
    """
    here = current if current is not None else sandbox.root
    if not sandbox.contains(here):
        here = sandbox.root

    children: List = [
        html.Div(
            str(here.resolve() if here.exists() else here),
            className="small text-muted mb-2",
            style={"wordBreak": "break-all", "fontFamily": "var(--font-mono)"},
        ),
    ]

    list_items: List = []

    # Parent entry — only when current is a real subdirectory of root.
    try:
        here_resolved = here.resolve(strict=False)
        root_resolved = sandbox.root.resolve(strict=False)
    except OSError:
        here_resolved = here
        root_resolved = sandbox.root
    if here_resolved != root_resolved:
        parent = here.parent
        if sandbox.contains(parent):
            list_items.append(
                _entry_item(
                    icon="↑",
                    label="Parent directory",
                    kind="parent",
                    path=parent,
                )
            )

    subdirs: List[Path] = []
    for entry in _safe_iterdir(here):
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        if entry.is_symlink() and not sandbox.contains(entry):
            continue
        subdirs.append(entry)

    subdirs.sort(key=lambda p: p.name.lower())
    for d in subdirs:
        list_items.append(
            _entry_item(
                icon="\U0001F4C1",
                label=f"{d.name}/",
                kind="dir",
                path=d,
            )
        )

    if list_items:
        children.append(dbc.ListGroup(list_items, flush=True))
    else:
        children.append(
            html.Div(
                "(no subdirectories)",
                className="text-muted small fst-italic",
            )
        )

    return html.Div(children)


def build_output_picker_modal(sandbox: SandboxRoot) -> dbc.Modal:
    """Build the output-directory picker modal.

    Three sections in the body:

        1. **Free-form path input** — the source of truth. The user can
           type a not-yet-existing path (e.g. ``output_2026_04_30``) and
           confirm; :func:`ensure_output_dir` will create it.
        2. **Tree** — depth-1 listing of subdirectories of the currently
           browsed directory. Clicking a folder navigates the tree; the
           selected folder also gets pasted into the path input on
           confirmation.
        3. **Cancel / Confirm buttons** in the footer.

    Args:
        sandbox: Containment primitive used as the security boundary for
            the tree.

    Returns:
        A :class:`dbc.Modal` with id :data:`ids.RC_MODAL_OUTPUT`,
        ``is_open=False``, ``size="lg"``, ``scrollable=True``,
        ``backdrop=True``, and ``centered=True``.
    """
    body = html.Div(
        [
            # Sibling store so re-renders of the body don't replace it.
            dcc.Store(
                id=ids.RC_STORE_BROWSE_DIR_OUTPUT,
                data=str(sandbox.root),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id=ids.RC_INPUT_OUTPUT_PATH,
                            type="text",
                            placeholder=(
                                "absolute path or sandbox-relative"
                                " (may not yet exist)"
                            ),
                            debounce=True,
                        ),
                        width=12,
                    ),
                ],
                className="g-2 mb-2",
            ),
            html.Div(
                id=ids.RC_MODAL_OUTPUT_BODY,
                children=[render_output_dir_tree(sandbox)],
            ),
        ]
    )

    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.RC_BTN_OUTPUT_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Use this directory",
                id=ids.RC_BTN_OUTPUT_CONFIRM,
                color="primary",
                n_clicks=0,
            ),
        ]
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick output directory")),
            dbc.ModalBody(body),
            footer,
        ],
        id=ids.RC_MODAL_OUTPUT,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


def ensure_output_dir(
    sandbox: SandboxRoot,
    candidate: str,
) -> Optional[Path]:
    """Resolve ``candidate`` to an in-sandbox path and ``mkdir -p`` it.

    Used by the confirm callback. Falls through to the sandbox's
    ``resolve(strict=False)`` so non-existent path components are
    accepted (this is exactly the case where the user typed
    ``output_<timestamp>``).

    Args:
        sandbox: Containment primitive.
        candidate: Path string from :data:`ids.RC_INPUT_OUTPUT_PATH`.

    Returns:
        The absolute, sandbox-contained, freshly-created path on success;
        ``None`` if the path escapes the sandbox or cannot be created.
    """
    try:
        resolved = sandbox.resolve(candidate)
    except ValueError:
        logger.warning("output dir escapes sandbox: %r", candidate)
        return None
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("could not mkdir %s: %s", resolved, exc)
        return None
    return resolved

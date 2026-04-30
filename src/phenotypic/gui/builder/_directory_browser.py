"""Server-side directory tree component for picking test images on the HPCC.

This module provides the layout-only Dash components used by the pipeline
builder's image-source picker. It walks one directory level at a time so that
remote filesystems (e.g. HPCC scratch) stay responsive, and never traverses
through symlinks that escape the configured root.

Callbacks are wired up in ``_callbacks.py``; nothing here registers callbacks
or imports the builder state model.

Component IDs are exposed as module-level constants so that the callback
module has a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Image file extensions surfaced by the directory tree (case-insensitive).
IMAGE_EXTS = {
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".raw",
    ".nef",
    ".cr2",
    ".arw",
    ".dng",
}

#: Sentinel path injected into the path input when the user clicks
#: "Use synthetic plate" and the bundled synthetic plate cannot be located on
#: disk. Callbacks should detect this and fall back to
#: ``phenotypic.data.load_synth_yeast_plate()`` rather than treating it as a
#: real path.
SYNTHETIC_SENTINEL = "<synthetic>"

# Component IDs (single source of truth for callbacks in Phase 3).
DIR_PICKER_PATH_INPUT = "dir-picker-path"
DIR_PICKER_USE_PATH_BTN = "dir-picker-use-path"
DIR_PICKER_SYNTH_BTN = "dir-picker-synth"
DIR_PICKER_TREE_CONTAINER = "dir-picker-tree"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_synthetic_plate_path() -> Path:
    """Locate the on-disk path of the bundled synthetic yeast plate image.

    The builder needs a concrete file path to populate the path input when the
    user clicks "Use synthetic plate". We resolve the same RGB PNG that
    :func:`phenotypic.data.load_synth_yeast_plate` reads.

    Returns:
        The absolute path of ``synthetic_test_plate/yeast_plate_rgb.png`` if
        it exists on disk, otherwise the sentinel
        :data:`SYNTHETIC_SENTINEL` (wrapped in :class:`pathlib.Path`). The
        sentinel is returned as a ``Path`` so the return type stays
        consistent; callers should check via ``str(path) ==
        SYNTHETIC_SENTINEL`` before treating the value as a filesystem path.

    Notes:
        We import ``phenotypic.data._sample_image_data`` lazily to avoid
        pulling the whole image-IO stack at module import time (this layout
        module is imported during Dash app construction).
    """
    try:
        from phenotypic.data import _sample_image_data
    except Exception:
        return Path(SYNTHETIC_SENTINEL)

    candidate = (
        Path(_sample_image_data.__current_file_dir)
        / "synthetic_test_plate"
        / "yeast_plate_rgb.png"
    )
    if candidate.is_file():
        return candidate.resolve()
    return Path(SYNTHETIC_SENTINEL)


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves under ``root`` (inclusive).

    Used to keep symlink traversal contained to the configured root so a
    rogue symlink in the browsed tree cannot expose the wider filesystem.
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    if resolved == root_resolved:
        return True
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _safe_iterdir(path: Path) -> List[Path]:
    """Iterate ``path`` defensively, returning ``[]`` on any OS error.

    Permission errors, broken symlinks, and missing directories all collapse
    to an empty list so the UI can still render an empty list-group instead
    of crashing the callback.
    """
    try:
        return list(path.iterdir())
    except (OSError, PermissionError):
        return []


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

def directory_tree(root: Path, current: Path | None = None) -> html.Div:
    """Render a depth-1 directory tree rooted at ``root``.

    The tree shows the absolute path of ``current or root`` as a header,
    optionally followed by a "↑ Parent directory" entry (when navigation
    upward stays within ``root``), then all immediate subdirectories
    (alphabetical), then all immediate image files (alphabetical) whose
    extension is in :data:`IMAGE_EXTS` (case-insensitive).

    Hidden entries (names starting with ``.``) are skipped. Symlinks whose
    target resolves outside ``root`` are skipped to prevent the user from
    navigating off the configured root.

    Args:
        root: Configured image root. Used as both the security boundary and
            the fallback location when ``current`` is not supplied.
        current: Directory currently being viewed. If ``None``, defaults to
            ``root``. If outside ``root``, falls back to ``root``.

    Returns:
        A :class:`dash.html.Div` containing the header and the listing as a
        :class:`dash_bootstrap_components.ListGroup`. Each list item carries a
        pattern-matching id of the form
        ``{"type": "dir-entry", "kind": "dir" | "file" | "parent", "path":
        str(path)}`` so the callbacks module can wire one handler for the
        whole tree.
    """
    here = current if current is not None else root
    if not _is_within(here, root):
        here = root

    children: List = [
        html.Div(
            str(here.resolve() if here.exists() else here),
            className="text-monospace small text-muted mb-2",
            style={"wordBreak": "break-all"},
        ),
    ]

    list_items: List = []

    # Parent entry (only when current is set, distinct from root, and stays
    # within the configured root after going up one level).
    if current is not None:
        try:
            current_resolved = here.resolve()
            root_resolved = root.resolve()
        except OSError:
            current_resolved = here
            root_resolved = root
        if current_resolved != root_resolved:
            parent = here.parent
            if _is_within(parent, root):
                list_items.append(
                    dbc.ListGroupItem(
                        [html.Span("↑ "), html.Span("Parent directory")],
                        id={
                            "type": "dir-entry",
                            "kind": "parent",
                            "path": str(parent),
                        },
                        action=True,
                        n_clicks=0,
                    )
                )

    # Walk depth-1 of `here`, filtering hidden entries and out-of-root symlinks.
    entries = _safe_iterdir(here)

    subdirs: List[Path] = []
    files: List[Path] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if entry.is_symlink() and not _is_within(entry, root):
            # Don't expose anything beyond the configured root via symlinks.
            continue
        if is_dir:
            subdirs.append(entry)
        else:
            if entry.suffix.lower() in IMAGE_EXTS:
                files.append(entry)

    subdirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())

    for d in subdirs:
        list_items.append(
            dbc.ListGroupItem(
                [html.Span("\U0001F4C1 "), html.Span(d.name + "/")],
                id={
                    "type": "dir-entry",
                    "kind": "dir",
                    "path": str(d),
                },
                action=True,
                n_clicks=0,
            )
        )
    for f in files:
        list_items.append(
            dbc.ListGroupItem(
                [html.Span("\U0001F5BC️ "), html.Span(f.name)],
                id={
                    "type": "dir-entry",
                    "kind": "file",
                    "path": str(f),
                },
                action=True,
                n_clicks=0,
            )
        )

    if list_items:
        children.append(dbc.ListGroup(list_items, flush=True))
    else:
        children.append(
            html.Div(
                "(no subdirectories or images)",
                className="text-muted small fst-italic",
            )
        )

    return html.Div(children)


def directory_picker(
    root: Path | None,
    current: Path | None = None,
    *,
    component_id: str = "dir-picker",
) -> html.Div:
    """Build the full image-source picker.

    Sections rendered top to bottom:

    1. **Path input row** — a free-form ``dcc.Input`` for typing or pasting
       an absolute path, alongside a "Use this path" button that the
       callback layer wires to commit the input value to the builder state.
    2. **Synthetic-plate button** — clicking populates the path input with
       the resolved disk path of the bundled synthetic yeast plate. If the
       file cannot be located, the input is populated with the
       :data:`SYNTHETIC_SENTINEL` string so the callback can fall back to
       :func:`phenotypic.data.load_synth_yeast_plate` programmatically.
    3. **Tree section** — only rendered when ``root`` is provided. Shows the
       depth-1 :func:`directory_tree` for ``current or root``.

    Args:
        root: Configured image-root directory (from the CLI's
            ``--image-root``). When ``None``, the directory tree section is
            omitted; the user can still type a path manually or pick the
            synthetic plate.
        current: Directory currently being viewed inside the tree. Ignored
            when ``root`` is ``None``.
        component_id: Reserved for future use; not currently consumed by
            this layout because the inner widgets use the module-level
            ``DIR_PICKER_*`` constants for their ids. Kept in the signature
            so callers can pass a namespace if multiple pickers ever need to
            coexist on the same page.

    Returns:
        A :class:`dash.html.Div` containing the three sections described
        above.
    """
    del component_id  # currently unused; reserved for future namespacing

    sections: List = []

    # 1. Path input row.
    path_row = dbc.Row(
        [
            dbc.Col(
                dcc.Input(
                    id=DIR_PICKER_PATH_INPUT,
                    type="text",
                    placeholder="/abs/path/to/image.tif",
                    debounce=True,
                    style={"width": "100%"},
                ),
                width=9,
            ),
            dbc.Col(
                dbc.Button(
                    "Use this path",
                    id=DIR_PICKER_USE_PATH_BTN,
                    color="primary",
                    n_clicks=0,
                ),
                width=3,
            ),
        ],
        className="g-2 mb-2",
    )
    sections.append(path_row)

    # 2. Synthetic-plate button.
    synth_button = dbc.Button(
        "Use synthetic plate",
        id=DIR_PICKER_SYNTH_BTN,
        color="secondary",
        outline=True,
        n_clicks=0,
        className="mb-2",
    )
    sections.append(synth_button)

    # 3. Tree section (only when a root is configured).
    if root is not None:
        tree: Optional[html.Div] = directory_tree(root, current)
        sections.append(
            html.Div(
                tree,
                id=DIR_PICKER_TREE_CONTAINER,
                className="border rounded p-2",
            )
        )
    else:
        sections.append(
            html.Div(
                id=DIR_PICKER_TREE_CONTAINER,
                children=html.Div(
                    "(no --image-root configured; type a path above"
                    " or use the synthetic plate)",
                    className="text-muted small fst-italic",
                ),
                className="border rounded p-2",
            )
        )

    return html.Div(sections)

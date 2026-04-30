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
from typing import FrozenSet, List, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Image file extensions surfaced by the directory tree (case-insensitive).
IMAGE_EXTS: FrozenSet[str] = frozenset(
    {
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
)

#: Pipeline-config file extensions surfaced when the modal is loading a saved
#: pipeline. The dash builder writes ``ImagePipeline.to_json`` outputs, so only
#: ``.json`` is currently recognised — YAML support is intentionally out of
#: scope.
PIPELINE_EXTS: FrozenSet[str] = frozenset({".json"})

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


def _entry_item(
    *,
    icon: str,
    label: str,
    id_type: str,
    kind: str,
    path: Path,
) -> dbc.ListGroupItem:
    """Build one clickable :class:`dbc.ListGroupItem` for the directory tree.

    Args:
        icon: Leading single-character glyph (folder, picture, up-arrow).
        label: Text shown after the icon (filename or directory name).
        id_type: ``DIR_ENTRY_TYPE_*`` value placed in the pattern-matching id.
        kind: One of ``"parent"``, ``"dir"``, ``"file"``.
        path: Filesystem path serialised into the id so callbacks can act on
            it without needing extra state.
    """
    return dbc.ListGroupItem(
        [html.Span(f"{icon} "), html.Span(label)],
        id={"type": id_type, "kind": kind, "path": str(path)},
        action=True,
        n_clicks=0,
    )


def directory_tree(
    root: Path,
    current: Path | None = None,
    *,
    extensions: FrozenSet[str] | None = IMAGE_EXTS,
    select_files: bool = True,
    id_type: str = "dir-entry",
) -> html.Div:
    """Render a depth-1 directory listing rooted at ``root``.

    Builds a :class:`dbc.ListGroup` showing the immediate children of
    ``current`` (or ``root`` when ``current`` is ``None``): an optional
    "↑ Parent directory" entry, then subdirectories alphabetically, then
    selectable files whose extension matches ``extensions``. Hidden entries
    (names starting with ``.``) and symlinks that resolve outside ``root``
    are excluded.

    By default the tree surfaces :data:`IMAGE_EXTS`, which includes ``.tif``
    and raw formats (``.nef``, ``.cr2``, ``.arw``, ``.dng``) commonly
    produced by DSLR cameras used to photograph agar plates. Pass
    :data:`PIPELINE_EXTS` to filter for saved ``ImagePipeline.to_json()``
    files, or ``None`` to show all files regardless of extension.

    Args:
        root: Configured working directory. Acts as both the security boundary
            (navigation cannot leave this directory, even via symlinks) and the
            starting location when ``current`` is ``None``.
        current: Directory currently being viewed. Defaults to ``root`` when
            ``None``. Silently falls back to ``root`` if ``current`` resolves
            outside ``root``.
        extensions: Allowed file extensions (lowercase, with leading dot).
            Defaults to :data:`IMAGE_EXTS` so the image picker shows plate
            image files suitable for loading with :class:`phenotypic.Image` or
            :class:`phenotypic.GridImage`. Pass :data:`PIPELINE_EXTS` for the
            JSON pipeline browser, or ``None`` to surface all files. Ignored
            when ``select_files`` is ``False``.
        select_files: When ``True`` (default), matching files appear as
            clickable list items below the subdirectory entries. When
            ``False``, only directories are shown — used by the Save modal
            where the user picks a target folder and types the filename
            separately.
        id_type: Value placed in the ``"type"`` slot of each item's
            pattern-matching id. Each modal passes a distinct value (e.g.
            :data:`ids.DIR_ENTRY_TYPE_IMAGE`, :data:`ids.DIR_ENTRY_TYPE_JSON`,
            :data:`ids.DIR_ENTRY_TYPE_SAVE`) so its callback can subscribe
            without conflicting with trees rendered by other modals on the
            same page.

    Returns:
        A :class:`dash.html.Div` containing a path header and the listing as
        a :class:`dbc.ListGroup`. Each item carries a pattern-matching id of
        the form ``{"type": id_type, "kind": "dir" | "file" | "parent",
        "path": str(path)}`` so the callback layer can wire a single handler
        for the whole tree via ``Input({"type": id_type, "kind": ALL, "path":
        ALL}, "n_clicks")``.
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
                    _entry_item(
                        icon="↑",
                        label="Parent directory",
                        id_type=id_type,
                        kind="parent",
                        path=parent,
                    )
                )

    # Walk depth-1 of `here`, filtering hidden entries and out-of-root symlinks.
    subdirs: List[Path] = []
    files: List[Path] = []
    for entry in _safe_iterdir(here):
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
        elif select_files and (
            extensions is None or entry.suffix.lower() in extensions
        ):
            files.append(entry)

    subdirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())

    for d in subdirs:
        list_items.append(
            _entry_item(
                icon="\U0001F4C1",
                label=f"{d.name}/",
                id_type=id_type,
                kind="dir",
                path=d,
            )
        )
    for f in files:
        list_items.append(
            _entry_item(
                icon="\U0001F5BC️",
                label=f.name,
                id_type=id_type,
                kind="file",
                path=f,
            )
        )

    if list_items:
        children.append(dbc.ListGroup(list_items, flush=True))
    else:
        empty_msg = (
            "(no subdirectories)"
            if not select_files
            else "(no subdirectories or matching files)"
        )
        children.append(
            html.Div(
                empty_msg,
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

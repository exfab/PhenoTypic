"""List image files under a source root, grouped by relative subfolder.

Produces the ``{dataset_rel: [filename, ...]}`` map that drives the Browse
tab's two cascading dropdowns. ``dataset_rel`` is the image's parent
directory relative to the source root (``"."`` for files directly under
the root), so arbitrary nesting collapses to one flat set of dataset keys.
"""
from __future__ import annotations

import logging
from pathlib import Path

from phenotypic.gui._config import IMAGE_EXTS

logger = logging.getLogger(__name__)

__all__ = ["list_datasets"]


def list_datasets(source_root: Path) -> dict[str, list[str]]:
    """Return an ordered ``{dataset_rel: [filename, ...]}`` map.

    Hidden files/dirs (leading ``.``) and symlinks whose target escapes the
    source root are skipped. Keys and filename lists are sorted.
    """
    source_root = Path(source_root)
    try:
        root_resolved = source_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return {}
    out: dict[str, list[str]] = {}
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            rel = path.relative_to(source_root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue  # hidden dotfile / dot-dir (e.g. .phenotypic cache)
        try:
            path.resolve(strict=False).relative_to(root_resolved)
        except ValueError:
            continue  # symlink escaping the source root
        out.setdefault(rel.parent.as_posix(), []).append(path.name)
    for files in out.values():
        files.sort()
    return dict(sorted(out.items()))

"""List image files under a source root, grouped by relative subfolder.

Produces the ``{dataset_rel: [filename, ...]}`` map that drives the Browse
tab's two cascading dropdowns. ``dataset_rel`` is the image's parent
directory relative to the source root (``"."`` for files directly under
the root), so arbitrary nesting collapses to one flat set of dataset keys.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from phenotypic.gui._config import IMAGE_EXTS
from phenotypic.gui.browse._source_item import is_source_store

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
    if is_source_store(source_root):
        return {".": [source_root.name]}

    out: dict[str, list[str]] = {}
    try:
        walk = os.walk(source_root, topdown=True, followlinks=False)
        for current, directory_names, filenames in walk:
            current_path = Path(current)
            relative_parent = current_path.relative_to(source_root)
            if any(part.startswith(".") for part in relative_parent.parts):
                directory_names[:] = []
                continue

            retained_directories: list[str] = []
            for directory_name in directory_names:
                directory = current_path / directory_name
                if directory_name.startswith(".") or directory.is_symlink():
                    continue
                if is_source_store(directory):
                    out.setdefault(relative_parent.as_posix(), []).append(
                        directory_name
                    )
                    continue
                retained_directories.append(directory_name)
            directory_names[:] = retained_directories

            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = current_path / filename
                if path.suffix.lower() not in IMAGE_EXTS or path.is_symlink():
                    continue
                try:
                    path.resolve(strict=True).relative_to(root_resolved)
                except (OSError, RuntimeError, ValueError):
                    continue
                if path.is_file():
                    out.setdefault(relative_parent.as_posix(), []).append(
                        filename
                    )
    except (OSError, RuntimeError, ValueError):
        return {}
    for files in out.values():
        files.sort()
    return dict(sorted(out.items()))

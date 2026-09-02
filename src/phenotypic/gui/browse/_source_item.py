"""Canonical resolution of one Browse source item."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from phenotypic.sdk_ import is_zarr_store_name

__all__ = [
    "is_source_store",
    "resolve_source_item",
    "source_item_relative_path",
]


def is_source_store(path: Path) -> bool:
    """Return whether ``path`` is an exact, non-symlink OME-Zarr directory."""
    candidate = Path(path)
    return (
        is_zarr_store_name(candidate)
        and not candidate.is_symlink()
        and candidate.is_dir()
    )


def resolve_source_item(
    source_root: Path,
    dataset_rel: str,
    filename: str,
) -> Path:
    """Resolve one listed Browse item below a container or direct-store root.

    A directly selected store is represented as the root dataset containing
    its own name. This helper collapses that UI representation back to the
    store itself instead of constructing ``<store>/<store>``.
    """
    root = Path(source_root)
    dataset = PurePosixPath(dataset_rel)
    name = PurePosixPath(filename)
    if (
        not filename
        or name.is_absolute()
        or len(name.parts) != 1
        or name.parts[0] in {".", ".."}
        or "\\" in filename
        or dataset.is_absolute()
        or ".." in dataset.parts
        or "\\" in dataset_rel
    ):
        raise ValueError("invalid Browse source identity")

    if is_source_store(root):
        if dataset_rel != "." or filename != root.name:
            raise ValueError("item does not identify the direct store root")
        return root

    relative = name if dataset_rel == "." else dataset / name
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Browse source item escapes its root") from exc
    if candidate.is_symlink():
        raise ValueError("Browse source items cannot be symlinks")
    return candidate


def source_item_relative_path(
    source_root_rel: str,
    dataset_rel: str,
    filename: str,
) -> str:
    """Return one Browse item path relative to the GUI sandbox."""
    root = PurePosixPath(source_root_rel)
    if (
        is_zarr_store_name(root)
        and dataset_rel == "."
        and filename == root.name
    ):
        return root.as_posix()
    parts = [part for part in (source_root_rel, dataset_rel) if part != "."]
    return PurePosixPath(*parts, filename).as_posix()

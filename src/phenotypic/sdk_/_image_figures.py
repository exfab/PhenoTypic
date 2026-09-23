"""Per-image figures inside an OME-Zarr store (spec 2026-09-22 §1).

Storage-neutral: the caller decides directory names, filenames, formats and
media types; this module writes bytes where it is told, hashes them, and
describes them. It never imports a plotting library.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._atomic_io import atomic_write_json

#: Same document every `tables/*` group uses.
_GROUP_DOCUMENT: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {},
}


@dataclass(frozen=True)
class StoredFigureFile:
    """One rendering of one page. ``filename`` is final; no separators."""

    format: str
    media_type: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class StoredFigurePage:
    """One page and the renderings that succeeded for it.

    ``metadata`` is strict, key-sorted JSON by the time it gets here: the
    builder round-trips it and refuses a page it cannot (spec §1).
    """

    key: str
    label: str | None
    backend: str
    metadata: Mapping[str, Any]
    files: tuple[StoredFigureFile, ...]


@dataclass(frozen=True)
class StoredFigureBinding:
    """One binding's pages. ``directory`` is the sanitized group name."""

    binding_id: str
    plot_class: str
    directory: str
    pages: tuple[StoredFigurePage, ...]


@dataclass(frozen=True)
class StoredFigureFailure:
    """A failure at the finest level available (spec §1)."""

    binding: str
    page: str | None
    format: str | None
    error: str


@dataclass(frozen=True)
class StoredFigures:
    """Everything one image's store will say about its figures, in order."""

    bindings: tuple[StoredFigureBinding, ...]
    failed: tuple[StoredFigureFailure, ...]


def write_image_figures(
    store_part: Path, figures: StoredFigures
) -> dict[str, object]:
    """Write ``figures/`` into an unpromoted part and return its descriptor.

    Args:
        store_part: An unpromoted ``*.ome.zarr.part`` directory in which
            ``figures/`` does not yet exist. Callers rewriting a promoted store
            remove the part's copied ``figures/`` first: those copies are hard
            links into the live store, and writing through one would change
            the published bytes (``replace_image_tables``).
        figures: The built figures.

    Returns:
        ``{"figures": descriptor}``, to apply with
        :func:`apply_image_figures_attributes`.
    """
    from . import ngff_

    group = Path(store_part) / ngff_.FIGURES_GROUP
    group.mkdir(parents=True, exist_ok=True)
    atomic_write_json(group / ngff_.STORE_ROOT_JSON, _GROUP_DOCUMENT)
    bindings: dict[str, object] = {}
    for binding in figures.bindings:
        directory = group / binding.directory
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / ngff_.STORE_ROOT_JSON, _GROUP_DOCUMENT)
        pages = []
        for page in binding.pages:
            entries = []
            for stored in page.files:
                target = ngff_.long_path(directory / stored.filename)
                Path(target).write_bytes(stored.data)
                entries.append({
                    "format": stored.format,
                    "media_type": stored.media_type,
                    "path": f"{ngff_.FIGURES_GROUP}/{binding.directory}/{stored.filename}",
                    "sha256": hashlib.sha256(stored.data).hexdigest(),
                })
            pages.append({
                "key": page.key,
                "label": page.label,
                "backend": page.backend,
                "metadata": dict(page.metadata),
                "files": entries,
            })
        bindings[binding.binding_id] = {"class": binding.plot_class, "pages": pages}
    descriptor = {
        "schema_version": ngff_.FIGURES_SCHEMA_VERSION,
        "bindings": bindings,
        "failed": [
            {"binding": f.binding, "page": f.page, "format": f.format, "error": f.error}
            for f in figures.failed
        ],
    }
    return {ngff_.PhenotypicAttr.FIGURES: descriptor}


def apply_image_figures_attributes(
    phenotypic: dict[str, object], fragment: dict[str, object] | None
) -> None:
    """Make the root's ``figures`` key equal *fragment*, removal included.

    ``None`` removes the key: a pipeline with no ``PlotImage`` binding has no
    figures, and a measure-mode rebuild must drop a stale descriptor -- the
    same total-function rule as ``apply_image_tables_attributes``.
    """
    from . import ngff_

    if fragment is None:
        phenotypic.pop(ngff_.PhenotypicAttr.FIGURES, None)
    else:
        phenotypic[ngff_.PhenotypicAttr.FIGURES] = fragment[ngff_.PhenotypicAttr.FIGURES]


def read_image_figures_descriptor(store_path: Path) -> dict[str, Any] | None:
    """Return a store's figures descriptor, or ``None`` when it has none.

    A root with no ``phenotypic`` block -- a third-party OME-Zarr store --
    has none, like a PhenoTypic store written without figures.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        json.JSONDecodeError: If the root is present but unparseable.
    """
    from . import ngff_

    phenotypic = ngff_.read_root_attributes(Path(store_path)).get(
        ngff_.PhenotypicAttr.ROOT
    )
    if not isinstance(phenotypic, dict):
        return None
    descriptor = phenotypic.get(ngff_.PhenotypicAttr.FIGURES)
    return descriptor if isinstance(descriptor, dict) else None


__all__ = [
    "StoredFigureBinding",
    "StoredFigureFailure",
    "StoredFigureFile",
    "StoredFigurePage",
    "StoredFigures",
    "apply_image_figures_attributes",
    "read_image_figures_descriptor",
    "write_image_figures",
]

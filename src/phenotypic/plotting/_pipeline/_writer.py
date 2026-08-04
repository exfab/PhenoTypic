"""Safe, best-effort publication for one or many plot pages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any

from phenotypic.sdk_._file_locking import exclusive_path_lock

from phenotypic.abc_.plotting import PlotOutput

from ._adapter import FigureAdapter
from ._output import normalize_plot_output

logger = logging.getLogger(__name__)

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class PlotPublicationBlocked(RuntimeError):
    """Raised when a late publication predicate rejects a plot write."""


def safe_path_component(value: str) -> str:
    """Return one filesystem-safe path component.

    Args:
        value: Human-readable identifier or label.

    Returns:
        Sanitized non-empty component.

    Raises:
        ValueError: If the value is traversal-like or sanitizes to nothing.
    """
    if not isinstance(value, str):
        raise TypeError(f"path component must be str, got {type(value).__name__}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe path component {value!r}")
    component = _UNSAFE_COMPONENT.sub("-", value.strip()).strip("-.")
    if not component or component in {".", ".."}:
        raise ValueError(f"path component {value!r} has no safe filename characters")
    return component


def publish_plot_output(
    value: Any | PlotOutput,
    directory: Path,
    *,
    plot_id: str,
    plot_class: str | None = None,
    publication_guard: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish successful pages and an authoritative per-plot manifest.

    A page failure is logged and omitted while sibling pages continue. The
    manifest is replaced last and therefore lists only durable page files.

    Args:
        value: Raw supported figure or normalized multi-page output.
        directory: Destination directory for page PNGs and the manifest.
        plot_id: Stable binding ID used in diagnostics.
        plot_class: Producer class name. Defaults to ``plot_id`` for direct
            writer calls.
        publication_guard: Optional GUI compare-and-set predicate rechecked
            immediately before directory creation and every canonical page
            or manifest replacement. CLI callers omit it.

    Returns:
        JSON-native manifest payload.
    """
    _require_plot_publication(publication_guard)
    output = normalize_plot_output(value)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        return _publish_plot_output_locked(
            output,
            directory,
            plot_id=plot_id,
            plot_class=plot_class,
            publication_guard=publication_guard,
        )


def _publish_plot_output_locked(
    output: PlotOutput,
    directory: Path,
    *,
    plot_id: str,
    plot_class: str | None,
    publication_guard: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Publish one plot generation while its directory lock is held."""
    used: dict[str, str] = {}
    pages: list[dict[str, Any]] = []
    for page in output.pages:
        label = page.label or page.key
        try:
            stem = safe_path_component(label)
        except Exception:
            stem = "page"
        base_stem = stem
        folded = stem.casefold()
        attempt = 0
        while folded in used and used[folded] != page.key:
            digest_input = (
                page.key if attempt == 0 else f"{page.key}:{attempt}"
            )
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8]
            stem = f"{base_stem}-{digest}"
            folded = stem.casefold()
            attempt += 1
        used[folded] = page.key

        filename = f"{stem}.png"
        destination = directory / filename
        temporary = directory / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            backend = FigureAdapter.backend_name(page.figure)
            FigureAdapter.save_png(page.figure, temporary)
            _require_plot_publication(publication_guard)
            os.replace(temporary, destination)
        except PlotPublicationBlocked:
            temporary.unlink(missing_ok=True)
            FigureAdapter.close(page.figure)
            raise
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            temporary.unlink(missing_ok=True)
            FigureAdapter.close(page.figure)
            logger.warning(
                "Plot %s page %s failed during save: %s",
                plot_id,
                page.key,
                exc,
            )
            continue
        pages.append(
            {
                "key": page.key,
                "label": page.label,
                "file": filename,
                "backend": backend,
                "metadata": dict(page.metadata),
            }
        )

    manifest = {
        "schema_version": 1,
        "plot_id": plot_id,
        "class": plot_class or plot_id,
        "pages": pages,
    }
    manifest_path = directory / "manifest.json"
    temporary_manifest = directory / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _require_plot_publication(publication_guard)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def _require_plot_publication(
    publication_guard: Callable[[], bool] | None,
) -> None:
    """Fail closed immediately before a canonical plot mutation."""
    if publication_guard is not None and not publication_guard():
        raise PlotPublicationBlocked(
            "Plot publication blocked because its output snapshot changed."
        )


__all__ = [
    "PlotPublicationBlocked",
    "publish_plot_output",
    "safe_path_component",
]

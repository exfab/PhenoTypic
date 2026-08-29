"""Cheap, revision-aware source-image metadata probing for Browse."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import exifread
from PIL import Image as PILImage

from phenotypic.gui._config import BROWSE_RENDER_SCHEMA_VERSION, RAW_IMAGE_EXTS
from phenotypic.sdk_ import (
    is_zarr_store_name,
    source_image_suffix,
    store_revision_identity,
)

logger = logging.getLogger(__name__)

__all__ = ["SourceRevision", "SourceProbeError", "probe_source"]


class SourceProbeError(OSError):
    """Raised when a stable source revision cannot be inspected."""


@dataclass(frozen=True)
class SourceRevision:
    """Immutable identity and header metadata for one source-file revision.

    ``cache_key`` intentionally includes the sandbox identity and rendering
    parameters. The source path itself is excluded from representations so it
    cannot accidentally enter logs.
    """

    source_path: Path = field(repr=False, compare=False)
    sandbox_identity: str
    relative_path: str
    size_bytes: int
    mtime_ns: int
    ctime_ns: int | None
    store_revision: str | None
    width: int | None
    height: int | None
    exif: Mapping[str, str] = field(compare=False, hash=False)
    tile_size: int = 254
    overlap: int = 1
    render_schema: int = BROWSE_RENDER_SCHEMA_VERSION

    @property
    def source_id(self) -> str:
        """Stable source identity independent of the current file revision."""
        return _digest((self.sandbox_identity, self.relative_path))

    @property
    def cache_key(self) -> str:
        """Fixed-length key for cache paths and revision-addressed URLs."""
        return _digest(
            (
                self.sandbox_identity,
                self.relative_path,
                self.size_bytes,
                self.mtime_ns,
                self.ctime_ns,
                self.store_revision,
                self.render_schema,
                self.tile_size,
                self.overlap,
            )
        )

    def matches_disk(self) -> bool:
        """Return whether size and timestamps still identify this revision."""
        if self.store_revision is not None:
            try:
                return (
                    store_revision_identity(self.source_path)
                    == self.store_revision
                )
            except (OSError, ValueError):
                return False
        try:
            stat = self.source_path.stat()
        except OSError:
            return False
        return (
            stat.st_size == self.size_bytes
            and stat.st_mtime_ns == self.mtime_ns
            and getattr(stat, "st_ctime_ns", None) == self.ctime_ns
        )


def probe_source(
    source: Path,
    *,
    sandbox_root: Path | None = None,
    relative_path: str | None = None,
    tile_size: int = 254,
    overlap: int = 1,
) -> SourceRevision:
    """Read source identity, dimensions, and EXIF without decoding pixels.

    Args:
        source: Existing source image.
        sandbox_root: Security-boundary root used to namespace persistent
            cache entries. Defaults to the source parent for metadata-only
            callers.
        relative_path: POSIX path below ``sandbox_root``. Derived when omitted.
        tile_size: DZI tile edge included in the cache identity.
        overlap: DZI tile overlap included in the cache identity.

    Returns:
        A stable :class:`SourceRevision` captured around header inspection.

    Raises:
        SourceProbeError: If the file is missing, outside the sandbox, or
            changes while its headers are being inspected.
    """
    started = time.perf_counter()
    source = Path(source).resolve(strict=False)
    root = Path(sandbox_root or source.parent).resolve(strict=False)
    try:
        derived_rel = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceProbeError("source is outside the sandbox") from exc
    rel = relative_path or derived_rel
    try:
        before = source.stat()
        before_store_revision = (
            store_revision_identity(source)
            if is_zarr_store_name(source)
            else None
        )
    except OSError as exc:
        raise SourceProbeError("source cannot be inspected") from exc

    width, height = _header_dimensions(source)
    imported = {} if before_store_revision is not None else _read_exif(source)

    try:
        after = source.stat()
        # ``store_revision_identity`` already captures a stable generation:
        # O(1) from PhenoTypic's root-last publication token, or a guarded
        # recursive fallback for third-party mutable stores. Wrapping it in a
        # second call doubles the GPFS traversal without strengthening the
        # point-in-time guarantee.
        after_store_revision = before_store_revision
    except OSError as exc:
        raise SourceProbeError("source disappeared during inspection") from exc
    before_identity = (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ctime_ns", None),
    )
    after_identity = (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ctime_ns", None),
    )
    if (
        before_identity != after_identity
        or before_store_revision != after_store_revision
    ):
        raise SourceProbeError("source changed during inspection")

    revision = SourceRevision(
        source_path=source,
        sandbox_identity=_digest((str(root),)),
        relative_path=rel,
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=getattr(after, "st_ctime_ns", None),
        store_revision=after_store_revision,
        width=width,
        height=height,
        exif=MappingProxyType(imported),
        tile_size=tile_size,
        overlap=overlap,
    )
    logger.info(
        "Browse source probe: revision=%s dimensions=%sx%s elapsed_ms=%.2f",
        revision.cache_key[:12],
        revision.width,
        revision.height,
        (time.perf_counter() - started) * 1000,
    )
    return revision


def _header_dimensions(source: Path) -> tuple[int | None, int | None]:
    if is_zarr_store_name(source):
        return None, None
    try:
        with PILImage.open(source) as image:
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001 - RAW support is optional
        if source_image_suffix(source).lower() not in RAW_IMAGE_EXTS:
            logger.debug("image header read failed", exc_info=True)
            return None, None
    try:
        import rawpy  # type: ignore[import-not-found]

        with rawpy.imread(str(source)) as raw:
            sizes = raw.sizes
            return int(sizes.width), int(sizes.height)
    except Exception:  # noqa: BLE001 - best-effort metadata
        logger.debug("RAW header read failed", exc_info=True)
        return None, None


def _read_exif(source: Path) -> dict[str, str]:
    try:
        with source.open("rb") as handle:
            tags = exifread.process_file(handle, details=False)
    except Exception:  # noqa: BLE001 - best-effort metadata
        logger.debug("EXIF header read failed", exc_info=True)
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def _digest(parts: tuple[Any, ...]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

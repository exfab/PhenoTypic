"""Deep Zoom Image (DZI) tile pyramid generation for the results viewer.

Generates an OpenSeadragon-compatible DZI pyramid from a PNG, with two
backends: a fast :mod:`pyvips`-driven path when ``pyvips`` is importable,
and a portable Pillow fallback used otherwise. Tile output, manifest
schema, and cache invalidation rules are identical across backends so
callers (and OpenSeadragon) cannot tell which path produced the tiles.

Output layout under ``output_dir``::

    <output_dir>/<png_stem>.dzi              # XML manifest
    <output_dir>/<png_stem>_files/<level>/<x>_<y>.png   # tiles

Tile filenames use the OpenSeadragon convention ``<col>_<row>.png``
(column first, then row). Per-image :class:`threading.Lock` instances
serialise duplicate concurrent requests for the same source image so two
in-flight requests don't double-tile.

**NOT DEAD CODE.** Two specs each say the tiler is "removed from this path" --
the results Plate path (`2026-08-26-viewer-viv-rebuild` section 6.1) and the
builder node-preview path (the same spec, section 7). Read together they
suggest a module that can now be deleted. It cannot. Live consumers:

    browse/_app.py:39                  DZI_BACKEND_INFO
    browse/_preparation.py:716         tile()
    browse/_preparation_routes.py:95   DZI_BACKEND_INFO
    builder/_point_picker.py:418       tile()

Browse keeps libvips -> DZI -> ``BrowseCache`` -> OpenSeadragon as its ONLY
pixel path (viewer-viv-rebuild spec section 9) because it reads arbitrary
source images with no run behind them, and so has no store to read. The point
picker picks points on a source image before any pipeline node has run, for
the same reason. Neither has an OME-Zarr store to hand a browser-side reader,
which is what the two "removed" paths replaced the tiler WITH.
"""

from __future__ import annotations

import functools
import logging
import math
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image as PILImage

try:
    import pyvips  # type: ignore[import-untyped,import-not-found]
except (
    ImportError,
    OSError,
) as exc:  # pragma: no cover - environment-dependent
    # ImportError: pyvips Python package not installed.
    # OSError: pyvips installed but the libvips C library is missing
    # (cffi raises OSError from dlopen). Either way fall back to Pillow.
    pyvips = None  # type: ignore[assignment]
    _PYVIPS_IMPORT_ERROR: BaseException | None = exc
else:
    _PYVIPS_IMPORT_ERROR = None

__all__ = [
    "DZI_BACKEND_INFO",
    "DziBackendInfo",
    "resolve_dzi_backend",
    "tile",
]

logger = logging.getLogger(__name__)

# Per-image locks: serialises duplicate concurrent tile requests for the
# same source PNG so we never double-write the cache. Bounded LRU so a
# long-running viewer that browses thousands of images doesn't grow the
# lock table without limit; an evicted lock that was being held would
# still block correctly via its existing references on the stack, and
# eviction only happens after ``maxsize`` distinct paths have been seen.
_LOCK_CACHE_SIZE = 512
_STORAGE_ERROR_MARKERS = (
    "disk full",
    "file name too long",
    "input/output error",
    "no space left on device",
    "not a directory",
    "permission denied",
    "quota exceeded",
    "read-only file system",
    "too many open files",
)

BackendMode = Literal["auto", "pillow", "pyvips"]
BackendName = Literal["pillow", "pyvips"]


@dataclass(frozen=True, slots=True)
class DziBackendInfo:
    """Resolved DZI generation backend.

    Attributes:
        name: Effective backend name.
        version: Native libvips version for the ``pyvips`` backend, otherwise
            ``None``.
        fallback_reason: Sanitized reason that automatic selection used Pillow,
            or ``None`` when no fallback occurred.
    """

    name: BackendName
    version: str | None
    fallback_reason: str | None


def _libvips_version() -> str | None:
    """Return the loaded native libvips version, when available."""
    if pyvips is None:
        return None
    try:
        return ".".join(str(pyvips.version(part)) for part in range(3))
    except (AttributeError, TypeError, ValueError):
        return None


def _unavailable_reason() -> str:
    """Return a path-free reason for automatic Pillow fallback."""
    if _PYVIPS_IMPORT_ERROR is None:
        return "pyvips is unavailable"
    return f"{type(_PYVIPS_IMPORT_ERROR).__name__}: libvips unavailable"


def resolve_dzi_backend(mode: BackendMode = "auto") -> DziBackendInfo:
    """Resolve a requested DZI backend without mutating global state.

    Args:
        mode: ``"auto"`` prefers pyvips and falls back to Pillow,
            ``"pillow"`` forces the portable backend, and ``"pyvips"``
            requires a working pyvips import.

    Returns:
        Immutable information about the effective backend.

    Raises:
        ValueError: If ``mode`` is not supported.
        RuntimeError: If pyvips is required but unavailable.
    """
    if mode not in {"auto", "pillow", "pyvips"}:
        raise ValueError(f"Unsupported DZI backend mode: {mode!r}")
    if mode == "pillow":
        return DziBackendInfo(
            name="pillow", version=None, fallback_reason=None
        )
    if pyvips is not None:
        return DziBackendInfo(
            name="pyvips", version=_libvips_version(), fallback_reason=None
        )
    if mode == "pyvips":
        raise RuntimeError(
            f"pyvips backend requested but {_unavailable_reason()}"
        ) from _PYVIPS_IMPORT_ERROR
    return DziBackendInfo(
        name="pillow", version=None, fallback_reason=_unavailable_reason()
    )


# Stable, startup-inspectable backend information. Callers that need a forced
# test/diagnostic backend use ``resolve_dzi_backend`` or ``tile(..., backend=)``.
DZI_BACKEND_INFO = resolve_dzi_backend()


@functools.lru_cache(maxsize=_LOCK_CACHE_SIZE)
def _get_lock(png_path: Path) -> threading.Lock:
    """Return a per-image lock, creating one on first access.

    Args:
        png_path: Source PNG path used as the lock key.

    Returns:
        A :class:`threading.Lock` unique to ``png_path``.
    """
    return threading.Lock()


def tile(
    png_path: Path,
    output_dir: Path,
    tile_size: int = 254,
    overlap: int = 1,
    *,
    backend: BackendMode = "auto",
) -> Path:
    """Return path to the DZI manifest, generating tiles if absent or stale.

    The function is idempotent: if a manifest already exists and is at
    least as new as the source PNG, the existing manifest path is
    returned without re-tiling. Otherwise the cache directory for this
    image is wiped and regenerated.

    Args:
        png_path: Source PNG to tile. Must exist on disk.
        output_dir: Directory in which to write ``<stem>.dzi`` and the
            ``<stem>_files/`` tile pyramid. Created if missing.
        tile_size: Edge length of an interior tile in pixels (default
            254). Edge tiles may be smaller; tiles bordering a neighbour
            include ``overlap`` extra pixels on that side.
        overlap: Per-side overlap in pixels with neighbouring tiles
            (default 1). OpenSeadragon needs at least 1 px of overlap to
            avoid seams during interpolation.
        backend: Backend selection mode. ``"auto"`` prefers pyvips,
            ``"pillow"`` provides a deterministic fallback/test seam, and
            ``"pyvips"`` requires pyvips.

    Returns:
        Absolute path to the ``<stem>.dzi`` XML manifest.

    Raises:
        FileNotFoundError: If ``png_path`` does not exist.
    """
    png_path = Path(png_path)
    output_dir = Path(output_dir)
    if not png_path.exists():
        raise FileNotFoundError(f"Source PNG not found: {png_path}")

    manifest_path = output_dir / f"{png_path.stem}.dzi"
    files_dir = output_dir / f"{png_path.stem}_files"

    lock = _get_lock(png_path)
    with lock:
        # Cache hit: manifest is at least as fresh as the source PNG.
        if (
            manifest_path.exists()
            and manifest_path.stat().st_mtime >= png_path.stat().st_mtime
        ):
            return manifest_path

        # Stale or partial: wipe the tile directory so we don't mix
        # tiles from a previous interrupted run with the new pyramid.
        if files_dir.exists():
            shutil.rmtree(files_dir)
        if manifest_path.exists():
            manifest_path.unlink()

        output_dir.mkdir(parents=True, exist_ok=True)

        with PILImage.open(png_path) as probe:
            width, height = probe.size
        backend_info = resolve_dzi_backend(backend)
        logger.info(
            "DZI tile generation start: backend=%s image=%s size=%dx%d",
            backend_info.name,
            png_path.name,
            width,
            height,
        )
        started = time.perf_counter()

        effective_backend = backend_info.name
        if backend_info.name == "pyvips":
            try:
                _tile_with_pyvips(png_path, output_dir, tile_size, overlap)
            except _PyvipsDzsaveError:
                # A libvips operation error from dzsave is safe to retry with
                # Pillow after removing every partial artifact. Filesystem
                # errors and failures outside dzsave propagate unchanged.
                _remove_partial_output(manifest_path, files_dir)
                logger.warning(
                    "DZI pyvips dzsave failed; retrying once with Pillow: "
                    "image=%s",
                    png_path.name,
                )
                _tile_with_pillow(png_path, output_dir, tile_size, overlap)
                effective_backend = "pillow"
        else:
            _tile_with_pillow(png_path, output_dir, tile_size, overlap)

        elapsed = time.perf_counter() - started
        logger.info(
            "DZI tile generation done: backend=%s image=%s elapsed=%.3fs",
            effective_backend,
            png_path.name,
            elapsed,
        )

    return manifest_path


class _PyvipsDzsaveError(RuntimeError):
    """Internal marker for a pyvips operation error raised by dzsave only."""


def _is_pyvips_runtime_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is the loaded pyvips runtime error type."""
    error_type = getattr(pyvips, "Error", None)
    return isinstance(error_type, type) and isinstance(exc, error_type)


def _is_storage_error(exc: BaseException) -> bool:
    """Return whether an error represents a non-retryable storage failure.

    pyvips reports native filesystem failures through its generic ``Error``
    class rather than preserving ``OSError`` subclasses. Inspecting the error
    chain and libvips' stable errno text keeps those failures from being
    mistaken for a codec/operation failure that Pillow could recover from.
    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, OSError):
            return True
        message = str(current).casefold()
        if any(marker in message for marker in _STORAGE_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _remove_partial_output(manifest_path: Path, files_dir: Path) -> None:
    """Delete an incomplete DZI manifest and tile tree before a retry."""
    if files_dir.exists():
        shutil.rmtree(files_dir)
    if manifest_path.exists():
        manifest_path.unlink()


def _tile_with_pyvips(
    png_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
) -> None:
    """Generate a DZI pyramid via :mod:`pyvips` ``dzsave``.

    ``dzsave`` writes the manifest to ``<base>.dzi`` and tiles into
    ``<base>_files/`` using the layout described in the module
    docstring, so the on-disk result is interchangeable with the Pillow
    fallback.

    Args:
        png_path: Source PNG.
        output_dir: Destination directory for ``<stem>.dzi`` and tiles.
        tile_size: Edge length of an interior tile in pixels.
        overlap: Per-side neighbour overlap in pixels.
    """
    assert pyvips is not None  # for type narrowing
    img = pyvips.Image.new_from_file(str(png_path), access="sequential")
    base = output_dir / png_path.stem
    try:
        img.dzsave(
            str(base),
            tile_size=tile_size,
            overlap=overlap,
            suffix=".png",
            layout="dz",
        )
    except Exception as exc:
        if not _is_pyvips_runtime_error(exc) or _is_storage_error(exc):
            raise
        raise _PyvipsDzsaveError from exc


def _tile_with_pillow(
    png_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
) -> None:
    """Generate a DZI pyramid using :mod:`PIL` only.

    Builds the pyramid by descending from the full-resolution level
    ``N`` to a 1x1 (or 2x2) level ``0``. Each step downsamples the
    previous level by 2x with Lanczos resampling. Each level is sliced
    into ``tile_size``-pitch tiles, with up to ``overlap`` extra pixels
    on every side that has a neighbour (so interior tiles measure
    ``tile_size + 2 * overlap`` and edge tiles are smaller).

    Args:
        png_path: Source PNG.
        output_dir: Destination directory for ``<stem>.dzi`` and tiles.
        tile_size: Edge length of an interior tile in pixels.
        overlap: Per-side neighbour overlap in pixels.
    """
    stem = png_path.stem
    files_dir = output_dir / f"{stem}_files"
    files_dir.mkdir(parents=True, exist_ok=True)

    with PILImage.open(png_path) as src:
        # OpenSeadragon expects RGB(A); converting once up front keeps
        # the resize/save loop free of mode surprises.
        if src.mode not in ("RGB", "RGBA", "L", "LA"):
            src = src.convert("RGBA")
        full = src.copy()

    width, height = full.size
    # Standard DZI: max level index N = ceil(log2(max(W, H))). The
    # special-case for 1x1 inputs keeps log2(1) = 0 but still emits a
    # single level.
    max_dim = max(width, height, 1)
    max_level = max(0, int(math.ceil(math.log2(max_dim))))

    # Build the pyramid top-down: store one PIL image per level.
    # level_images[level] is the image rendered at that DZI level.
    level_images: dict[int, PILImage.Image] = {max_level: full}
    current = full
    for level in range(max_level - 1, -1, -1):
        new_w = max(1, math.ceil(current.size[0] / 2))
        new_h = max(1, math.ceil(current.size[1] / 2))
        current = current.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        level_images[level] = current

    # Slice each level into tiles. Tiles include `overlap` extra pixels
    # on every side that abuts a neighbour, so interior tiles are
    # (tile_size + 2*overlap) square; tiles touching the image border
    # are clipped to the image extent.
    for level in range(max_level + 1):
        level_dir = files_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        level_img = level_images[level]
        lw, lh = level_img.size
        cols = max(1, math.ceil(lw / tile_size))
        rows = max(1, math.ceil(lh / tile_size))
        for row in range(rows):
            for col in range(cols):
                x_offset = col * tile_size
                y_offset = row * tile_size
                # Add overlap only on sides with a neighbour. Clamp to
                # the image bounds so border tiles end exactly at the
                # image edge.
                x_start = max(0, x_offset - overlap)
                y_start = max(0, y_offset - overlap)
                x_end = min(lw, x_offset + tile_size + overlap)
                y_end = min(lh, y_offset + tile_size + overlap)
                tile_img = level_img.crop((x_start, y_start, x_end, y_end))
                tile_path = level_dir / f"{col}_{row}.png"
                tile_img.save(tile_path, format="PNG")

    _write_dzi_manifest(
        output_dir / f"{stem}.dzi",
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
    )


def _write_dzi_manifest(
    manifest_path: Path,
    *,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> None:
    """Write the DZI XML manifest at ``manifest_path``.

    The schema is fixed by the OpenSeadragon DZI spec; we emit it as a
    literal f-string rather than building an :mod:`xml.etree` tree
    because the structure has exactly two elements and never varies.

    Args:
        manifest_path: Destination ``.dzi`` file path.
        width: Full-resolution image width in pixels.
        height: Full-resolution image height in pixels.
        tile_size: Edge length of an interior tile in pixels.
        overlap: Per-side neighbour overlap in pixels.
    """
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        '       Format="png"\n'
        f'       Overlap="{overlap}"\n'
        f'       TileSize="{tile_size}">\n'
        f'    <Size Width="{width}" Height="{height}"/>\n'
        "</Image>\n"
    )
    manifest_path.write_text(xml, encoding="utf-8")

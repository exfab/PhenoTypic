"""Read EXIF capture time without decoding pixels.

The Browse Timeline orders the time axis (no-CSV / EXIF path) by capture
time over potentially hundreds of source images. Routing that through
``Image.imread(...).rgb[:]`` (as ``browse/_metadata.read`` does) would decode
every image — far too expensive (spec §15.11). ``exifread`` parses only the
EXIF block, so this is cheap, and results are cached per ``(path, mtime_ns)``.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import exifread

logger = logging.getLogger(__name__)

__all__ = ["read_capture_time"]


@functools.lru_cache(maxsize=4096)
def _read_capture_time_cached(path_str: str, mtime_ns: int) -> str | None:
    del mtime_ns  # cache-key only (invalidates when the file changes)
    try:
        with open(path_str, "rb") as handle:
            tags = exifread.process_file(
                handle, details=False, stop_tag="DateTimeOriginal"
            )
    except Exception:  # noqa: BLE001 - capture time is best-effort
        logger.debug("EXIF read failed for %s", path_str, exc_info=True)
        return None
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        value = tags.get(key)
        if value is not None:
            return str(value)
    return None


def read_capture_time(path: Path) -> str | None:
    """Return the EXIF capture-time string for ``path``, or ``None``.

    Prefers ``DateTimeOriginal`` (true capture) over the bare ``DateTime``
    (often the file write/scan time), mirroring ``browse/_metadata``'s
    ordering. Best-effort: any failure returns ``None``.
    """
    path = Path(path)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    return _read_capture_time_cached(str(path), mtime_ns)

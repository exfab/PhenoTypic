"""Read display metadata (dims, file size, EXIF) from a source image file.

EXIF is pulled from ``phenotypic.Image``'s imported metadata, which is
populated by ``exifread`` for both JPEG and TIFF-based RAW (NEF/CR2). Any
field that is absent or unreadable is silently omitted — the panel degrades
gracefully rather than raising.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from phenotypic import Image

logger = logging.getLogger(__name__)

__all__ = ["read"]


def _extract_exif(imported: dict[str, Any]) -> dict[str, str]:
    """Pull capture-time + camera make/model from an exifread-style dict.

    The matching is deliberately defensive about EXIF's near-duplicate keys:

    * ``captured`` prefers ``DateTimeOriginal`` (true capture time) over the
      bare ``DateTime`` (often the file/scan *write* time), so the two are
      tried in separate ordered passes rather than as one substring set.
    * ``make`` / ``model`` match keys whose lowercased form *ends with* the
      target word and does **not** contain ``"lens"``, so ``Image Model`` /
      ``Image Make`` win while ``LensModel`` / ``CameraModelName`` are
      ignored.
    """

    def _find_substring(needle: str) -> str | None:
        for key, value in imported.items():
            if needle in key.lower():
                return str(value)
        return None

    def _find_body_field(word: str) -> str | None:
        for key, value in imported.items():
            key_lower = key.lower()
            if key_lower.endswith(word) and "lens" not in key_lower:
                return str(value)
        return None

    out: dict[str, str] = {}
    # Ordered fallback: true capture time first, file/scan write time second.
    captured = _find_substring("datetimeoriginal") or _find_substring("datetime")
    make = _find_body_field("make")
    model = _find_body_field("model")
    if captured:
        out["captured"] = captured
    if make:
        out["make"] = make
    if model:
        out["model"] = model
    return out


def read(original: Path) -> dict[str, Any]:
    """Return ``{width, height, bytes, exif}`` for ``original`` (best-effort)."""
    original = Path(original)
    info: dict[str, Any] = {"width": None, "height": None, "bytes": None, "exif": {}}
    try:
        info["bytes"] = original.stat().st_size
    except OSError:
        pass
    try:
        img = Image.imread(original)
        arr = img.rgb[:]
        info["height"], info["width"] = int(arr.shape[0]), int(arr.shape[1])
        imported = dict(getattr(img._metadata, "imported", {}) or {})
    except Exception:  # noqa: BLE001 - metadata is best-effort
        logger.debug("metadata read failed for %s", original, exc_info=True)
        return info
    info["exif"] = _extract_exif(imported)
    return info

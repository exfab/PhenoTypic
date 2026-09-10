"""Source-file → faithful 8-bit RGB PNG, with an ephemeral temp tile cache.

The Browse tab never tiles a source file directly; it first normalizes any
supported format (standard *or* camera RAW) to an 8-bit RGB PNG via
``phenotypic.Image.imread`` + ``skimage.util.img_as_ubyte`` (a faithful
full-range downcast — no auto-contrast), then hands that PNG to the shared
DZI tiler.

**This module owns no cache.** :func:`normalize_to_png` writes to a
destination its caller supplies, which in production is ``BrowseCache``'s
``staged.normalized_png`` (``_preparation.py``, ``_tile_routes.py``). An
earlier ephemeral ``tempfile.gettempdir()/phenotypic/browse`` cache lived
here behind ``browse_cache_base`` / ``cache_png_path`` / ``init_cache`` /
``wipe_cache``; the persistent cache replaced it and those four lost their
last production caller. Deleted in P6 Task 7 — do not reintroduce a
module-owned cache here, because two caches keyed differently for one
purpose is what made the dead set hard to spot.
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

from PIL import Image as PILImage
from skimage.util import img_as_ubyte

from phenotypic import Image
from phenotypic._gui._config import RAW_IMAGE_EXTS
from phenotypic.sdk_ import source_image_suffix

logger = logging.getLogger(__name__)

__all__ = [
    "SourceRenderUnavailable",
    "encode_token",
    "decode_token",
    "normalize_to_png",
]


class SourceRenderUnavailable(RuntimeError):
    """Raised when a source file cannot be decoded on this platform.

    The common case is camera RAW on Windows, where ``rawpy`` is excluded.
    The tile route maps this to a 422 + an inline viewer notice.
    """


def encode_token(sandbox_rel: str) -> str:
    """Encode a sandbox-relative POSIX path as a slash-free base64url token."""
    raw = base64.urlsafe_b64encode(sandbox_rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def decode_token(token: str) -> str:
    """Inverse of :func:`encode_token`. Raises on malformed input."""
    pad = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode((token + pad).encode("ascii")).decode(
        "utf-8"
    )


def normalize_to_png(original: Path, cache_png: Path) -> Path:
    """Render ``original`` to a faithful 8-bit RGB PNG at ``cache_png``.

    Idempotent: returns the existing PNG when it is at least as new as the
    source. RAW that cannot be decoded raises :class:`SourceRenderUnavailable`;
    a decode failure on a standard format re-raises the original error.
    """
    original = Path(original)
    if (
        cache_png.exists()
        and cache_png.stat().st_mtime >= original.stat().st_mtime
    ):
        return cache_png
    try:
        rgb = Image.imread(original).rgb[:]
    except Exception as exc:  # noqa: BLE001 - classify by extension below
        if source_image_suffix(original).lower() in RAW_IMAGE_EXTS:
            raise SourceRenderUnavailable(
                f"cannot decode RAW source on this platform: {original.name}"
            ) from exc
        raise
    rgb8 = img_as_ubyte(rgb)
    cache_png.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_png.with_name(
        f".{cache_png.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        PILImage.fromarray(rgb8).save(temporary, format="PNG")
        temporary.replace(cache_png)
    finally:
        temporary.unlink(missing_ok=True)
    return cache_png

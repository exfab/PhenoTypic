"""Source-file → faithful 8-bit RGB PNG, with an ephemeral temp tile cache.

The Browse tab never tiles a source file directly; it first normalizes any
supported format (standard *or* camera RAW) to an 8-bit RGB PNG via
``phenotypic.Image.imread`` + ``skimage.util.img_as_ubyte`` (a faithful
full-range downcast — no auto-contrast), then hands that PNG to the shared
DZI tiler. The cache lives under ``tempfile.gettempdir()/phenotypic/browse``,
keyed by a slash-free base64url token of the image's sandbox-relative path,
and is wiped on launch + at process exit.
"""

from __future__ import annotations

import atexit
import base64
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from PIL import Image as PILImage
from skimage.util import img_as_ubyte

from phenotypic import Image
from phenotypic.gui._config import BROWSE_CACHE_TMP_SUBPATH, RAW_IMAGE_EXTS

logger = logging.getLogger(__name__)

__all__ = [
    "SourceRenderUnavailable",
    "encode_token",
    "decode_token",
    "browse_cache_base",
    "cache_png_path",
    "wipe_cache",
    "init_cache",
    "normalize_to_png",
]

_atexit_registered = False


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


def browse_cache_base() -> Path:
    """The ephemeral cache root (recomputed each call so ``$TMPDIR`` is honoured)."""
    return Path(tempfile.gettempdir()).joinpath(*BROWSE_CACHE_TMP_SUBPATH)


def cache_png_path(token: str) -> Path:
    """Path to the normalized PNG the DZI tiler consumes for ``token``."""
    return browse_cache_base() / f"{token}.png"


def wipe_cache() -> None:
    """Best-effort recursive delete of the cache base. Never raises."""
    shutil.rmtree(browse_cache_base(), ignore_errors=True)


def init_cache() -> None:
    """Wipe stale tiles on launch and register an ``atexit`` cleanup (idempotent)."""
    global _atexit_registered
    wipe_cache()
    browse_cache_base().mkdir(parents=True, exist_ok=True)
    if not _atexit_registered:
        atexit.register(wipe_cache)
        _atexit_registered = True


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
        if original.suffix.lower() in RAW_IMAGE_EXTS:
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

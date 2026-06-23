"""Whole-image → cached downscaled thumbnail, plus a Flask route factory.

Mirrors ``gui/_shared/tiles.register_crop_route``: a surface supplies a
``resolve_source`` callable mapping a URL identity to an on-disk source PNG,
and this module owns the shared downscale + self-invalidating disk cache +
serving route. The cache filename embeds the source ``st_mtime_ns`` so a
regenerated source is served fresh without a stat-then-compare (spec §15.6);
writes are atomic (tempfile + os.replace).
"""
from __future__ import annotations

import base64
import functools
import io
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import dash
from flask import Blueprint, Response, jsonify, request, send_file
from PIL import Image as PILImage

from phenotypic.gui._config import snap_thumb_bucket

logger = logging.getLogger(__name__)

# Per-thumbnail render locks: serialise duplicate concurrent requests for the
# same cache key so a background-warm sweep firing many fetches never decodes +
# downscales the same thumbnail twice. Bounded LRU so a long-running viewer that
# browses thousands of images doesn't grow the lock table without limit; an
# evicted lock that was being held still blocks correctly via its existing
# references, and the atomic write covers the rare eviction-while-held race
# (spec §15.6). Mirrors ``results_viewer/_dzi_tiler._get_lock``.
_LOCK_CACHE_SIZE = 512


@functools.lru_cache(maxsize=_LOCK_CACHE_SIZE)
def _get_lock(key: str) -> threading.Lock:
    """Return a per-cache-key render lock, creating one on first access.

    Args:
        key: The thumbnail cache filename used as the lock key.

    Returns:
        A :class:`threading.Lock` unique to ``key``.
    """
    return threading.Lock()


class ThumbUnavailable(RuntimeError):
    """Raised by a resolver when a source cannot be decoded on this platform.

    The common case is camera RAW on Windows. The route maps this to 422 + a
    fixed client message.
    """


def thumb_cache_name(identity: str, bucket: int, mtime_ns: int) -> str:
    """Return a flat, safe, self-invalidating cache filename.

    Args:
        identity: The URL identity (may contain ``/``).
        bucket: The snapped thumbnail size bucket.
        mtime_ns: Source ``st_mtime_ns`` — embedded so a regenerated source
            maps to a new filename (self-invalidating).

    Returns:
        ``<base64url(identity)>_<bucket>_<mtime_ns>.png`` (no path separators).
    """
    token = base64.urlsafe_b64encode(identity.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{token}_{bucket}_{mtime_ns}.png"


def downscale_to_thumb(src_png: Path, size: int) -> bytes:
    """Downscale ``src_png`` so its longest edge is ``size`` px; return PNG bytes.

    Aspect ratio is preserved (``PILImage.thumbnail``). The source is converted
    to RGB so palette/RGBA inputs serve consistently. ``thumbnail`` never
    upscales, so a source smaller than ``size`` is returned at its own size.

    Args:
        src_png: Path to the source PNG (already normalized for Browse; the
            overlay PNG for Results).
        size: Target longest-edge length in pixels.

    Returns:
        PNG-encoded bytes of the downscaled RGB image.
    """
    with PILImage.open(src_png) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((size, size), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def register_thumbnail_route(
    app: dash.Dash,
    *,
    segment: str,
    resolve_source: Callable[[str], Path],
    cache_base: Path,
) -> None:
    """Mount ``GET /<segment>/<identity>?size=`` serving cached thumbnails.

    Args:
        app: The Dash app whose Flask server is extended.
        segment: URL path segment to mount under (e.g. ``"thumb"``). Also
            seeds the blueprint name so multiple segments coexist on one server.
        resolve_source: ``identity -> Path`` to the source PNG. Raise
            ``ThumbUnavailable`` for an undecodable source (→ 422) or
            ``FileNotFoundError`` for a missing one (→ 404). The resolver owns
            all path/sandbox validation.
        cache_base: Directory for the self-invalidating thumbnail cache
            (created on demand).
    """
    bp = Blueprint(f"timeline_thumb_{segment}", __name__, url_prefix=f"/{segment}")
    cache_base = Path(cache_base)

    @bp.route("/<path:identity>")
    def thumb_endpoint(identity: str) -> Response | tuple[str, int]:
        size = request.args.get("size", type=int)
        if size is None or size <= 0:
            return ("bad request: missing or invalid ?size=<int>", 400)
        bucket = snap_thumb_bucket(size)
        try:
            source = resolve_source(identity)
        except ThumbUnavailable as exc:
            logger.info("thumb unavailable for %s: %s", identity, exc)
            return _json_error("source cannot be rendered on this platform", 422)
        except FileNotFoundError:
            return _json_error("source not found", 404)
        source = Path(source)
        if not source.is_file():
            return _json_error("source not found", 404)

        cache_base.mkdir(parents=True, exist_ok=True)
        cache_file = cache_base / thumb_cache_name(
            identity, bucket, source.stat().st_mtime_ns
        )
        # Per-source render lock (spec §4.2/§15.6): when the background-warm
        # sweep fires many concurrent fetches, two requests for the SAME
        # thumbnail must not both decode+downscale. Double-checked locking —
        # cheap existence test outside the lock, authoritative re-check inside.
        if not cache_file.exists():
            with _get_lock(cache_file.name):
                if not cache_file.exists():
                    try:
                        data = downscale_to_thumb(source, bucket)
                    except Exception:
                        logger.exception("thumb generation failed for %s", identity)
                        return _json_error("thumbnail generation failed", 500)
                    _atomic_write_bytes(cache_file, data)
        return send_file(cache_file, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered timeline thumb route under /%s", segment)


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically (tempfile in the same dir + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_error(message: str, status: int) -> Response:
    """Build a small JSON error ``Response`` with the given status code."""
    response = jsonify({"error": message})
    response.status_code = status
    return response

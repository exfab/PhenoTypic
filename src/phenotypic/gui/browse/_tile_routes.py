"""Flask blueprint serving token-keyed DZI manifests + tiles for Browse.

The frontend points OpenSeadragon at ``/tiles/<token>.dzi`` where ``<token>``
is a slash-free base64url encoding of the image's path relative to the frozen
``SandboxRoot``. The blueprint validates + decodes the token, resolves the
original file through ``sandbox.resolve`` (the sole security boundary),
normalizes it to a cached 8-bit PNG, and lazily tiles it with the shared DZI
tiler. Mirrors ``results_viewer/_tile_routes.py`` with one token segment in
place of ``<dataset>/<stem>``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import dash
from flask import Blueprint, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from phenotypic.gui._config import BROWSE_TILES_PREFIX
from phenotypic.gui.browse import _source_render
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

#: DZI tile filenames are ``<col>_<row>.png`` per the OpenSeadragon spec.
_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")
#: base64url alphabet (no padding) — what ``encode_token`` produces.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")

__all__ = ["register"]


def register(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Mount the token-keyed DZI routes on ``app.server``.

    Two routes are exposed under the ``/tiles`` URL prefix:

    * ``GET /tiles/<token>.dzi`` — validates + decodes the token, resolves
      the source file through the sandbox, normalizes it to a cached 8-bit
      PNG, lazily tiles it, and returns the DZI XML manifest.
    * ``GET /tiles/<token>_files/<level>/<filename>`` — returns a single
      tile PNG from the ephemeral cache. Tile generation is *not* triggered
      here; the manifest endpoint is responsible for that.

    Args:
        app: The Dash application whose Flask server should be extended.
        sandbox: The frozen-at-launch ``SandboxRoot`` — the sole security
            boundary. Every decoded token is resolved through it; escapes
            map to 404.
    """
    bp = Blueprint("browse_tiles", __name__, url_prefix=BROWSE_TILES_PREFIX)

    def _resolve_original(token: str) -> Path | None:
        """Validate + decode ``token`` and resolve it inside the sandbox.

        Returns the resolved source file path, or ``None`` when the token is
        malformed, escapes the sandbox, or names a non-file.
        """
        if not _TOKEN_RE.match(token):
            return None
        try:
            rel = _source_render.decode_token(token)
        except Exception:  # noqa: BLE001 - malformed token
            return None
        try:
            resolved = sandbox.resolve(rel)
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        return resolved

    @bp.route("/<token>.dzi")
    def manifest(token: str) -> Response:
        """Serve the DZI manifest, normalizing + tiling the source if needed."""
        original = _resolve_original(token)
        if original is None:
            return _json_error("invalid or unknown image", 404)
        cache_png = _source_render.cache_png_path(token)
        try:
            _source_render.normalize_to_png(original, cache_png)
        except _source_render.SourceRenderUnavailable as exc:
            # Log the server-side detail but return a fixed client message;
            # never trust the exception text for the response body.
            logger.info("source render unavailable for token=%s: %s", token, exc)
            return _json_error(
                "source image cannot be rendered on this platform", 422
            )
        except Exception:
            logger.exception("source render failed for token=%s", token)
            return _json_error("render failed", 500)
        try:
            _dzi_tiler.tile(cache_png, _source_render.browse_cache_base())
        except Exception:
            logger.exception("DZI tiling failed for token=%s", token)
            return _json_error("tile generation failed", 500)
        return send_from_directory(
            _source_render.browse_cache_base(),
            f"{token}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<token>_files/<int:level>/<filename>")
    def tile_endpoint(token: str, level: int, filename: str) -> Response:
        """Serve an individual tile PNG from the ephemeral cache."""
        if not _TOKEN_RE.match(token):
            return _json_error("invalid token", 404)
        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            return _json_error("invalid tile filename", 404)
        tile_dir = _source_render.browse_cache_base() / f"{token}_files" / str(level)
        if not tile_dir.is_dir():
            return _json_error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered Browse tile routes under %s", BROWSE_TILES_PREFIX)


def _json_error(message: str, status: int) -> Response:
    """Build a small JSON error ``Response`` with the given status code."""
    response = jsonify({"error": message})
    response.status_code = status
    return response

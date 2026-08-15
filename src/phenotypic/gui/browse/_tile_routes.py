"""Revision-addressed Browse preview, DZI manifest, and tile routes."""

from __future__ import annotations

import re
import time

import dash
from flask import Blueprint, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from phenotypic.gui._config import BROWSE_TILES_PREFIX
from phenotypic.gui.browse import _source_render
from phenotypic.gui.browse._preparation_routes import (
    BrowsePreparationApi,
    resolve_revision,
)
from phenotypic.gui.browse._cache import BrowseCache
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._source_probe import SourceProbeError
from phenotypic.gui.shell._sandbox import SandboxRoot

_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_REVISION_RE = re.compile(r"^[a-f0-9]{64}$")
_IMMUTABLE_CACHE = "private, max-age=31536000, immutable"

__all__ = ["register"]


def register(
    app: dash.Dash,
    api: BrowsePreparationApi | SandboxRoot,
) -> None:
    """Mount revision-addressed assets and legacy Timeline-compatible DZI URLs."""
    if isinstance(api, SandboxRoot):
        sandbox = api
        cache = BrowseCache.for_sandbox(sandbox.root)
        api = BrowsePreparationApi(
            sandbox=sandbox,
            cache=cache,
            manager=BrowsePreparationManager(
                cache,
                normalize=lambda source,
                destination: _source_render.normalize_to_png(
                    source,
                    destination,
                ),
            ),
        )
    asset_bp = Blueprint("browse_assets", __name__, url_prefix="/assets")
    legacy_bp = Blueprint(
        "browse_tiles", __name__, url_prefix=BROWSE_TILES_PREFIX
    )

    def _revision(token: str, key: str | None = None):
        if not _TOKEN_RE.fullmatch(token):
            raise FileNotFoundError
        if key is not None and not _REVISION_RE.fullmatch(key):
            raise FileNotFoundError
        return resolve_revision(api.sandbox, token, key)

    def _selection_identity() -> tuple[str, int]:
        client_id = request.args.get("client_id", "asset-route")
        if not (0 < len(client_id) <= 128):
            client_id = "asset-route"
        try:
            generation = max(0, int(request.args.get("generation", "0")))
        except ValueError:
            generation = 0
        return client_id, generation

    def _prepare(token: str, key: str | None = None):
        lookup_started = time.perf_counter()
        revision = _revision(token, key)
        entry = api.cache.entry(revision)
        cache_hit = entry.dzi_ready
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        client_id, generation = _selection_identity()
        handle = api.select(client_id, generation, revision)
        return revision, entry, handle, cache_hit, lookup_ms

    @asset_bp.get("/<token>/<revision>/preview.png")
    def preview(token: str, revision: str) -> Response:
        try:
            source, entry, handle, _cache_hit, lookup_ms = _prepare(
                token, revision
            )
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        wait_started = time.perf_counter()
        handle.preview_ready.wait(timeout=120.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not source.matches_disk():
            return _error("source image changed", 409)
        if not entry.preview_ready:
            snapshot = handle.snapshot()
            status = 422 if snapshot.phase == "failed" else 500
            return _error("source preview unavailable", status)
        response = send_from_directory(
            entry.root,
            entry.preview.name,
            mimetype="image/png",
        )
        snapshot = handle.snapshot()
        return _asset_headers(
            response,
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )

    @asset_bp.get("/<token>/<revision>/preview-if-ready.png")
    def preview_if_ready(token: str, revision: str) -> Response:
        started = time.perf_counter()
        try:
            source = _revision(token, revision)
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        entry = api.cache.entry(source)
        lookup_ms = (time.perf_counter() - started) * 1000
        if not entry.preview_ready:
            return _error("preview not prepared", 404)
        response = send_from_directory(
            entry.root,
            entry.preview.name,
            mimetype="image/png",
        )
        return _asset_headers(response, lookup_ms, 0.0)

    @asset_bp.get("/<token>/<revision>/image.dzi")
    def manifest(token: str, revision: str) -> Response:
        try:
            source, entry, handle, _cache_hit, lookup_ms = _prepare(
                token, revision
            )
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        wait_started = time.perf_counter()
        handle.complete.wait(timeout=600.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not source.matches_disk():
            return _error("source image changed", 409)
        if not entry.dzi_ready:
            snapshot = handle.snapshot()
            status = 409 if snapshot.error_code == "source_changed" else 422
            return _error("source image cannot be prepared", status)
        response = send_from_directory(
            entry.dzi_dir,
            entry.dzi_manifest.name,
            mimetype="application/xml",
        )
        snapshot = handle.snapshot()
        return _asset_headers(
            response,
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )

    @asset_bp.get("/<token>/<revision>/image_files/<int:level>/<filename>")
    def tile(token: str, revision: str, level: int, filename: str) -> Response:
        try:
            source = _revision(token, revision)
        except FileNotFoundError:
            return _error("invalid or unknown image", 404)
        except SourceProbeError:
            return _error("source image changed", 409)
        if secure_filename(
            filename
        ) != filename or not _TILE_NAME_RE.fullmatch(filename):
            return _error("invalid tile filename", 404)
        entry = api.cache.entry(source)
        tile_dir = entry.dzi_dir / "normalized_files" / str(level)
        if not entry.dzi_ready or not tile_dir.is_dir():
            return _error("tile cache missing", 404)
        response = send_from_directory(
            tile_dir, filename, mimetype="image/png"
        )
        response.headers["Cache-Control"] = _IMMUTABLE_CACHE
        return response

    @legacy_bp.get("/<token>.dzi")
    def legacy_manifest(token: str) -> Response:
        try:
            _source, entry, handle, _cache_hit, lookup_ms = _prepare(token)
        except (FileNotFoundError, SourceProbeError):
            return _error("invalid or unknown image", 404)
        wait_started = time.perf_counter()
        handle.complete.wait(timeout=600.0)
        wait_ms = (time.perf_counter() - wait_started) * 1000
        if not entry.dzi_ready:
            return _error(
                "source image cannot be rendered on this platform", 422
            )
        response = send_from_directory(
            entry.dzi_dir,
            entry.dzi_manifest.name,
            mimetype="application/xml",
        )
        response.headers["Cache-Control"] = "private, no-cache"
        snapshot = handle.snapshot()
        response.headers["Server-Timing"] = _server_timing(
            lookup_ms,
            wait_ms,
            normalization_ms=snapshot.normalization_ms,
            dzi_ms=snapshot.dzi_ms,
        )
        return response

    @legacy_bp.get("/<token>_files/<int:level>/<filename>")
    def legacy_tile(token: str, level: int, filename: str) -> Response:
        try:
            source = _revision(token)
        except (FileNotFoundError, SourceProbeError):
            return _error("invalid or unknown image", 404)
        if secure_filename(
            filename
        ) != filename or not _TILE_NAME_RE.fullmatch(filename):
            return _error("invalid tile filename", 404)
        entry = api.cache.entry(source)
        tile_dir = entry.dzi_dir / "normalized_files" / str(level)
        if not entry.dzi_ready or not tile_dir.is_dir():
            return _error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(asset_bp)
    app.server.register_blueprint(legacy_bp)


def _asset_headers(
    response: Response,
    lookup_ms: float,
    wait_ms: float,
    *,
    normalization_ms: float = 0.0,
    dzi_ms: float = 0.0,
) -> Response:
    response.headers["Cache-Control"] = _IMMUTABLE_CACHE
    response.headers["Server-Timing"] = _server_timing(
        lookup_ms,
        wait_ms,
        normalization_ms=normalization_ms,
        dzi_ms=dzi_ms,
    )
    return response


def _server_timing(
    lookup_ms: float,
    wait_ms: float,
    *,
    normalization_ms: float = 0.0,
    dzi_ms: float = 0.0,
) -> str:
    return (
        f"cache;dur={lookup_ms:.2f}, queue;dur={wait_ms:.2f}, "
        f"normalize;dur={normalization_ms:.2f}, dzi;dur={dzi_ms:.2f}"
    )


def _error(message: str, status: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response

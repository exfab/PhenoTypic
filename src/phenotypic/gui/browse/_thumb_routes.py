"""Mount the Browse Timeline thumbnail route.

Thin adapter over the Phase 1 ``register_thumbnail_route`` factory: the
resolver decodes the base64url token, resolves it through the sandbox (the
sole security boundary), and normalizes it to the cached 8-bit PNG that the
factory then downscales to the requested bucket. RAW that cannot be decoded
on this platform maps to ``ThumbUnavailable`` (→ 422), mirroring the DZI route.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash

from phenotypic.gui._config import BROWSE_THUMB_URL_SEGMENT
from phenotypic.gui._shared.timeline import ThumbUnavailable, register_thumbnail_route
from phenotypic.gui.browse import _source_render
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Mount the token-keyed thumbnail route on ``app.server``."""

    def resolve_source(token: str) -> Path:
        try:
            rel = _source_render.decode_token(token)
            resolved = sandbox.resolve(rel)
        except Exception as exc:  # noqa: BLE001 - malformed/escaping token → 404
            raise FileNotFoundError(token) from exc
        if not resolved.is_file():
            raise FileNotFoundError(token)
        cache_png = _source_render.cache_png_path(token)
        try:
            _source_render.normalize_to_png(resolved, cache_png)
        except _source_render.SourceRenderUnavailable as exc:
            raise ThumbUnavailable(str(exc)) from exc
        return cache_png

    register_thumbnail_route(
        app,
        segment=BROWSE_THUMB_URL_SEGMENT,
        resolve_source=resolve_source,
        cache_base=_source_render.browse_cache_base() / "thumb",
    )
    logger.debug("Registered Browse thumbnail route under /%s", BROWSE_THUMB_URL_SEGMENT)

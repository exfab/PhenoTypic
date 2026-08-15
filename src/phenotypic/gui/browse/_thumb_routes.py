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
from phenotypic.gui._shared.timeline import (
    ThumbUnavailable,
    register_thumbnail_route,
)
from phenotypic.gui.browse._cache import BrowseCache
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._preparation_routes import (
    BrowsePreparationApi,
    resolve_revision,
)
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(
    app: dash.Dash,
    sandbox: SandboxRoot,
    preparation_api: BrowsePreparationApi | None = None,
) -> None:
    """Mount the token-keyed thumbnail route on ``app.server``."""
    if preparation_api is None:
        cache = BrowseCache.for_sandbox(sandbox.root)
        preparation_api = BrowsePreparationApi(
            sandbox=sandbox,
            cache=cache,
            manager=BrowsePreparationManager(cache),
        )

    def resolve_source(token: str) -> Path:
        try:
            revision = resolve_revision(sandbox, token)
        except Exception as exc:  # noqa: BLE001 - malformed/escaping token → 404
            raise FileNotFoundError(token) from exc
        handle = preparation_api.manager.request_preview(revision)
        if not handle.preview_ready.wait(timeout=120.0):
            raise ThumbUnavailable("preview preparation timed out")
        entry = preparation_api.cache.entry(revision)
        if not entry.preview_ready:
            raise ThumbUnavailable("preview unavailable")
        return entry.preview

    register_thumbnail_route(
        app,
        segment=BROWSE_THUMB_URL_SEGMENT,
        resolve_source=resolve_source,
        cache_base=preparation_api.cache.timeline_thumbs_root,
    )
    logger.debug(
        "Registered Browse thumbnail route under /%s", BROWSE_THUMB_URL_SEGMENT
    )

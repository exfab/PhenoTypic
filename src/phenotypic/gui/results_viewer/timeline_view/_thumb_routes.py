"""Mount the Results Timeline thumbnail route (overlay → cached downscale).

Thin adapter over the Phase 1 ``register_thumbnail_route`` factory: the
resolver decodes a ``(dataset, stem)`` identity, guards both halves with the
DZI route's path-component check, and resolves the overlay PNG via
``OutputRoot.overlay_path``. The factory downscales it to the requested
size bucket and serves a self-invalidating, atomically-written disk cache
under the output root's ``.viewer_cache/timeline_thumbs`` (persists with the
run). Per spec §15.6 the warm sweep decodes the file and relies on the disk
cache — it does NOT lean on the small ``_load_overlay_rgb`` LRU.

Overlay PNGs are plain 8-bit RGB and always decode, so the resolver only
ever raises ``FileNotFoundError`` (→ 404) for an unknown/missing/unsafe
identity; the factory reserves ``ThumbUnavailable`` (→ 422) for a
genuinely-undecodable source, which the happy path never hits.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash

from phenotypic.gui._config import VIEWER_THUMB_URL_SEGMENT
from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui._shared.timeline import register_thumbnail_route
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

__all__ = ["register", "encode_cell_ref", "decode_cell_ref"]

#: Subdir of the output root's viewer cache for downscaled overlay thumbnails.
_THUMB_CACHE_SUBDIR = "timeline_thumbs"


def encode_cell_ref(dataset: str, stem: str) -> str:
    """Encode a ``(dataset, stem)`` pair into a single URL-path identity.

    A stem never contains ``/`` and a dataset is a single path component,
    so ``dataset/stem`` round-trips by splitting on the LAST ``/``.
    """
    return f"{dataset}/{stem}"


def decode_cell_ref(identity: str) -> tuple[str, str]:
    """Inverse of :func:`encode_cell_ref` (split on the last ``/``)."""
    dataset, _, stem = identity.rpartition("/")
    return dataset, stem


def register(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the ``(dataset, stem)`` thumbnail route on ``app.server``."""

    def resolve_source(identity: str) -> Path:
        dataset, stem = decode_cell_ref(identity)
        if not (dataset and stem):
            raise FileNotFoundError(identity)
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            raise FileNotFoundError(identity)
        if not output_root.has_overlay(dataset, stem):
            raise FileNotFoundError(identity)
        overlay = output_root.overlay_path(dataset, stem)
        if not overlay.is_file():
            raise FileNotFoundError(identity)
        return overlay

    register_thumbnail_route(
        app,
        segment=VIEWER_THUMB_URL_SEGMENT,
        resolve_source=resolve_source,
        cache_base=output_root.viewer_cache_dir / _THUMB_CACHE_SUBDIR,
    )
    logger.debug(
        "Registered Results Timeline thumbnail route under /%s for root=%s",
        VIEWER_THUMB_URL_SEGMENT,
        output_root.root,
    )

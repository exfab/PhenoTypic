"""Mount the Results Timeline thumbnail route (overlay → cached downscale).

Thin adapter over the Phase 1 ``register_thumbnail_route`` factory: the
resolver decodes a ``(dataset, stem)`` identity, guards both halves with the
DZI route's path-component check, and resolves the overlay PNG via
``OutputRoot.overlay_path``. The factory downscales it to the requested
size bucket and serves a self-invalidating, atomically-written disk cache
under the fingerprinted external GUI cache. The selected output tree remains
byte-identical. Per spec §15.6 the warm sweep decodes the file and relies on
the disk cache; it does not lean on the small ``_load_overlay_rgb`` LRU.

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
from phenotypic.sdk_ import atomic_write_bytes, bytes_fingerprint

logger = logging.getLogger(__name__)

__all__ = ["register", "encode_cell_ref", "decode_cell_ref"]

#: Subdir of the external viewer cache for downscaled overlay thumbnails.
_THUMB_CACHE_SUBDIR = "timeline_thumbs"
_SOURCE_SNAPSHOT_SUBDIR = "overlay_source_snapshots"


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
        return _stable_overlay_snapshot(
            output_root,
            dataset=dataset,
            stem=stem,
            overlay=overlay,
        )

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


def _stable_overlay_snapshot(
    output_root: OutputRoot,
    *,
    dataset: str,
    stem: str,
    overlay: Path,
) -> Path:
    """Copy one verified overlay revision into the external content cache."""
    if not output_root.snapshot_is_current():
        raise FileNotFoundError("bound output snapshot changed")
    source_token = output_root.bound_image_source_token(dataset, stem)
    try:
        source_bytes = overlay.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(overlay) from exc
    if (
        not output_root.snapshot_is_current()
        or not output_root.image_source_token_is_current(
            dataset,
            stem,
            source_token,
        )
    ):
        raise FileNotFoundError("bound output snapshot changed")

    digest = bytes_fingerprint(source_bytes).removeprefix("sha256:")
    snapshot = (
        output_root.viewer_cache_dir
        / _SOURCE_SNAPSHOT_SUBDIR
        / dataset
        / stem
        / f"{digest}.png"
    )
    if not snapshot.is_file():
        atomic_write_bytes(snapshot, source_bytes)
    return snapshot

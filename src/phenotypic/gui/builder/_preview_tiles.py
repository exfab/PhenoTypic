"""Store-layer -> PNG staging + DZI tile blueprint for the node-preview modal.

The renderer (`_image_renderer`) and the DZI tiler stay unchanged: for a
requested (scope, node, channel) we read the layer from the node's OME-Zarr
store, project it to an 8-bit PNG, and hand the PNG path to
``_dzi_tiler.tile``. With pyvips installed the tiler streams; resident RAM
stays near zero.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import dash
import numpy as np
from flask import Blueprint, Response, send_from_directory
from PIL import Image as PILImage
from werkzeug.utils import secure_filename

from phenotypic import Image
from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._image_renderer import (
    _label_map_to_rgb, _normalize_to_uint8,
)
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._tile_routes import _TILE_NAME_RE, _json_error
from phenotypic.sdk_ import ngff_

logger = logging.getLogger(__name__)

PREVIEW_TILES_PREFIX = "/preview-tiles"
_VALID_CHANNELS = ("rgb", "gray", "detect_mat", "objmap", "overlay")
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")

__all__ = [
    "PREVIEW_TILES_PREFIX",
    "stage_channel_png",
    "preview_dzi_url",
    "register_node_preview_routes",
]


def _src_png_path(scope_dir: Path, block_id: str, channel: str) -> Path:
    return scope_dir / "tiles_src" / f"{block_id}__{channel}.png"


def _channel_to_rgb_uint8(store_path: Path, channel: str) -> np.ndarray:
    if channel == "overlay":
        detect = Image.load_layer_zarr(store_path, "detect_mat")
        objmap = Image.load_layer_zarr(store_path, "objmap")
        base = _normalize_to_uint8(detect)
        base = base[..., :3] if base.ndim == 3 else np.stack([base] * 3, -1)
        try:
            from skimage.color import label2rgb
            rgb = label2rgb(objmap, image=base, bg_label=0, alpha=0.4,
                            image_alpha=1.0, kind="overlay")
            return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        except Exception:  # noqa: BLE001
            return _label_map_to_rgb(objmap)
    arr = Image.load_layer_zarr(store_path, channel)
    if channel == "objmap":
        return _label_map_to_rgb(arr)
    u8 = _normalize_to_uint8(arr)
    if u8.ndim == 2:
        return np.stack([u8] * 3, axis=-1)
    return u8[..., :3]


def stage_channel_png(scope_dir: Path, block_id: str, channel: str,
                      store_path: Path) -> Path:
    """Render a channel from a node store to a cached PNG (idempotent).

    Freshness is measured against the store's root ``zarr.json``, never
    against the store DIRECTORY. A directory's ``st_mtime_ns`` does not move
    when a nested chunk is rewritten (verified), so the directory is not a
    sound staleness key even where it happens to work; the root is the file
    the promote writes last on every publish, so it moves exactly when the
    store's contents do.
    """
    png_path = _src_png_path(scope_dir, block_id, channel)
    root_mtime_ns = (store_path / ngff_.STORE_ROOT_JSON).stat().st_mtime_ns
    if png_path.exists() and png_path.stat().st_mtime_ns >= root_mtime_ns:
        return png_path
    rgb = _channel_to_rgb_uint8(store_path, channel)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = png_path.with_name(f".{png_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        PILImage.fromarray(rgb, mode="RGB").save(tmp_path, format="PNG")
        tmp_path.replace(png_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return png_path


def preview_dzi_url(url_prefix: str, session_id: str, scope_hash: str,
                    block_id: str, channel: str) -> str:
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return f"{base}preview-tiles/{session_id}/{scope_hash}/{block_id}/{channel}.dzi"


def _validate(session_id, scope_hash, block_id, channel) -> Optional[Response]:
    if (
        is_safe_path_component(session_id)
        and bool(_HASH_RE.match(scope_hash))
        and is_safe_path_component(block_id)
        and channel in _VALID_CHANNELS
    ):
        return None
    return _json_error("invalid preview tile request", 404)


def register_node_preview_routes(app: dash.Dash) -> None:
    """Register the preview DZI blueprint on the Flask server."""
    bp = Blueprint("builder_node_preview", __name__, url_prefix=PREVIEW_TILES_PREFIX)

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<channel>.dzi")
    def manifest(session_id, scope_hash, block_id, channel) -> Response:
        err = _validate(session_id, scope_hash, block_id, channel)
        if err is not None:
            return err
        sdir = pc.preview_cache_root() / session_id / scope_hash
        manifest_path = sdir / "manifest.json"
        if not manifest_path.exists():
            return _json_error("scope not cached", 404)
        nodes = json.loads(manifest_path.read_text()).get("nodes", {})
        node = nodes.get(block_id)
        if node is None:
            return _json_error("node not cached", 404)
        store_path = sdir / node["store"]
        # The ROOT, not the directory: the promote writes it last, so an
        # interrupted write leaves a directory with no root and the store
        # reads as ABSENT rather than as partial -- the same disposition
        # ``_preview_cache`` gives it when building the manifest.
        if not (store_path / ngff_.STORE_ROOT_JSON).is_file():
            return _json_error("node store missing", 404)
        try:
            png_path = stage_channel_png(sdir, block_id, channel, store_path)
            _dzi_tiler.tile(png_path, sdir / "dzi")
        except KeyError:
            logger.debug("layer not available in store for %s/%s", block_id, channel)
            return _json_error("layer not available", 404)
        except Exception:  # noqa: BLE001
            logger.exception("preview tile generation failed")
            return _json_error("tile generation failed", 500)
        return send_from_directory(
            sdir / "dzi", f"{block_id}__{channel}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<channel>_files/<int:level>/<filename>")
    def tile_endpoint(session_id, scope_hash, block_id, channel, level,
                      filename) -> Response:
        err = _validate(session_id, scope_hash, block_id, channel)
        if err is not None:
            return err
        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            return _json_error("invalid tile filename", 404)
        tile_dir = (
            pc.preview_cache_root() / session_id / scope_hash / "dzi"
            / f"{block_id}__{channel}_files" / str(level)
        )
        if not tile_dir.is_dir():
            return _json_error("tile cache missing", 404)
        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug("Registered node-preview tile routes under %s", PREVIEW_TILES_PREFIX)

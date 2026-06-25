"""Flask blueprint serving DZI manifests and tiles for the results viewer.

The viewer's frontend points OpenSeadragon at
``/tiles/<dataset>/<stem>.dzi``; this module mounts the matching Flask
routes on the Dash app's underlying ``app.server``. The blueprint is
purposefully thin: validation, lazy DZI tile generation through
:mod:`phenotypic.gui.results_viewer._dzi_tiler`, and ``send_from_directory``
for byte streaming. Path-traversal hardening is enforced explicitly even
though dataset names come from the filesystem scan, since the URL is
user-controllable.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import cast, get_args

import dash
from flask import Blueprint, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from phenotypic.gui._config import VIEWER_TILES_PREFIX
from phenotypic.gui._shared.tiles import (
    LayerName,
    _load_hdf_layer_rgb,
    is_safe_path_component,
)
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

#: DZI tile filenames are ``<col>_<row>.png`` per the OpenSeadragon spec.
_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")

#: Backwards-compatible alias — the path-traversal guard moved to
#: :mod:`phenotypic.gui._shared.tiles`. Re-exported under its historical
#: private name so callers importing it from here keep working.
_is_safe_path_component = is_safe_path_component

#: Sentinel "layer" naming the baked overlay PNG source (not an HDF layer).
#: A standalone deliverables bundle has only overlays, so it always resolves
#: to this; full runs use it when the user explicitly asks for ``?layer=overlay``.
_OVERLAY_LAYER = "overlay"

#: Every ``?layer=`` value the DZI route accepts: the displayable HDF layers
#: (:data:`phenotypic.gui._shared.tiles.LayerName`) plus the overlay sentinel.
_VALID_DZI_LAYERS: tuple[str, ...] = (*get_args(LayerName), _OVERLAY_LAYER)


def _dzi_cache_dir_for(
    cache_root: Path, dataset: str, stem: str, layer: str
) -> Path:
    """Return the per-(image, layer) DZI cache directory.

    The DZI cache gained a *layer* dimension in Task 9 so the same image can
    cache an ``rgb`` pyramid alongside an ``objmap`` one without collision:
    ``<cache_root>/<dataset>/<stem>/<layer>/``. ``cache_root`` is
    :attr:`OutputRoot.cache_dir` (``<root>/.viewer_cache/dzi``).

    Args:
        cache_root: The DZI cache root (``OutputRoot.cache_dir``).
        dataset: Dataset name (already path-validated by the caller).
        stem: Image stem (already path-validated by the caller).
        layer: Layer key — one of :data:`_VALID_DZI_LAYERS`.

    Returns:
        The ``cache_root / dataset / stem / layer`` directory path (not
        created; the caller ``mkdir``\\ s it).
    """
    return cache_root / dataset / stem / layer


def _resolve_dzi_layer(
    layer_raw: str | None, *, has_results: bool, has_hdf: bool
) -> str | None:
    """Normalize a raw ``?layer=`` value to its cache-dir key, or ``None`` if invalid.

    Resolution rules:

    * Omitted (``None``) → ``"rgb"`` for a full run (``has_results``), else the
      ``"overlay"`` sentinel (a standalone bundle has no HDF layers to source).
    * A value outside :data:`_VALID_DZI_LAYERS` → ``None`` (the caller 404s).
    * When no per-image HDF is available, or the caller asked for the overlay
      explicitly, the layer collapses to ``"overlay"`` so the manifest and tile
      endpoints agree on the cache dir and the overlay PNG is tiled.

    Args:
        layer_raw: The raw ``?layer=`` query value (``None`` when omitted).
        has_results: Whether per-image ``results/`` HDFs exist
            (``OutputRoot.has_results``).
        has_hdf: Whether *this* image has a full-res HDF
            (``OutputRoot.hdf_path(...) is not None``).

    Returns:
        The normalized layer key (one of :data:`_VALID_DZI_LAYERS`), or
        ``None`` for an unrecognized value.
    """
    if layer_raw is None:
        layer = "rgb" if has_results else _OVERLAY_LAYER
    elif layer_raw in _VALID_DZI_LAYERS:
        layer = layer_raw
    else:
        return None
    if not has_hdf or layer == _OVERLAY_LAYER:
        return _OVERLAY_LAYER
    return layer


def register(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the DZI tile-serving routes on ``app.server``.

    Two routes are exposed under the ``/tiles`` URL prefix:

    * ``GET /tiles/<dataset>/<stem>.dzi`` — returns the DZI XML
      manifest, lazily generating the tile pyramid on first request.
    * ``GET /tiles/<dataset>/<stem>_files/<level>/<filename>`` — returns
      a single tile PNG. Tile generation is *not* triggered here; the
      manifest endpoint is responsible for that.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory.
            Captured by closure and used to resolve overlay PNGs and
            the per-(image, layer) cache directory.
    """
    # ``flask.request`` is imported function-local (never as a module
    # attribute) for the same reason ``_shared.tiles.register_crop_route``
    # does: it is a Werkzeug ``LocalProxy`` whose ``__name__`` access outside
    # a request context raises ``RuntimeError``, which the public-namespace
    # walk in ``tests/unit/test_pickleable.py`` would trip during parametrize
    # ID generation. Both route handlers below close over this one binding.
    from flask import request

    bp = Blueprint("results_viewer_tiles", __name__, url_prefix=VIEWER_TILES_PREFIX)

    @bp.route("/<dataset>/<stem>.dzi")
    def manifest(dataset: str, stem: str) -> Response:
        """Serve the DZI XML manifest, generating the pyramid if needed.

        The pyramid is keyed by ``(dataset, stem, layer)``: ``?layer=`` selects
        which full-res HDF layer (``rgb`` / ``detect_mat`` / ``objmap``) sources
        the pixels, defaulting to ``rgb`` for a full run. A standalone
        deliverables bundle (no per-image HDF) — or an explicit
        ``?layer=overlay`` — tiles the baked overlay PNG instead.
        """
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            logger.warning(
                "Rejected tile manifest request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return _json_error("invalid dataset or stem", 404)

        h5 = output_root.hdf_path(dataset, stem)
        layer = _resolve_dzi_layer(
            request.args.get("layer", type=str),
            has_results=output_root.has_results,
            has_hdf=h5 is not None,
        )
        if layer is None:
            return _json_error("invalid layer", 404)

        # Capability gate that works for a standalone deliverables bundle
        # (which has no ``results/`` to anchor a dataset-dir existence check):
        # accept when either a full-res per-image HDF or a baked overlay PNG
        # exists for this image.
        if h5 is None and not output_root.has_overlay(dataset, stem):
            return _json_error(
                f"no image source for {dataset!r}/{stem!r}", 404
            )

        cache_dir = _dzi_cache_dir_for(output_root.cache_dir, dataset, stem, layer)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Skip-if-exists: a per-(image, layer) pyramid already on disk is
        # served directly. This guards the expensive HDF branch, whose
        # unconditional ``_load_hdf_layer_rgb(...).save(source_png)`` bumps the
        # source PNG mtime on every request and so defeats ``_dzi_tiler.tile``'s
        # own freshness check (re-tiling each load). It matches the overlay
        # branch's idempotency; mtime-staleness is intentionally ignored,
        # exactly as the overlay path already does.
        if not (cache_dir / f"{stem}.dzi").exists():
            try:
                if layer == _OVERLAY_LAYER:
                    if not output_root.has_overlay(dataset, stem):
                        return _json_error(
                            f"no overlay for {dataset!r}/{stem!r}", 404
                        )
                    _dzi_tiler.tile(
                        output_root.overlay_path(dataset, stem), cache_dir
                    )
                else:
                    # ``_resolve_dzi_layer`` collapses every layer to the overlay
                    # sentinel when no HDF is available, so ``h5`` is non-None here.
                    assert h5 is not None
                    source_png = cache_dir / f"{stem}.png"
                    _load_hdf_layer_rgb(
                        str(h5), os.stat(h5).st_mtime_ns, cast(LayerName, layer)
                    ).save(source_png)
                    _dzi_tiler.tile(source_png, cache_dir)
            except Exception:
                logger.exception(
                    "DZI tile generation failed: dataset=%s stem=%s layer=%s",
                    dataset,
                    stem,
                    layer,
                )
                return _json_error("tile generation failed", 500)

        return send_from_directory(
            cache_dir,
            f"{stem}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<dataset>/<stem>_files/<int:level>/<filename>")
    def tile_endpoint(
        dataset: str, stem: str, level: int, filename: str
    ) -> Response:
        """Serve an individual tile PNG from the per-(image, layer) cache."""
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            logger.warning(
                "Rejected tile request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return _json_error("invalid dataset or stem", 404)

        # Reject anything that would even let `secure_filename` lose
        # information; tile filenames must match exactly ``\d+_\d+.png``.
        secured = secure_filename(filename)
        if secured != filename or not _TILE_NAME_RE.match(filename):
            logger.warning(
                "Rejected tile request with unsafe filename: %r",
                filename,
            )
            return _json_error("invalid tile filename", 404)

        # Resolve the layer the SAME way ``manifest`` did so a tile request
        # lands in the per-layer pyramid the manifest generated.
        layer = _resolve_dzi_layer(
            request.args.get("layer", type=str),
            has_results=output_root.has_results,
            has_hdf=output_root.hdf_path(dataset, stem) is not None,
        )
        if layer is None:
            return _json_error("invalid layer", 404)

        cache_dir = _dzi_cache_dir_for(output_root.cache_dir, dataset, stem, layer)
        tile_dir = cache_dir / f"{stem}_files" / str(level)
        if not tile_dir.is_dir():
            # Manifest endpoint is responsible for tiling; if a tile
            # request beats it, return 404 rather than firing tile
            # generation here (avoids racy concurrent writes).
            return _json_error(
                f"tile cache missing for {dataset!r}/{stem!r}", 404
            )

        return send_from_directory(
            tile_dir, filename, mimetype="image/png"
        )

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer tile routes under /tiles for root=%s",
        output_root.root,
    )


def _json_error(message: str, status: int) -> Response:
    """Build a small JSON error ``Response`` with the given status code.

    Args:
        message: Human-readable error string surfaced to the caller.
        status: HTTP status code to attach.

    Returns:
        A Flask :class:`~flask.Response` with ``application/json`` body.
    """
    response = jsonify({"error": message})
    response.status_code = status
    return response

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
import shutil
import tempfile
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
from phenotypic.sdk_ import file_fingerprint
from phenotypic.sdk_._file_locking import exclusive_path_lock

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
_SOURCE_TOKEN_FILENAME = ".source-token"


class _SourceSnapshotChanged(RuntimeError):
    """Raised when a request no longer matches its bound output revision."""


class _DziLayerUnavailable(KeyError):
    """Raised when neither the requested HDF layer nor an overlay exists."""


def _dzi_cache_dir_for(
    cache_root: Path, dataset: str, stem: str, layer: str
) -> Path:
    """Return the per-(image, layer) DZI cache directory.

    The DZI cache gained a *layer* dimension in Task 9 so the same image can
    cache an ``rgb`` pyramid alongside an ``objmap`` one without collision:
    ``<cache_root>/<dataset>/<stem>/<layer>/``. ``cache_root`` is the
    fingerprinted external :attr:`OutputRoot.cache_dir`; it is never beneath
    the selected output tree.

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

    bp = Blueprint(
        "results_viewer_tiles", __name__, url_prefix=VIEWER_TILES_PREFIX
    )

    @bp.route("/<dataset>/<stem>.dzi")
    def manifest(dataset: str, stem: str) -> Response:
        """Serve the DZI XML manifest, generating the pyramid if needed.

        The pyramid is keyed by ``(dataset, stem, layer)``: ``?layer=`` selects
        which full-res HDF layer (``rgb`` / ``detect_mat`` / ``objmap``) sources
        the pixels, defaulting to ``rgb`` for a full run. A standalone
        deliverables bundle (no per-image HDF) — or an explicit
        ``?layer=overlay`` — tiles the baked overlay PNG instead.
        """
        if not is_safe_path_component(dataset) or not is_safe_path_component(
            stem
        ):
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
        if not output_root.snapshot_is_current():
            return _json_error(
                "source snapshot changed; refresh Results before viewing",
                409,
            )
        source_token = output_root.bound_image_source_token(dataset, stem)

        cache_dir = _dzi_cache_dir_for(
            output_root.cache_dir, dataset, stem, layer
        )

        try:
            _publish_dzi_cache(
                output_root,
                dataset=dataset,
                stem=stem,
                layer=layer,
                h5=h5,
                cache_dir=cache_dir,
                source_token=source_token,
            )
        except _SourceSnapshotChanged:
            return _json_error(
                "source snapshot changed; refresh Results before viewing",
                409,
            )
        except _DziLayerUnavailable:
            return _json_error(
                f"HDF layer {layer!r} not found for {dataset!r}/{stem!r}",
                404,
            )
        except Exception:
            logger.exception(
                "DZI tile generation failed: dataset=%s stem=%s layer=%s",
                dataset,
                stem,
                layer,
            )
            return _json_error("tile generation failed", 500)

        if not output_root.snapshot_is_current():
            return _json_error(
                "source snapshot changed; refresh Results before viewing",
                409,
            )
        if not output_root.image_source_token_is_current(
            dataset,
            stem,
            source_token,
        ):
            return _json_error(
                "image source changed; refresh Results before viewing",
                409,
            )
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
        if not is_safe_path_component(dataset) or not is_safe_path_component(
            stem
        ):
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
        if not output_root.snapshot_is_current():
            return _json_error(
                "source snapshot changed; refresh Results before viewing",
                409,
            )

        cache_dir = _dzi_cache_dir_for(
            output_root.cache_dir, dataset, stem, layer
        )
        tile_dir = cache_dir / f"{stem}_files" / str(level)
        if not tile_dir.is_dir():
            # Manifest endpoint is responsible for tiling; if a tile
            # request beats it, return 404 rather than firing tile
            # generation here (avoids racy concurrent writes).
            return _json_error(
                f"tile cache missing for {dataset!r}/{stem!r}", 404
            )
        source_token = output_root.bound_image_source_token(dataset, stem)
        if not _cache_source_token_is_current(
            output_root,
            dataset=dataset,
            stem=stem,
            cache_dir=cache_dir,
            source_token=source_token,
        ):
            return _json_error(
                "image source changed; reload its manifest before viewing",
                409,
            )

        return send_from_directory(tile_dir, filename, mimetype="image/png")

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer tile routes under /tiles for root=%s",
        output_root.root,
    )


def _publish_dzi_cache(
    output_root: OutputRoot,
    *,
    dataset: str,
    stem: str,
    layer: str,
    h5: Path | None,
    cache_dir: Path,
    source_token: str,
) -> None:
    """Publish one complete DZI generation only for the bound source revision."""
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.with_name(f".{cache_dir.name}.publish.lock")
    with exclusive_path_lock(lock_path):
        manifest_path = cache_dir / f"{stem}.dzi"
        if manifest_path.is_file() and _cache_source_token_is_current(
            output_root,
            dataset=dataset,
            stem=stem,
            cache_dir=cache_dir,
            source_token=source_token,
        ):
            if not output_root.snapshot_is_current():
                raise _SourceSnapshotChanged
            return
        if (
            not output_root.snapshot_is_current()
            or not output_root.image_source_token_is_current(
                dataset,
                stem,
                source_token,
            )
        ):
            raise _SourceSnapshotChanged

        staging_dir = Path(
            tempfile.mkdtemp(
                dir=cache_dir.parent,
                prefix=f".{cache_dir.name}.",
                suffix=".generation",
            )
        )
        try:
            _generate_dzi_stage(
                output_root,
                dataset=dataset,
                stem=stem,
                layer=layer,
                h5=h5,
                staging_dir=staging_dir,
            )
            (staging_dir / _SOURCE_TOKEN_FILENAME).write_text(
                source_token,
                encoding="ascii",
            )
            if (
                not output_root.snapshot_is_current()
                or not output_root.image_source_token_is_current(
                    dataset,
                    stem,
                    source_token,
                )
            ):
                raise _SourceSnapshotChanged
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.replace(staging_dir, cache_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)


def _cache_source_token_is_current(
    output_root: OutputRoot,
    *,
    dataset: str,
    stem: str,
    cache_dir: Path,
    source_token: str,
) -> bool:
    """Return whether a DZI cache belongs to the current requested source."""
    try:
        cached_token = (cache_dir / _SOURCE_TOKEN_FILENAME).read_text(
            encoding="ascii"
        )
    except OSError:
        return False
    return (
        cached_token == source_token
        and output_root.image_source_token_is_current(
            dataset,
            stem,
            source_token,
        )
    )


def _generate_dzi_stage(
    output_root: OutputRoot,
    *,
    dataset: str,
    stem: str,
    layer: str,
    h5: Path | None,
    staging_dir: Path,
) -> None:
    """Generate one unpublished DZI directory from its selected source."""
    if layer == _OVERLAY_LAYER:
        if not output_root.has_overlay(dataset, stem):
            raise _DziLayerUnavailable
        _tile_overlay_source(output_root, dataset, stem, staging_dir)
        return

    assert h5 is not None
    source_png = staging_dir / f"{stem}.png"
    try:
        _ensure_hdf_layer_source_png(
            h5,
            cast(LayerName, layer),
            source_png,
        )
        _dzi_tiler.tile(source_png, staging_dir)
    except KeyError:
        if not output_root.has_overlay(dataset, stem):
            raise _DziLayerUnavailable from None
        logger.warning(
            "HDF layer %s missing for %s/%s; tiling overlay fallback",
            layer,
            dataset,
            stem,
        )
        _tile_overlay_source(output_root, dataset, stem, staging_dir)


def _ensure_hdf_layer_source_png(
    h5: Path, layer: LayerName, source_png: Path
) -> None:
    """Refresh the rendered HDF-layer source PNG only when the HDF is newer."""
    h5_stat = os.stat(h5)
    if (
        source_png.exists()
        and source_png.stat().st_mtime_ns >= h5_stat.st_mtime_ns
    ):
        return
    content_token = int(
        file_fingerprint(h5).removeprefix("sha256:")[:16],
        16,
    )
    _load_hdf_layer_rgb(str(h5), content_token, layer).save(source_png)
    os.utime(source_png, ns=(h5_stat.st_mtime_ns, h5_stat.st_mtime_ns))


def _tile_overlay_source(
    output_root: OutputRoot,
    dataset: str,
    stem: str,
    cache_dir: Path,
) -> None:
    """Tile the baked overlay PNG into ``cache_dir``."""
    _dzi_tiler.tile(output_root.overlay_path(dataset, stem), cache_dir)


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

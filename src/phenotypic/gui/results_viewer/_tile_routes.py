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
import re

import dash
from flask import Blueprint, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from phenotypic.gui._config import VIEWER_TILES_PREFIX
from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

#: DZI tile filenames are ``<col>_<row>.png`` per the OpenSeadragon spec.
_TILE_NAME_RE = re.compile(r"^\d+_\d+\.png$")

#: Backwards-compatible alias — the path-traversal guard moved to
#: :mod:`phenotypic.gui._shared.tiles`. Re-exported under its historical
#: private name so callers importing it from here keep working.
_is_safe_path_component = is_safe_path_component


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
            the per-image cache directory.
    """
    bp = Blueprint("results_viewer_tiles", __name__, url_prefix=VIEWER_TILES_PREFIX)

    @bp.route("/<dataset>/<stem>.dzi")
    def manifest(dataset: str, stem: str) -> Response:
        """Serve the DZI XML manifest, generating the pyramid if needed."""
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            logger.warning(
                "Rejected tile manifest request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return _json_error("invalid dataset or stem", 404)

        # Capability gate that works for a standalone deliverables bundle
        # (which has no ``results/`` to anchor a dataset-dir existence check):
        # accept when either a full-res per-image HDF or a baked overlay PNG
        # exists for this image. The DZI tiler still needs the overlay PNG
        # (HDF -> DZI tiling lands in Task 9), so the overlay gate below stays
        # as the tiling prerequisite.
        if (
            output_root.hdf_path(dataset, stem) is None
            and not output_root.has_overlay(dataset, stem)
        ):
            return _json_error(
                f"no image source for {dataset!r}/{stem!r}", 404
            )
        if not output_root.has_overlay(dataset, stem):
            return _json_error(
                f"no overlay for {dataset!r}/{stem!r}", 404
            )

        dataset_cache_dir = output_root.cache_dir / dataset
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = output_root.overlay_path(dataset, stem)
        try:
            _dzi_tiler.tile(overlay_path, dataset_cache_dir)
        except Exception:
            logger.exception(
                "DZI tile generation failed: dataset=%s stem=%s",
                dataset,
                stem,
            )
            return _json_error("tile generation failed", 500)

        return send_from_directory(
            dataset_cache_dir,
            f"{stem}.dzi",
            mimetype="application/xml",
        )

    @bp.route("/<dataset>/<stem>_files/<int:level>/<filename>")
    def tile_endpoint(
        dataset: str, stem: str, level: int, filename: str
    ) -> Response:
        """Serve an individual tile PNG from the per-image cache."""
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

        tile_dir = (
            output_root.cache_dir / dataset / f"{stem}_files" / str(level)
        )
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

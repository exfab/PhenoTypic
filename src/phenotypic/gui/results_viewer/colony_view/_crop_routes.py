"""Flask blueprint serving per-colony overlay crops for the colony-view tab.

The colony-view grid renders one fixed-size thumbnail per colony by
pointing ``<img>`` tags at this blueprint's URL. The route looks up the
colony's centroid in :attr:`OutputRoot.master_df`, opens the dataset's
overlay PNG, and crops a ``size`` x ``size`` window around it via
:func:`phenotypic.gui.results_viewer.colony_view._cropper.crop_overlay`.
Path-traversal hardening mirrors :mod:`_tile_routes`; the same regex
guards both ``<dataset>`` and ``<stem>`` URL captures.
"""

from __future__ import annotations

import logging

import dash
import polars as pl
from flask import Blueprint, Response, request

from phenotypic.gui._config import COLONY_CROPS_URL_SEGMENT
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._tile_routes import _is_safe_path_component
from phenotypic.gui.results_viewer.colony_view._cropper import crop_overlay

logger = logging.getLogger(__name__)

#: Lower bound on the ``?size=`` query parameter. Smaller crops would not
#: hold a useful colony preview; rejecting them early avoids confused
#: callers.
_MIN_CROP_SIZE = 16

#: Upper bound on the ``?size=`` query parameter. Anything larger is
#: almost certainly the result of a bug in the caller (the colony-view
#: grid picks crops on the order of 64-512 px); cap it to avoid a 4k+
#: PNG allocation per request.
_MAX_CROP_SIZE = 4096

#: Sanity ceiling on the parsed ``<label>`` URL component. Real
#: ``ObjectLabel`` values are dense small integers; anything beyond a
#: billion is almost certainly malformed input.
_MAX_OBJECT_LABEL = 10**9


def register(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the per-colony crop route on ``app.server``.

    Exposes one route under the ``/crops`` URL prefix:

    * ``GET /crops/<dataset>/<stem>/<label>.png?size=<int>`` — returns a
      PNG-encoded ``size`` x ``size`` crop of the dataset's overlay PNG
      centered on the colony with ``ObjectLabel == label`` in image
      ``stem``.

    The route never mutates state and never writes to disk; the on-disk
    cache used by the DZI tile route is intentionally not reused here
    because per-colony crops are tiny and inexpensive to recompute.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory.
            Captured by closure and used to resolve overlay PNGs and the
            master measurements DataFrame.
    """
    bp = Blueprint("results_viewer_crops", __name__, url_prefix=f"/{COLONY_CROPS_URL_SEGMENT}")

    @bp.route("/<dataset>/<stem>/<label>.png")
    def crop_endpoint(dataset: str, stem: str, label: str) -> Response | tuple[str, int]:
        """Serve a single PNG colony crop for ``(dataset, stem, label)``."""
        # --- 1. Path-component validation --------------------------------
        if not _is_safe_path_component(dataset) or not _is_safe_path_component(stem):
            logger.warning(
                "Rejected crop request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return ("bad request: invalid dataset or stem", 400)

        # --- 2. Label parsing --------------------------------------------
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            logger.warning("Rejected crop request with non-numeric label: %r", label)
            return ("bad request: label must be an integer", 400)
        if label_int < 0 or label_int > _MAX_OBJECT_LABEL:
            logger.warning("Rejected crop request with out-of-range label: %d", label_int)
            return ("bad request: label out of range", 400)

        # --- 3. Size parsing ---------------------------------------------
        size = request.args.get("size", type=int)
        if size is None:
            return ("bad request: missing required ?size=<int>", 400)
        if size < _MIN_CROP_SIZE or size > _MAX_CROP_SIZE:
            return (
                f"bad request: size must be between {_MIN_CROP_SIZE} and "
                f"{_MAX_CROP_SIZE} (got {size})",
                400,
            )

        # --- 4. Lookup ----------------------------------------------------
        # Cast key columns explicitly so the comparison still matches when
        # the master frame stores Metadata_ImageFile as Categorical or
        # ObjectLabel as a narrower int type.
        try:
            row = (
                output_root.master_df.filter(
                    (pl.col(KEY_IMAGE_FILE).cast(pl.String) == stem)
                    & (pl.col(KEY_OBJECT_LABEL).cast(pl.Int64) == label_int)
                )
                .select(["Bbox_CenterRR", "Bbox_CenterCC"])
                .head(1)
            )
        except Exception:
            logger.exception(
                "Master DataFrame lookup failed for dataset=%s stem=%s label=%d",
                dataset,
                stem,
                label_int,
            )
            return ("internal error: master measurements lookup failed", 500)

        if row.is_empty():
            return (
                f"not found: no row for stem={stem!r} label={label_int}",
                404,
            )

        center_rr = float(row.get_column("Bbox_CenterRR")[0])
        center_cc = float(row.get_column("Bbox_CenterCC")[0])

        # --- 5. Overlay path ---------------------------------------------
        if not output_root.has_overlay(dataset, stem):
            return (
                f"not found: overlay not found for {dataset!r}/{stem!r}",
                404,
            )
        overlay_png = output_root.overlay_path(dataset, stem)

        # --- 6. Crop ------------------------------------------------------
        try:
            png_bytes = crop_overlay(overlay_png, center_rr, center_cc, size)
        except Exception:
            logger.exception(
                "Crop generation failed for dataset=%s stem=%s label=%d size=%d",
                dataset,
                stem,
                label_int,
                size,
            )
            return ("internal error: crop generation failed", 500)

        # --- 7. Response --------------------------------------------------
        response = Response(png_bytes, mimetype="image/png")
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer colony-crop routes under /crops for root=%s",
        output_root.root,
    )


__all__ = ["register"]

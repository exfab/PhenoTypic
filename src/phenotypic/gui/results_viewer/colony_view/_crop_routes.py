"""Flask blueprint serving per-colony overlay crops for the colony-view tab.

The colony-view grid renders one fixed-size thumbnail per colony by
pointing ``<img>`` tags at this blueprint's URL. The route is the shared
crop-route factory :func:`phenotypic.gui._shared.tiles.register_crop_route`
mounted under the colony's ``/crops`` segment; it looks up the colony's
centroid in :attr:`OutputRoot.master_df`, opens the dataset's overlay
PNG, and crops a ``size`` x ``size`` window around it. The QC review tab
mounts the same factory under its own segment.
"""

from __future__ import annotations

import dash

from phenotypic.gui._config import COLONY_CROPS_URL_SEGMENT
from phenotypic.gui._shared.tiles import register_crop_route
from phenotypic.gui.results_viewer._output_root import OutputRoot


def register(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the per-colony crop route under ``/crops`` on ``app.server``.

    Thin adapter over
    :func:`phenotypic.gui._shared.tiles.register_crop_route` pinned to the
    colony view's :data:`COLONY_CROPS_URL_SEGMENT`. Exposes:

    * ``GET /crops/<dataset>/<stem>/<label>.png?size=<int>`` — a
      PNG-encoded ``size`` x ``size`` crop of the dataset's overlay PNG
      centered on the colony with ``Object_Label == label`` in image
      ``stem``.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory.
    """
    register_crop_route(app, output_root, COLONY_CROPS_URL_SEGMENT)


__all__ = ["register"]

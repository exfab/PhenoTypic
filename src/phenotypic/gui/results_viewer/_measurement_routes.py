"""One column of one image's embedded measurement table, as JSON.

The per-object measurements now live *inside* each ``*.ome.zarr`` store, at
``tables/measurements/table.parquet``. The obvious way to get them to a
browser would be to let the existing ``/zarr/`` byte route serve that file.
It deliberately cannot, and this blueprint exists so that stays true.

**Why not widen the byte route.** ``readable_roots_for`` derives its
allow-list from the store's own ``attributes.phenotypic`` block, which names
series and label paths and never ``tables/``; a round-2 security review rated
the first revision's bypass **Major**. The viewer is unauthenticated and the
documented Open OnDemand recipe is ``--host 0.0.0.0`` on a shared cluster
(``gui_hub.md:116, :124``), so "anything that can reach the port" is the
audience. Adding ``tables/`` to the allow-list would reverse that decision,
ship ~130 columns to display one, and put a Parquet reader in the browser
bundle.

**The narrowing is also just smaller.** A real table from the migration-test
run is 71 KB for its ~130 columns; one column of it is ~2 KB.

**``column`` is a closed value set, not a path.** It is checked against the
store's own ``measurement_columns`` -- in
:func:`~phenotypic.sdk_.read_embedded_measurement_column`, *before* the
Parquet is opened -- so an unknown name is a 400 that never reaches the
filesystem and the parameter cannot be used to probe for columns a store does
not have.

The error contract matches ``/zarr/`` deliberately, because the two surfaces
must agree about one store:

* absent store, or a store carrying **no** ``tables`` descriptor -> **404**.
  An absent descriptor is a NORMAL state -- a ``--mode process`` run never
  measures -- so this is "nothing to show", never "measurement pending".
* a store this build cannot decode -> **422** with the store's own message.
* a column the store does not declare, or one that is not numeric -> **400**.

``require_readable_store`` raises ``FileNotFoundError``, ``KeyError`` **and**
``ValueError``. ``KeyError`` is not an ``OSError``, so it is named explicitly
or a store with no ``phenotypic`` block yields a 500 -- and with ``--debug``
plus ``--host 0.0.0.0`` an unhandled exception is the Werkzeug interactive
debugger.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import dash
from flask import Blueprint, Response, abort, jsonify, request

from phenotypic.gui._config import VIEWER_MEASUREMENTS_PREFIX
from phenotypic.gui._shared.tiles import (
    StoreUnreadable,
    _readable_block,
    is_safe_path_component,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import read_embedded_measurement_column

logger = logging.getLogger(__name__)


def register_measurement_routes(
    app: dash.Dash, output_root: OutputRoot
) -> None:
    """Mount the per-image measurement JSON route on ``app.server``.

    Exposes ``GET /measurements/<dataset>/<stem>?column=<name>``, answering
    with the column keyed by ``Object_Label`` plus the range it spans.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory. Captured
            by closure and used to resolve each image's store directory.
    """
    blueprint = Blueprint(
        "results_viewer_measurements",
        __name__,
        url_prefix=VIEWER_MEASUREMENTS_PREFIX,
    )

    @blueprint.route("/<dataset>/<stem>")
    def measurement_column(dataset: str, stem: str) -> Response:
        """Serve one measurement column of one image's embedded table."""
        if not is_safe_path_component(dataset) or not is_safe_path_component(
            stem
        ):
            abort(400)

        column = request.args.get("column", type=str)
        if not column:
            abort(400, description="query parameter 'column' is required")

        store: Path | None = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)

        try:
            # Gate the schema version through the SAME helper ``/zarr/`` and
            # ``crop_colony`` use, so all three agree about one store. It runs
            # first because the reader's own version refusal is a bare
            # ``ValueError``, which is indistinguishable at this boundary from
            # "the store does not declare this column" -- a 400, not a 422.
            _readable_block(store)
            values = read_embedded_measurement_column(store, column)
        except StoreUnreadable as exc:
            # 422, NOT 404 -- matching ``/zarr/`` and ``crop_colony``. A store
            # this build cannot decode is a run-wide, actionable condition;
            # 404 would tell the user "no such image", which is false.
            logger.error(
                "Unreadable store for %s/%s: %s", dataset, stem, exc
            )
            abort(422, description=str(exc))
        except (OSError, KeyError):
            # No root ``zarr.json`` (a promote in flight), no ``phenotypic``
            # block, or -- the routine case -- no ``tables`` descriptor at
            # all. ``KeyError`` is NOT an ``OSError`` and must be named.
            abort(404)
        except ValueError:
            # The store does not declare this column. The Parquet was never
            # opened; the allow-list check happens first.
            abort(400, description=f"unknown measurement column: {column}")
        except TypeError as exc:
            # A declared column that holds strings (``ColorLab_MedoidColorHex``).
            # There is no scale over hex triplets, so this is the caller's
            # error, not the store's.
            abort(400, description=str(exc))

        finite = [
            value
            for value in values.values()
            if value is not None
            and not math.isnan(value)
            and not math.isinf(value)
        ]
        payload = {
            "column": column,
            # JSON object keys are strings; the client re-parses them as the
            # integer ``Object_Label`` the store joined on.
            "values": {
                str(label): (
                    value
                    if value is not None
                    and not math.isnan(value)
                    and not math.isinf(value)
                    else None
                )
                for label, value in values.items()
            },
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "n": len(values),
        }
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.server.register_blueprint(blueprint)
    logger.debug(
        "Registered results viewer measurement route under %s for root=%s",
        VIEWER_MEASUREMENTS_PREFIX,
        output_root.root,
    )

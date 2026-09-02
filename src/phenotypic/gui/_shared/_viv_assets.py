"""Serve the ONE vendored Viv bundle + facade to a second Dash sub-app.

The results viewer owns the vendored artifacts on disk
(``results_viewer/_assets/viv/viv-bundle.min.js`` and
``results_viewer/_assets/viv_viewer.js``) and Dash's own asset walk mounts
them for that app. The builder's node-preview pane needs the same two files,
and **committing a second ~2.6 MB copy is the thing this module exists to
prevent**: two artifacts under one build recipe (``tools/viv-bundle/``), with
no npm in CI to catch them drifting apart.

**Why a Flask blueprint and not Dash's ``assets_folder``.** Dash takes exactly
**one** ``assets_folder`` per app, and ``builder/assets/`` already holds
``builder.js``, ``preview.js``, ``builder.css``, ``cytoscape-dagre.min.js``,
``point_picker.js``, ``palette_dnd.js``, ``viewport_ops.js`` and
``wire_drawing.js``. Repointing it at the results viewer's ``_assets/`` would
drop all eight. This closes spec section 10's OQ4.

Consumers add the two URLs to ``dash.Dash(external_scripts=...)``, joined to
their own ``url_prefix`` the way :data:`SHARED_LOGO_PATH` is -- the hub's
``DispatcherMiddleware`` strips the mount prefix before Flask routes, so the
blueprint mounts at :data:`VIV_ASSETS_PREFIX` on every server while the
browser sees ``/builder/_viv/...``.

**Load order does not matter.** ``viv_viewer.js`` reads ``window.__vivBundle``
inside ``ready()``, on the first call from a Dash callback, never at module
scope -- which is precisely why it survives Dash loading it *before* the
bundle in the results viewer.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Flask, send_from_directory

#: URL-prefix path where the blueprint mounts on each Flask server.
VIV_ASSETS_PREFIX = "/_viv"

#: Layout-friendly paths, joined with a sub-app's ``url_prefix``. The bundle
#: is listed first only for readability; see the load-order note above.
VIV_BUNDLE_PATH = "_viv/viv-bundle.min.js"
VIV_FACADE_PATH = "_viv/viv_viewer.js"

_BLUEPRINT_NAME = "phenotypic_gui_viv_assets"

#: Source of truth on disk: the results viewer's own asset directory. Read
#: from there rather than copied, so there is one artifact.
_VIV_ASSET_DIR = (
    Path(__file__).resolve().parent.parent / "results_viewer" / "_assets"
)

#: ``filename`` -> path relative to :data:`_VIV_ASSET_DIR`. An allow-list
#: rather than a ``<path:filename>`` passthrough: this blueprint's whole job
#: is two files, and a passthrough would expose the results viewer's entire
#: asset tree from an unrelated sub-app.
_SERVED: dict[str, str] = {
    "viv-bundle.min.js": "viv/viv-bundle.min.js",
    "viv_viewer.js": "viv_viewer.js",
}


def _build_blueprint() -> Blueprint:
    bp = Blueprint(_BLUEPRINT_NAME, __name__)

    @bp.route("/<filename>")
    def _serve(filename: str):
        relative = _SERVED.get(filename)
        if relative is None:
            return ("not found", 404)
        return send_from_directory(_VIV_ASSET_DIR, relative)

    return bp


def register_viv_assets(server: Flask) -> None:
    """Register the vendored-Viv blueprint on the given Flask server.

    Idempotent: a second call on the same server is a no-op rather than a
    ``Blueprint`` name collision, matching
    :func:`~phenotypic.gui._shared._blueprint.register_shared_static`.

    Args:
        server: The sub-app's Flask server.
    """
    if _BLUEPRINT_NAME in server.blueprints:
        return
    server.register_blueprint(_build_blueprint(), url_prefix=VIV_ASSETS_PREFIX)


def viv_script_urls(url_prefix: str) -> list[str]:
    """Browser-visible URLs of the two vendored scripts, in load order.

    Args:
        url_prefix: The sub-app's mount-point prefix (``"/"`` standalone,
            ``"/builder/"`` under the hub).

    Returns:
        ``[bundle_url, facade_url]``, ready for ``external_scripts``.
    """
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return [f"{base}{VIV_BUNDLE_PATH}", f"{base}{VIV_FACADE_PATH}"]

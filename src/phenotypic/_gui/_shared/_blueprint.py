"""Flask blueprint serving the shared GUI static directory.

Each Dash sub-app registers this blueprint on its own ``app.server`` so
the same SVG file is served under every mount point's URL prefix. The
canonical logo lives in :mod:`phenotypic._assets` (``logos/``); layouts
reference it via :data:`SHARED_LOGO_PATH` joined to the sub-app's
``url_prefix``.
"""
from __future__ import annotations

from flask import Blueprint, Flask, send_from_directory

from phenotypic._assets import logos_dir

_STATIC_DIR = logos_dir()

#: URL-prefix path where the blueprint mounts on each Flask server.
SHARED_BLUEPRINT_PREFIX = "/_shared"

#: Layout-friendly path to the dashboard logo. Layouts join this with
#: their ``url_prefix`` so the URL resolves under both standalone and
#: dispatcher-mounted launches.
SHARED_LOGO_PATH = "_shared/dashboard_logo.svg"

_BLUEPRINT_NAME = "phenotypic_gui_shared"


def _build_blueprint() -> Blueprint:
    bp = Blueprint(_BLUEPRINT_NAME, __name__)

    @bp.route("/<path:filename>")
    def _serve(filename: str):
        return send_from_directory(_STATIC_DIR, filename)

    return bp


def register_shared_static(server: Flask) -> None:
    """Register the shared-static blueprint on the given Flask server.

    Idempotent: if a blueprint with the same name is already registered
    (e.g. a previous sub-app rebuild reused the same Flask server), the
    call is a no-op rather than raising.
    """
    if _BLUEPRINT_NAME in server.blueprints:
        return
    server.register_blueprint(_build_blueprint(), url_prefix=SHARED_BLUEPRINT_PREFIX)

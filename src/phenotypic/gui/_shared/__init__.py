"""Shared static assets used across every GUI sub-app.

The dashboard logo is identical in every Dash sub-app (builder, results
viewer, run console, analysis). Rather than vendoring four physical
copies into each app's assets folder, the file lives once at
``phenotypic/_assets/logos/dashboard_logo.svg`` and a tiny Flask blueprint
registered on each sub-app's server (and on the shell) serves the file
under ``/_shared/<filename>``.

Layouts reference the URL via
``f"{url_prefix}{SHARED_LOGO_PATH}"`` so the URL resolves correctly under
both standalone (``url_prefix="/"``) and dispatcher-mounted launches
(``url_prefix="/builder/"`` etc.).

The same one-artifact-many-mounts rule covers the vendored Viv bundle and
its facade -- see :mod:`phenotypic.gui._shared._viv_assets`, which serves the
results viewer's two files to any other sub-app that renders through
``window.phenotypicViv``.
"""
from phenotypic.gui._shared._blueprint import (
    SHARED_BLUEPRINT_PREFIX,
    SHARED_LOGO_PATH,
    register_shared_static,
)
from phenotypic.gui._shared._viv_assets import (
    VIV_ASSETS_PREFIX,
    VIV_BUNDLE_PATH,
    VIV_FACADE_PATH,
    register_viv_assets,
    viv_script_urls,
)

__all__ = [
    "SHARED_BLUEPRINT_PREFIX",
    "SHARED_LOGO_PATH",
    "VIV_ASSETS_PREFIX",
    "VIV_BUNDLE_PATH",
    "VIV_FACADE_PATH",
    "register_shared_static",
    "register_viv_assets",
    "viv_script_urls",
]

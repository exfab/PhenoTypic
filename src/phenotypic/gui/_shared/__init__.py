"""Shared static assets used across every GUI sub-app.

The dashboard logo is identical in every Dash sub-app (builder, results
viewer, run console, analysis). Rather than vendoring four physical
copies into each app's assets folder, the file lives once at
``_shared/_static/dashboard_logo.svg`` and a tiny Flask blueprint
registered on each sub-app's server (and on the shell) serves the file
under ``/_shared/<filename>``.

Layouts reference the URL via
``f"{url_prefix}{SHARED_LOGO_PATH}"`` so the URL resolves correctly under
both standalone (``url_prefix="/"``) and dispatcher-mounted launches
(``url_prefix="/builder/"`` etc.).
"""
from phenotypic.gui._shared._blueprint import (
    SHARED_BLUEPRINT_PREFIX,
    SHARED_LOGO_PATH,
    register_shared_static,
)

__all__ = [
    "SHARED_BLUEPRINT_PREFIX",
    "SHARED_LOGO_PATH",
    "register_shared_static",
]

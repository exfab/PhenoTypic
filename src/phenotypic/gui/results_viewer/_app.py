"""Dash app factory for the results viewer.

Builds a configured :class:`dash.Dash` instance with its tile-serving
and colony-crop Flask blueprints mounted, the validated
:class:`~phenotypic.gui.results_viewer._output_root.OutputRoot` plus the
curation-state
:class:`~phenotypic.gui.results_viewer._filtered_state.FilteredMeasurements`
stashed on ``app.server.config``, the layout assembled by
:func:`~phenotypic.gui.results_viewer._layout.build_app_layout`, and
all callbacks registered via
:func:`~phenotypic.gui.results_viewer._callbacks.register_callbacks`.

Phase 5 additions:
    * Optional ``output_root`` — when ``None`` the factory returns a
      Dash app whose layout is :func:`._layout.build_empty_state_layout`
      and which has NO blueprints, NO callbacks, and NO ``filtered_state``
      on ``app.server.config``. The hub uses this path to reach the
      ``/results/`` page before the user has selected a CLI output.
    * ``url_prefix`` — Mount-point prefix passed to ``dash.Dash`` as
      both ``requests_pathname_prefix`` and ``routes_pathname_prefix``,
      and stashed on ``app.server.config["pheno_url_prefix"]`` so
      callbacks (notably the colony-grid crop URLs) can construct
      hub-aware URLs at request time.
    * ``window.__phenotypicAppPrefix`` — injected via
      ``app.index_string`` so ``results_viewer.js`` can build DZI tile
      URLs that include the mount prefix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import Input, Output, State

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_MEASUREMENT_SCHEMA,
    CFG_OUTPUT_ROOT,
    CFG_QC_AUGMENTED_FRAME,
    CFG_URL_PREFIX,
    MOUNT_HOME,
    SANDBOX_API_VIEWER_OUTPUT_ROOT,
    TITLE_VIEWER,
)
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui._design import COLOR_BLUE, COLOR_SURFACE, inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui.results_viewer import _ids as ids, _tile_routes
from phenotypic.gui.results_viewer._callbacks import register_callbacks
from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
from phenotypic.gui.results_viewer._layout import (
    build_app_layout,
    build_empty_state_layout,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view import _crop_routes as colony_crop_routes
from phenotypic.gui.shell._ids import SHELL_SIDEBAR_SELECTION_STORE

logger = logging.getLogger(__name__)


def _index_string_with_prefix(url_prefix: str) -> str:
    """Return a Dash ``index_string`` template that injects the URL prefix.

    The injected ``<script>`` defines ``window.__phenotypicAppPrefix`` so
    ``results_viewer.js`` can build hub-aware URLs for DZI tiles and the
    vendored OpenSeadragon assets.

    ``url_prefix`` is escaped before embedding inside the JS string
    literal:

        * ``\\`` -> ``\\\\`` and ``"`` -> ``\\"`` keep the JS string
          well-formed.
        * ``</`` -> ``<\\/`` prevents a stray ``</script>`` from
          terminating the inline script tag (forward-slash escapes are
          legal in JS string literals).

    In practice only the hub composer sets the prefix and the values are
    static literals (``"/"``, ``"/results/"``); the escaping is defence
    in depth against future callers passing user-controlled values.
    """
    safe_prefix = (
        url_prefix.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("</", "<\\/")
    )
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "    <head>\n"
        "        {%metas%}\n"
        "        <title>{%title%}</title>\n"
        "        {%favicon%}\n"
        "        {%css%}\n"
        f'        <script>window.__phenotypicAppPrefix = "{safe_prefix}";</script>\n'
        "    </head>\n"
        "    <body>\n"
        "        {%app_entry%}\n"
        "        <footer>\n"
        "            {%config%}\n"
        "            {%scripts%}\n"
        "            {%renderer%}\n"
        "        </footer>\n"
        "    </body>\n"
        "</html>"
    )


def create_app(
    output_root: Optional[OutputRoot] = None,
    *,
    url_prefix: str = MOUNT_HOME,
) -> dash.Dash:
    """Build a Dash application instance for the results viewer.

    Args:
        output_root: Validated, read-only handle on a CLI output
            directory (see
            :meth:`phenotypic.gui.results_viewer._output_root.OutputRoot.discover`).
            ``None`` triggers the empty-state pathway: the factory skips
            blueprint registration, ``FilteredMeasurements.load``, and
            callback registration; ``app.layout`` is the empty-state
            placeholder.
        url_prefix: Mount-point prefix. Defaults to ``"/"`` (standalone
            launcher); the hub composer passes ``"/results/"``. Set as
            ``requests_pathname_prefix``/``routes_pathname_prefix`` on
            the Dash constructor and stashed on
            ``app.server.config["pheno_url_prefix"]``.

    Returns:
        A configured :class:`dash.Dash` instance whose ``app.run(...)``
        is the responsibility of the caller.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title=TITLE_VIEWER,
        # Pin to the in-package directory so the assets ship correctly
        # regardless of the user's CWD at launch.
        assets_folder=str(Path(__file__).parent / "_assets"),
        # See builder/_app.py for the rationale: when mounted under
        # the hub's DispatcherMiddleware the dispatcher strips the
        # mount prefix from PATH_INFO, so Dash must route at "/".
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )

    # Inject window.__phenotypicAppPrefix so results_viewer.js can build
    # mount-aware URLs for DZI tiles and OSD assets.
    app.index_string = _index_string_with_prefix(url_prefix)

    inject_design_tokens(app)
    register_shared_static(app.server)

    app.server.config[CFG_URL_PREFIX] = url_prefix

    if output_root is None:
        app.layout = build_empty_state_layout()
        _register_empty_state_callbacks(app, url_prefix=url_prefix)
        logger.debug(
            "Results viewer built in empty-state mode (url_prefix=%s)",
            url_prefix,
        )
        return app

    app.server.config[CFG_OUTPUT_ROOT] = output_root

    _tile_routes.register(app, output_root)

    filtered_state = FilteredMeasurements.load(output_root.root, output_root.master_df)
    app.server.config[CFG_FILTERED_STATE] = filtered_state
    colony_crop_routes.register(app, output_root)

    # MeasurementSchema cache shared by the Heatmap tab (and a future
    # QC tab) - lazily built once per app instance. Idempotent: do not
    # clobber an existing instance e.g. when the analysis sub-app has
    # already populated the key.
    if app.server.config.get(CFG_MEASUREMENT_SCHEMA) is None:
        app.server.config[CFG_MEASUREMENT_SCHEMA] = MeasurementSchema(
            output_root=Path(output_root.root)
        )
    # QC tab's augmented-frame cache starts empty; Wave E's QC writer
    # fills it on its first card refresh. The heatmap render callback
    # gracefully falls back to the plain filtered frame until then.
    app.server.config.setdefault(CFG_QC_AUGMENTED_FRAME, None)

    app.layout = build_app_layout(output_root, filtered_state, url_prefix=url_prefix)
    register_callbacks(app, output_root)

    return app


def _register_empty_state_callbacks(app: dash.Dash, *, url_prefix: str) -> None:
    """Wire the empty-state hand-off banner.

    Two callbacks:

    1. **Mirror selection -> banner.** A serverside callback watches
       :data:`SHELL_SIDEBAR_SELECTION_STORE` (mounted on this app's
       chrome wrapper) and toggles the banner's visibility, label, and
       Open-button ``disabled`` flag based on whether the selection
       has the ``is_cli_output`` capability.

    2. **Open button -> POST + redirect.** A clientside callback fetches
       ``/sandbox/api/viewer/output-root`` with the selection's
       ``abs_path`` (or rel ``path`` as fallback). On success the
       browser navigates to ``url_prefix`` so :class:`_ViewerProxy`
       resolves a freshly-built loaded viewer; on 4xx the JSON
       ``error`` is rendered into the inline error slot.
    """
    @app.callback(
        Output(ids.EMPTY_HANDOFF_BANNER, "style"),
        Output(ids.EMPTY_HANDOFF_LABEL, "children"),
        Output(ids.EMPTY_HANDOFF_OPEN_BUTTON, "disabled"),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
    )
    def _populate_handoff_banner(
        selection: "dict | None",
    ) -> "tuple":
        hidden = {"display": "none"}
        visible = {
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "marginTop": "1rem",
            "padding": "0.5rem 0.75rem",
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BLUE}",
            "borderRadius": "6px",
        }
        if not selection or not isinstance(selection, dict):
            return hidden, "(none)", True
        path = selection.get("path") or ""
        if not path:
            return hidden, "(none)", True
        caps = selection.get("capabilities") or {}
        is_cli_output = bool(caps.get("is_cli_output"))
        return visible, path, not is_cli_output

    # Clientside POST + navigate. Uses ``window.fetch`` with the prefix
    # so it works under any DispatcherMiddleware mount. On success the
    # callback calls ``window.location.assign(prefix)`` directly (forces
    # a full reload even though the URL is unchanged), which is what
    # ``_ViewerProxy`` needs to resolve the freshly-built session. On
    # 4xx the JSON ``error`` is rendered into the inline error slot.
    app.clientside_callback(
        """
        async function(n_clicks, selection) {
            if (!n_clicks || !selection) {
                return window.dash_clientside.no_update;
            }
            const path = selection.path;
            if (!path) { return "No sidebar selection."; }
            try {
                const resp = await fetch(
                    "__PHENO_API_OUTPUT_ROOT__",
                    {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({path: path}),
                    }
                );
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok) {
                    return (data && data.error) || ("HTTP " + resp.status);
                }
                window.location.assign(__PHENO_VIEWER_PREFIX__);
                return "";
            } catch (err) {
                return String(err);
            }
        }
        """.replace("__PHENO_API_OUTPUT_ROOT__", SANDBOX_API_VIEWER_OUTPUT_ROOT).replace("__PHENO_VIEWER_PREFIX__", repr(url_prefix)),
        Output(ids.EMPTY_HANDOFF_ERROR, "children"),
        Input(ids.EMPTY_HANDOFF_OPEN_BUTTON, "n_clicks"),
        State(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        prevent_initial_call=True,
    )


__all__ = ["create_app"]

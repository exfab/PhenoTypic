"""Shell chrome layout helpers.

The chrome wrap layer is the single place that knows about both the shell
(top bar + sidebar + release button) and the wrapped tool (pre-existing
Dash app). It is BOTH a layout mutator AND a callback registrar:

    * Layout mutation — wraps the tool's existing ``app.layout`` in chrome
      so tab navigation + sidebar + RSS readout appear above/around it.
    * Callback registration — every Dash app instance has its own callback
      dispatch table, so chrome callbacks (RSS interval, sidebar refresh,
      release-button click) must be registered on each wrapped app
      separately. This is not a workaround; it's how Dash multi-app
      composition works.

Phase 3 ships the standalone shell variant: ``create_app(sandbox)`` returns
a Dash app whose body is the home pane already wrapped in chrome. Phase 5
generalises this to wrap the builder + viewer + run console mounts via
``DispatcherMiddleware``.
"""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._config import (
    MOUNT_ANALYSIS,
    MOUNT_BUILDER,
    MOUNT_HOME,
    MOUNT_RUN,
    MOUNT_TUNE,
    MOUNT_VIEWER,
    RSS_INTERVAL_MS,
    SSH_TUNNEL_HINT,
    TITLE_HUB,
)
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui.shell._callbacks import register_chrome_callbacks
from phenotypic.gui.shell._ids import (
    SHELL_HELP_BUTTON,
    SHELL_HELP_MODAL,
    SHELL_MAIN_PANE,
    SHELL_ROOT_LABEL,
    SHELL_RSS_INTERVAL,
    SHELL_RSS_LABEL,
    SHELL_SIDEBAR_COLLAPSE_BUTTON,
    SHELL_SIDEBAR_COLLAPSE_STORE,
    SHELL_SOURCE_IMAGE_ROOT_CLEAR,
    SHELL_SOURCE_IMAGE_ROOT_LABEL,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
    SHELL_TAB_ANALYSIS,
    SHELL_TAB_BUILDER,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_TUNE,
    SHELL_TAB_VIEWER,
    SHELL_TOP_BAR,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._sidebar import build_sidebar

__all__ = ["build_top_bar", "build_help_modal", "wrap_in_chrome"]


# Shell chrome CSS lives in ``shell/_assets/shell.css``. Dash auto-emits a
# ``<link>`` to it from the shell app's ``<head>``, but sub-apps mounted under
# DispatcherMiddleware (``/builder/``, ``/results/``, ``/run/``) don't share
# the shell's assets folder, so their pages don't load the chrome CSS. To
# avoid path/dispatcher gymnastics, read shell.css once at import time and
# inline it into every chrome-wrapped layout below. The bytes are tiny
# (~6 KB) so the duplication is cheap.
_SHELL_CSS = (Path(__file__).parent / "_assets" / "shell.css").read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

#: Tab anchors. Plain ``html.A`` elements so the navigation crosses
#: WSGI mounts cleanly once Phase 5 wires the sub-apps.
#:
#: TODO(reverse-proxy): these are absolute paths starting with ``/``;
#: that's correct when the hub is served at the URL root (the v1 SSH-
#: tunnel deployment). If a future deployment puts the hub behind a
#: reverse proxy with a prefix (e.g. ``/phenotypic/``), thread that
#: prefix through ``wrap_in_chrome`` and prepend it here. Tracked
#: against the cloud-deploy hook in ``shell/_sandbox.py``.
_TAB_HREFS = {
    SHELL_TAB_HOME: MOUNT_HOME,
    SHELL_TAB_BUILDER: MOUNT_BUILDER,
    SHELL_TAB_VIEWER: MOUNT_VIEWER,
    SHELL_TAB_RUN: MOUNT_RUN,
    SHELL_TAB_TUNE: MOUNT_TUNE,
    SHELL_TAB_ANALYSIS: MOUNT_ANALYSIS,
}

_TAB_LABELS = {
    SHELL_TAB_HOME: "Home",
    SHELL_TAB_BUILDER: "Pipelines",
    SHELL_TAB_VIEWER: "Viewer",
    SHELL_TAB_RUN: "Run",
    SHELL_TAB_TUNE: "Tune",
    SHELL_TAB_ANALYSIS: "Analysis",
}

#: Display order for the top-bar tab nav. The sequence follows the user
#: workflow: land on Home, compose a pipeline in Builder, tune its
#: parameters in Tune, execute it from Run, inspect the output in Viewer,
#: and run downstream stats in Analysis.
TAB_DISPLAY_ORDER: tuple[str, ...] = (
    SHELL_TAB_HOME,
    SHELL_TAB_BUILDER,
    SHELL_TAB_TUNE,
    SHELL_TAB_RUN,
    SHELL_TAB_VIEWER,
    SHELL_TAB_ANALYSIS,
)


def build_top_bar(
    *,
    active_tab: str,
    sandbox: SandboxRoot,
) -> html.Header:
    """Build the top-bar element shown above every tool's main pane.

    Args:
        active_tab: One of the ``SHELL_TAB_*`` constants. The matching anchor
            gets a ``shell-tab-active`` class.
        sandbox: Sandbox root (echoed in the top-bar label).

    Returns:
        ``html.Header`` containing the title, root display, tab nav, RSS
        readout, and help button.
    """
    return html.Header(
        [
            html.Div(
                [
                    dbc.Button(
                        "«",
                        id=SHELL_SIDEBAR_COLLAPSE_BUTTON,
                        size="sm",
                        color="link",
                        n_clicks=0,
                        title="Toggle file explorer",
                        className="shell-sidebar-collapse-button",
                    ),
                    html.Strong(TITLE_HUB, className="shell-title"),
                    html.Span(
                        f"root: {sandbox.root}",
                        id=SHELL_ROOT_LABEL,
                        className="shell-root-label",
                        title=str(sandbox.root),
                    ),
                    html.Span(
                        "source: unset",
                        id=SHELL_SOURCE_IMAGE_ROOT_LABEL,
                        className="shell-source-label",
                        title="No source image root selected",
                    ),
                    dbc.Button(
                        "x",
                        id=SHELL_SOURCE_IMAGE_ROOT_CLEAR,
                        size="sm",
                        color="link",
                        n_clicks=0,
                        title="Clear source image root",
                        className="shell-source-clear-button",
                    ),
                ],
                className="shell-top-bar-left",
            ),
            html.Nav(
                [
                    _build_tab(tab_id, active_tab=active_tab)
                    for tab_id in TAB_DISPLAY_ORDER
                ],
                className="shell-tab-nav",
            ),
            html.Div(
                [
                    html.Span(
                        "RSS …",
                        id=SHELL_RSS_LABEL,
                        className="shell-rss-readout",
                    ),
                    dbc.Button(
                        "?",
                        id=SHELL_HELP_BUTTON,
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        className="shell-help-button",
                    ),
                ],
                className="shell-top-bar-right",
            ),
        ],
        id=SHELL_TOP_BAR,
        className="shell-top-bar",
    )


def build_help_modal() -> dbc.Modal:
    """Help modal triggered by the top-bar ``?`` button.

    Content: SSH-tunnel reminder, classifier-cache nuke command, link to
    docs, and the v1 cloud-deploy non-goal note (per spec).
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(f"{TITLE_HUB} — Help")),
            dbc.ModalBody(
                [
                    html.H6("SSH tunnel pattern"),
                    html.P(
                        "If the GUI is running on a remote cluster, "
                        "forward its port to your local machine:"
                    ),
                    html.Pre(
                        SSH_TUNNEL_HINT,
                        className="shell-help-pre",
                    ),
                    html.Hr(),
                    html.H6("Refresh sidebar / clear classifier cache"),
                    html.P(
                        "Click the Refresh button in the sidebar to bust "
                        "the capability classifier's cache. New files dropped "
                        "into the sandbox after that will be re-classified."
                    ),
                    html.Hr(),
                    html.H6("Cloud deployment"),
                    html.P(
                        "v1 of the GUI is single-user, frozen-at-launch — "
                        "designed for SSH-tunnelled workstation use. "
                        "Multi-user / cloud deployment is a non-goal for v1.",
                        className="text-muted",
                    ),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id={"type": "shell-help-close", "scope": "modal"},
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id=SHELL_HELP_MODAL,
        is_open=False,
        size="lg",
    )


# ---------------------------------------------------------------------------
# wrap_in_chrome
# ---------------------------------------------------------------------------

def wrap_in_chrome(
    app,  # type: ignore[no-untyped-def]
    *,
    active_tab: str,
    sandbox: SandboxRoot,
) -> None:
    """Wrap ``app.layout`` in the shell chrome and register chrome callbacks.

    Args:
        app: A :class:`dash.Dash` instance whose layout is fully assigned.
            ``app.layout`` is mutated in place — caller does not need to
            reassign it.
        active_tab: ``SHELL_TAB_*`` constant identifying which tab is
            highlighted in the top bar (the chrome doesn't navigate;
            the host calling ``wrap_in_chrome`` knows which mount this is).
        sandbox: Sandbox root.

    Side effects:
        * Sets ``app.layout`` to a new ``html.Div`` containing the chrome
          + the original body.
        * Registers chrome callbacks on ``app`` (RSS interval, help-modal
          toggle, sidebar refresh placeholder).
    """
    body = app.layout

    # Inject the shell chrome CSS into the page's ``<head>``. Sub-apps mounted
    # under DispatcherMiddleware don't share the shell's assets folder, so
    # Dash's auto-discovery only finds their own CSS files. Splicing the
    # shell stylesheet into the index template here makes the chrome render
    # styled on every mount (including the shell itself, where it duplicates
    # the auto-emitted ``<link>`` — harmless).
    inject_design_tokens(app)
    _inject_shell_css(app)

    app.layout = html.Div(
        [
            build_top_bar(active_tab=active_tab, sandbox=sandbox),
            html.Div(
                [
                    build_sidebar(sandbox),
                    html.Main(
                        body,
                        id=SHELL_MAIN_PANE,
                        className="shell-main-pane",
                    ),
                ],
                className="shell-body",
            ),
            build_help_modal(),
            dcc.Interval(id=SHELL_RSS_INTERVAL, interval=RSS_INTERVAL_MS, n_intervals=0),
            # Persists across mounts: each Dash instance reads the same
            # localStorage key and the clientside callback toggles the
            # ``shell-sidebar-collapsed`` class on the outer ``.shell-root``.
            dcc.Store(
                id=SHELL_SIDEBAR_COLLAPSE_STORE,
                storage_type="local",
                data=False,
            ),
            dcc.Store(
                id=SHELL_SOURCE_IMAGE_ROOT_STORE,
                storage_type="local",
                data=None,
            ),
        ],
        className="shell-root",
    )

    register_chrome_callbacks(app, sandbox)


def _inject_shell_css(app) -> None:  # type: ignore[no-untyped-def]
    """Splice shell.css into ``app.index_string`` (idempotent).

    Called from :func:`wrap_in_chrome`. We modify the index template
    rather than emitting a ``<link>`` because (a) Dash strips most
    layout-level head elements and (b) routing an absolute ``href``
    cleanly across DispatcherMiddleware mounts requires more plumbing
    than the inline CSS bytes are worth.
    """
    marker = "<!-- phenotypic-shell-css -->"
    if marker in app.index_string:
        return
    style_block = f"{marker}\n<style>\n{_SHELL_CSS}\n</style>"
    # Dash's default template ends the head with ``{%css%}`` immediately
    # before ``</head>``. Insert just after ``{%css%}`` so any sub-app
    # CSS still loads before our overrides (we want chrome styles to win).
    if "{%css%}" in app.index_string:
        app.index_string = app.index_string.replace(
            "{%css%}", "{%css%}\n" + style_block, 1
        )
    else:  # pragma: no cover - defensive: custom templates without {%css%}
        app.index_string = app.index_string.replace(
            "</head>", style_block + "\n</head>", 1
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_tab(tab_id: str, *, active_tab: str) -> html.A:
    href = _TAB_HREFS[tab_id]
    label = _TAB_LABELS[tab_id]
    classes = ["shell-tab"]
    if tab_id == active_tab:
        classes.append("shell-tab-active")
    return html.A(
        label,
        id=tab_id,
        href=href,
        className=" ".join(classes),
    )

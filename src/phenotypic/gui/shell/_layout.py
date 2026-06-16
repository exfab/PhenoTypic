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
from typing import NamedTuple
from urllib.parse import quote

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from lucide import lucide_icon  # type: ignore[import-untyped]

from phenotypic.gui._config import (
    MOUNT_ANALYSIS,
    MOUNT_BROWSE,
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
    SHELL_METADATA_CSV_STORE,
    SHELL_HELP_BUTTON,
    SHELL_HELP_MODAL,
    SHELL_MAIN_PANE,
    SHELL_ROOT_LABEL,
    SHELL_RSS_INTERVAL,
    SHELL_RSS_LABEL,
    SHELL_SETTINGS_BUTTON,
    SHELL_SETTINGS_INPUT_FOLDER_CLEAR,
    SHELL_SETTINGS_INPUT_FOLDER_PICK,
    SHELL_SETTINGS_METADATA_CSV_CLEAR,
    SHELL_SETTINGS_METADATA_CSV_LABEL,
    SHELL_SETTINGS_METADATA_CSV_PICK,
    SHELL_SETTINGS_POPOVER,
    SHELL_SIDEBAR_COLLAPSE_BUTTON,
    SHELL_SIDEBAR_COLLAPSE_STORE,
    SHELL_SOURCE_IMAGE_ROOT_LABEL,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
    SHELL_TAB_ANALYSIS,
    SHELL_TAB_BROWSE,
    SHELL_TAB_BUILDER,
    SHELL_TAB_GROUP_PIPELINE,
    SHELL_TAB_GROUP_RESULTS,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_TUNE,
    SHELL_TAB_VIEWER,
    SHELL_TOP_BAR,
    TUNE_PIPELINE_PATH_STORE,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._sidebar import build_sidebar
from phenotypic.gui.shell._source_picker import (
    build_metadata_csv_picker_modal,
    build_source_picker_modal,
)

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
_SHELL_ICON_STROKE = "#f8fafc"


def _lucide_img(icon_name: str, *, class_name: str) -> html.Img:
    """Return a lucide icon as a Dash image component."""
    svg = lucide_icon(
        icon_name,
        cls=class_name,
        width=18,
        height=18,
        stroke_width=2,
    )
    svg = svg.replace('stroke="currentColor"', f'stroke="{_SHELL_ICON_STROKE}"')
    return html.Img(
        src=f"data:image/svg+xml;utf8,{quote(svg)}",
        alt="",
        className=class_name,
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
    SHELL_TAB_BROWSE: MOUNT_BROWSE,
    SHELL_TAB_BUILDER: MOUNT_BUILDER,
    SHELL_TAB_VIEWER: MOUNT_VIEWER,
    SHELL_TAB_RUN: MOUNT_RUN,
    SHELL_TAB_TUNE: MOUNT_TUNE,
    SHELL_TAB_ANALYSIS: MOUNT_ANALYSIS,
}

_TAB_LABELS = {
    SHELL_TAB_HOME: "Home",
    SHELL_TAB_BROWSE: "Browse",
    SHELL_TAB_BUILDER: "Builder",
    SHELL_TAB_VIEWER: "Viewer",
    SHELL_TAB_RUN: "Run",
    SHELL_TAB_TUNE: "Tune",
    SHELL_TAB_ANALYSIS: "Analysis",
}


class _NavGroup(NamedTuple):
    """One dropdown tab group in the top-bar nav.

    A group renders as a ``dbc.DropdownMenu`` whose toggle carries
    ``label`` and ``id == group_id`` and whose menu items are the
    ``members`` (each a ``SHELL_TAB_*`` id). The toggle gets the gold
    ``shell-tab-group-active`` treatment whenever the active mount is one
    of ``members``.
    """

    label: str
    group_id: str
    members: tuple[str, ...]


#: Structured top-bar nav model. Entries are either a **leaf** tab id
#: (a bare ``SHELL_TAB_*`` string, rendered as a plain anchor) or a
#: :class:`_NavGroup` dropdown. The sequence follows the user workflow:
#: land on Home, then the **Pipeline** group (compose in Builder, tune in
#: Tune, execute in Run), then the **Results** group (inspect output in
#: Viewer, run downstream stats in Analysis).
NAV_MODEL: tuple["str | _NavGroup", ...] = (
    SHELL_TAB_HOME,
    SHELL_TAB_BROWSE,
    _NavGroup(
        "Pipeline",
        SHELL_TAB_GROUP_PIPELINE,
        (SHELL_TAB_BUILDER, SHELL_TAB_TUNE, SHELL_TAB_RUN),
    ),
    _NavGroup(
        "Results",
        SHELL_TAB_GROUP_RESULTS,
        (SHELL_TAB_VIEWER, SHELL_TAB_ANALYSIS),
    ),
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
                ],
                className="shell-top-bar-left",
            ),
            html.Nav(
                [
                    _build_nav_entry(entry, active_tab=active_tab)
                    for entry in NAV_MODEL
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
                        _lucide_img("settings", class_name="shell-settings-icon"),
                        id=SHELL_SETTINGS_BUTTON,
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        title="GUI settings",
                        className="shell-settings-button",
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


def build_settings_popover(sandbox: SandboxRoot) -> dbc.Popover:
    """Build the global GUI settings popover."""
    return dbc.Popover(
        [
            dbc.PopoverHeader("GUI settings"),
            dbc.PopoverBody(
                [
                    html.Div(
                        [
                            html.Div("Sandbox root", className="shell-settings-key"),
                            html.Div(
                                str(sandbox.root),
                                id=SHELL_ROOT_LABEL,
                                className="shell-settings-value shell-settings-path",
                                title=str(sandbox.root),
                            ),
                        ],
                        className="shell-settings-row",
                    ),
                    html.Div(
                        [
                            html.Div("Input folder", className="shell-settings-key"),
                            html.Div(
                                "source: unset",
                                id=SHELL_SOURCE_IMAGE_ROOT_LABEL,
                                className="shell-settings-value shell-settings-path",
                                title="No source image root selected",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Pick",
                                        id=SHELL_SETTINGS_INPUT_FOLDER_PICK,
                                        size="sm",
                                        color="primary",
                                        outline=True,
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Clear",
                                        id=SHELL_SETTINGS_INPUT_FOLDER_CLEAR,
                                        size="sm",
                                        color="secondary",
                                        outline=True,
                                        n_clicks=0,
                                    ),
                                ],
                                className="shell-settings-actions",
                            ),
                        ],
                        className="shell-settings-row",
                    ),
                    html.Div(
                        [
                            html.Div("Metadata CSV", className="shell-settings-key"),
                            html.Div(
                                "metadata: unset",
                                id=SHELL_SETTINGS_METADATA_CSV_LABEL,
                                className="shell-settings-value shell-settings-path",
                                title="No metadata CSV selected",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Pick",
                                        id=SHELL_SETTINGS_METADATA_CSV_PICK,
                                        size="sm",
                                        color="primary",
                                        outline=True,
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Clear",
                                        id=SHELL_SETTINGS_METADATA_CSV_CLEAR,
                                        size="sm",
                                        color="secondary",
                                        outline=True,
                                        n_clicks=0,
                                    ),
                                ],
                                className="shell-settings-actions",
                            ),
                        ],
                        className="shell-settings-row",
                    ),
                ],
                className="shell-settings-body",
            ),
        ],
        id=SHELL_SETTINGS_POPOVER,
        target=SHELL_SETTINGS_BUTTON,
        is_open=False,
        placement="bottom-end",
        className="shell-settings-popover",
    )


def build_help_modal() -> dbc.Modal:
    """Help modal triggered by the top-bar ``?`` button.

    Content: SSH-tunnel reminder, classifier-cache nuke command, link to
    docs, and the v1 cloud-deploy non-goal note (per spec).
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(f"{TITLE_HUB} -- Help")),
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
                        "v1 of the GUI is single-user, frozen-at-launch -- "
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
            build_settings_popover(sandbox),
            build_source_picker_modal(sandbox),
            build_metadata_csv_picker_modal(sandbox),
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
            dcc.Store(
                id=SHELL_METADATA_CSV_STORE,
                storage_type="local",
                data=None,
            ),
            dcc.Store(
                id=TUNE_PIPELINE_PATH_STORE,
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

def _build_nav_entry(
    entry: "str | _NavGroup", *, active_tab: str
):  # type: ignore[no-untyped-def]
    """Render one :data:`NAV_MODEL` entry — a leaf tab or a dropdown group."""
    if isinstance(entry, _NavGroup):
        return _build_tab_group(entry, active_tab=active_tab)
    return _build_tab(entry, active_tab=active_tab)


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


def _build_tab_group(
    group: _NavGroup, *, active_tab: str
) -> dbc.DropdownMenu:
    """Render a dropdown tab group (``Pipeline`` / ``Results``).

    The toggle is a pure menu opener (no own destination); the member
    items are real cross-mount anchors (``external_link=True`` forces a
    full-page navigation across the WSGI mounts, matching the leaf-tab
    behaviour). The toggle carries ``shell-tab-group-active`` whenever the
    active mount is one of the group's members, and the matching member
    item renders with Bootstrap's ``active`` class.
    """
    is_active = active_tab in group.members
    toggle_classes = ["shell-tab", "shell-tab-group"]
    if is_active:
        toggle_classes.append("shell-tab-group-active")
    return dbc.DropdownMenu(
        label=group.label,
        id=group.group_id,
        nav=True,
        in_navbar=True,
        toggleClassName=" ".join(toggle_classes),
        className="shell-tab-group-menu",
        children=[
            dbc.DropdownMenuItem(
                _TAB_LABELS[member],
                id=member,
                href=_TAB_HREFS[member],
                active=(member == active_tab),
                external_link=True,
                className="shell-tab-group-item",
            )
            for member in group.members
        ],
    )

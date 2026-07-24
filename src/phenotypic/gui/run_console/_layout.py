"""Run console layout — form + iframe panel + log tail + recent runs.

Stitches together the four panes of the Run console:

    * Left column: :func:`._form.build_form` (pickers + mode + advanced
      + slurm + actions).
    * Right column top: dashboard ``<iframe>`` panel (placeholder before
      a run starts; ``src`` rewritten to
      ``/runs/<rel>/deliverables/dashboard.html`` once a run is live).
    * Right column bottom: live log tail driven by
      :class:`~phenotypic.gui.run_console._runner.LocalRunner` snapshots.
    * Bottom: Recent Runs panel scanned via
      :func:`._recent_runs.scan_recent_runs`.

Public surface for the leader's ``_app.py`` swap:
    :func:`build_run_console_layout` — accepts ``sandbox``, ``registry``
    (and optional ``runner``) and returns a fully-mounted ``html.Div``
    with all required stores, modals, intervals, and toast.
"""
from __future__ import annotations

import logging
from typing import List

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._config import MOUNT_HOME
from phenotypic.gui._shared import SHARED_LOGO_PATH
from phenotypic.gui.run_console import _ids as ids
from phenotypic.gui.run_console._form import (
    build_form,
    build_input_picker_modal,
    build_output_picker_modal,
    build_pipeline_picker_modal,
)
from phenotypic.gui.run_console._recent_runs import RecentRunRow, scan_recent_runs
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["build_run_console_layout", "render_recents_table"]


# ---------------------------------------------------------------------------
# Sub-section builders
# ---------------------------------------------------------------------------


def _iframe_panel() -> html.Div:
    """Build the right-hand iframe panel with placeholder + iframe."""
    placeholder = html.Div(
        [
            html.H4(
                "Dashboard preview",
                className="run-console-iframe-empty-title",
            ),
            html.P(
                "Pick a pipeline + input directory + output directory,"
                " then click Run to start a pipeline. The live dashboard"
                " will appear here.",
                className="run-console-iframe-empty-body",
            ),
        ],
        id=ids.RC_IFRAME_PLACEHOLDER,
        className="run-console-iframe-empty",
    )

    iframe = html.Iframe(
        id=ids.RC_IFRAME,
        src="about:blank",
        className="run-console-iframe",
        style={"display": "none"},
    )

    status_banner = html.Div(
        id=ids.RC_STATUS_BANNER,
        className="run-console-status-banner",
        children="(no active run)",
    )
    refresh_button = dbc.Button(
        "Refresh",
        id=ids.RC_BTN_REFRESH_DASHBOARD,
        color="secondary",
        outline=True,
        size="sm",
        className="mb-2",
    )

    return html.Div(
        [status_banner, refresh_button, placeholder, iframe],
        id=ids.RC_IFRAME_PANEL,
        className="run-console-iframe-panel",
    )


def _log_tail() -> html.Div:
    """Build the log-tail ``<pre>`` block + interval timer."""
    return html.Div(
        [
            html.Div("Log tail", className="run-console-log-header"),
            html.Pre(
                "(no log yet)",
                id=ids.RC_LOG_TAIL,
                className="run-console-log-tail",
            ),
            dcc.Interval(
                id=ids.RC_INTERVAL_LOG,
                interval=1000,
                disabled=True,
            ),
            dcc.Interval(
                id=ids.RC_INTERVAL_DASHBOARD_POLL,
                interval=2000,
                disabled=True,
                max_intervals=-1,
            ),
        ],
        className="run-console-log-section",
    )


def render_recents_table(rows: List[RecentRunRow]) -> List:
    """Render the Recent Runs table body.

    Each row is a clickable ``html.Tr`` with a pattern-matching id of
    ``{"type": "rc-recents-row", "rel_path": <rel>}`` so a single
    callback can subscribe via ``ALL`` and dispatch on the clicked row.

    Args:
        rows: Output of :func:`._recent_runs.scan_recent_runs`.

    Returns:
        List of :class:`dash.html.Tr` rows ready to assign to the
        :data:`ids.RC_RECENTS_BODY` table body.
    """
    if not rows:
        return [
            html.Tr(
                [
                    html.Td(
                        "(no runs yet -- start one above)",
                        colSpan=4,
                        className="run-console-recents-empty",
                    )
                ]
            )
        ]

    out: List = []
    for row in rows:
        status_cls = f"run-console-recents-status run-console-recents-status-{row.status}"
        cells = [
            html.Td(row.rel_path, className="run-console-recents-cell-path"),
            html.Td(row.mode, className="run-console-recents-cell-mode"),
            html.Td(row.status, className=status_cls),
            html.Td(
                "yes" if row.has_dashboard else "no",
                className="run-console-recents-cell-dash",
            ),
        ]
        out.append(
            html.Tr(
                cells,
                id=ids.recents_row_id(row.rel_path),
                className="run-console-recents-row",
                n_clicks=0,
            )
        )
    return out


def _recents_panel(
    sandbox: SandboxRoot, registry: RunRegistry
) -> html.Div:
    """Build the Recent Runs panel."""
    initial_rows = scan_recent_runs(sandbox, registry=registry)
    header = html.Tr(
        [
            html.Th("Output"),
            html.Th("Mode"),
            html.Th("Status"),
            html.Th("Dashboard"),
        ]
    )
    table = dbc.Table(
        [
            html.Thead(header),
            html.Tbody(
                render_recents_table(initial_rows),
                id=ids.RC_RECENTS_BODY,
            ),
        ],
        bordered=False,
        hover=True,
        responsive=True,
        size="sm",
        className="run-console-recents-table",
    )
    return html.Div(
        [
            html.Div(
                "Recent Runs",
                className="run-console-recents-header",
            ),
            table,
        ],
        id=ids.RC_RECENTS,
        className="run-console-recents",
    )


def _stores() -> html.Div:
    """Build all run-console ``dcc.Store`` widgets in one Div."""
    return html.Div(
        [
            dcc.Store(id=ids.RC_STORE_FORM_STATE, data={}),
            dcc.Store(id=ids.RC_STORE_ACTIVE_RUN_ID, data=None),
            dcc.Store(id=ids.RC_STORE_ACTIVE_REL_PATH, data=None),
            dcc.Store(id=ids.RC_STORE_PIPELINE_PATH, data=None),
            dcc.Store(id=ids.RC_STORE_INPUT_DIR, data=None),
            dcc.Store(id=ids.RC_STORE_OUTPUT_DIR, data=None),
            dcc.Store(id=ids.RC_STORE_RECENTS_REFRESH, data=0),
        ]
    )


def _toast() -> dbc.Toast:
    """Build the floating notification toast."""
    return dbc.Toast(
        id=ids.RC_TOAST,
        header="Run console",
        is_open=False,
        dismissable=True,
        duration=5000,
        icon="primary",
        style={
            "position": "fixed",
            "top": 20,
            "right": 20,
            "minWidth": 320,
            "zIndex": 1080,
        },
    )


# ---------------------------------------------------------------------------
# Public layout factory
# ---------------------------------------------------------------------------


def build_run_console_layout(
    sandbox: SandboxRoot,
    *,
    registry: RunRegistry,
    runner: LocalRunner | None = None,
    url_prefix: str = MOUNT_HOME,
) -> html.Div:
    """Build the Run console layout.

    The leader's :func:`._app.create_app` swaps its placeholder layout
    for this one once Stream A + B land. Every component the callbacks
    register against is mounted here, including the three modal pickers
    (so the modals can stay closed but always-resolving).

    Args:
        sandbox: Frozen-at-launch sandbox; passed to the form and the
            recents panel.
        registry: Process-wide run registry; rehydrated for the recents
            panel and read by the log-tail / status callbacks.
        runner: Optional :class:`LocalRunner`. Reserved for future use
            (e.g. seeding "running now" state at first paint); the
            current callback layer reads the runner via
            ``app.server.config["pheno_runner"]`` instead.
        url_prefix: Mount-point prefix used to resolve the dashboard
            logo URL in the header. Defaults to ``MOUNT_HOME`` ("/")
            for standalone launches; the hub passes ``MOUNT_RUN``.

    Returns:
        A :class:`dash.html.Div` ready to assign to ``app.layout``.
    """
    del runner  # currently unused; reserved for symmetry with builder layout

    header = _header(url_prefix=url_prefix)
    form = build_form(sandbox)
    iframe_panel = _iframe_panel()
    log_section = _log_tail()
    recents_panel = _recents_panel(sandbox, registry)
    handoff_banner = _handoff_banner()
    modals = html.Div(
        [
            build_pipeline_picker_modal(sandbox),
            build_input_picker_modal(sandbox),
            build_output_picker_modal(sandbox),
        ]
    )

    main_row = dbc.Row(
        [
            dbc.Col(form, md=5, className="run-console-form-col-wrap"),
            dbc.Col(
                html.Div(
                    [iframe_panel, log_section],
                    className="run-console-right-col",
                ),
                md=7,
            ),
        ],
        className="g-3",
    )

    body = dbc.Container(
        [handoff_banner, main_row, recents_panel],
        fluid=True,
        className="run-console-container",
    )

    return html.Div(
        [_stores(), _toast(), modals, header, body],
        id=ids.RC_ROOT,
        className="run-console-root",
    )


def _header(*, url_prefix: str) -> html.Div:
    """Top-of-page header bar with dashboard logo and app title."""
    return html.Div(
        [
            html.Img(
                src=f"{url_prefix}{SHARED_LOGO_PATH}",
                alt="PhenoTypic",
                className="run-console-header__logo",
            ),
            html.H4("Run Console", className="run-console-header__title mb-0"),
        ],
        className="run-console-header",
    )


def _handoff_banner() -> html.Div:
    """Banner shown when ``SHELL_SIDEBAR_SELECTION_STORE`` carries a path.

    Renders four buttons — ``Set as pipeline`` / ``Set as input dir`` /
    ``Set as output dir`` / ``Dismiss`` — that route the sidebar's
    selection into the form's stores. The banner stays hidden when no
    selection is active; the consumer callback in ``_callbacks.py``
    flips ``style.display`` based on the store payload.
    """
    return html.Div(
        [
            html.Span(
                "Sidebar selection: ",
                className="run-console-handoff-prefix",
            ),
            html.Code(
                "(none)",
                id=ids.RC_HANDOFF_LABEL,
                className="run-console-handoff-label",
            ),
            dbc.Button(
                "Set as pipeline",
                id=ids.RC_HANDOFF_USE_PIPELINE,
                size="sm",
                color="primary",
                outline=True,
                disabled=True,
                n_clicks=0,
                className="ms-2",
            ),
            dbc.Button(
                "Set as input dir",
                id=ids.RC_HANDOFF_USE_INPUT,
                size="sm",
                color="primary",
                outline=True,
                disabled=True,
                n_clicks=0,
                className="ms-1",
            ),
            dbc.Button(
                "Set as output dir",
                id=ids.RC_HANDOFF_USE_OUTPUT,
                size="sm",
                color="primary",
                outline=True,
                disabled=True,
                n_clicks=0,
                className="ms-1",
            ),
            dbc.Button(
                "Dismiss",
                id=ids.RC_HANDOFF_DISMISS,
                size="sm",
                color="secondary",
                outline=True,
                n_clicks=0,
                className="ms-1",
            ),
        ],
        id=ids.RC_HANDOFF_BANNER,
        className="run-console-handoff-banner",
        style={"display": "none"},
    )

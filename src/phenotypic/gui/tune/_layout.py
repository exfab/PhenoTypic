"""Layout builders for the ``/tune/`` co-pilot.

The tune page is a four-view sub-tab stack — Monitor (live read), Curate
(shortlist + overlays), Space (search-space inference + export), and Launch
(render the ``run`` command). Only the active view is shown; the switch callback
(:mod:`._callbacks`) toggles which container carries the ``tune-view-hidden``
class. Each view dispatches to its own builder module (:func:`build_monitor_view`
here; ``_curate`` / ``_space`` / ``_launch`` for the rest).

The page is built as a **persistent shell** (:func:`build_layout`) — the
run-root store, the run-picker chrome (Bind-run button + label + note + modal),
and a swappable :data:`~phenotypic.gui.tune._ids.TUNE_PAGE_BODY` container — that
is identical whether or not a run is bound. When no run is bound the body holds a
pick-a-run prompt (:func:`build_empty_body`); when a run is bound (either at
construction or via the runtime bind callback) the body holds the four-view stack
(:func:`build_loaded_body`). Keeping the store + picker OUTSIDE the swappable body
is what lets the bind callback re-render only the body while the store it just
wrote survives — and lets every view's callbacks reach the store as ``State``
regardless of which sub-tab is active.

Every color / font / spacing value comes from
:mod:`phenotypic.gui._design` (via class names that reference the injected
CSS custom properties); this module hard-codes none.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import _nav
from phenotypic.gui.tune._run_picker import build_run_picker_modal, build_run_picker_row
from phenotypic.tune._strategies._config import STRATEGY_CHOICES

if TYPE_CHECKING:
    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.tune._run_root import TuneRunRoot

#: The default active sub-tab when a run is first opened.
_DEFAULT_VIEW: ids.SubTabName = "monitor"


def build_layout(
    root: "Optional[TuneRunRoot]" = None, *, sandbox: "Optional[SandboxRoot]" = None
) -> html.Div:
    """Render the persistent tune page shell.

    The shell is identical for the empty (``root is None``) and loaded states:
    the run-root store, the run-picker row + modal (so a run can be bound — or
    re-bound — at runtime), and the swappable :data:`ids.TUNE_PAGE_BODY`. Only
    the body differs — :func:`build_empty_body` (pick-a-run prompt) vs.
    :func:`build_loaded_body` (the four-view stack). The bind callback
    (:mod:`._callbacks`) swaps the body and writes the store; keeping both at the
    page root (outside the body) means the store the callback writes is not torn
    down when the body re-renders.

    Args:
        root: The validated tune output handle the page reads from, or ``None``
            to mount the empty state until a run is bound at runtime.
        sandbox: The frozen-at-launch sandbox root; threaded into the run picker
            AND the Curate view's Image Source picker. ``None`` degrades both
            pickers to a note (the standalone-without-sandbox path).

    Returns:
        The page body: the run-root store, the run-picker chrome, and the
        swappable page body (empty-state prompt or the four loaded views).
    """
    store_data = {"path": str(root.path)} if root is not None else None
    bound_path = str(root.path) if root is not None else None
    active_destination: _nav.Destination = "monitor" if root is not None else "setup"
    body = (
        build_loaded_body(root, sandbox=sandbox)
        if root is not None
        else build_empty_body()
    )
    children: list[Component] = [
        # Run-root descriptor (path only) the Monitor poll, the overlay-poll
        # self-heal, the Space/Launch callbacks, AND the bind callback all
        # read/write. Lives at the PAGE ROOT (not inside the Monitor sub-view or
        # the swappable body) so every view's callbacks can reach it as ``State``
        # regardless of the active sub-tab, and so a body swap never tears it down.
        dcc.Store(id=ids.TUNE_RUN_ROOT_STORE, data=store_data),
        dcc.Store(id=ids.TUNE_ACTIVE_DESTINATION_STORE, data=active_destination),
        dcc.Store(id=ids.TUNE_SETUP_PIPELINE_STORE, data=None),
        dcc.Store(id=ids.TUNE_SETUP_AUTHORED_SPEC_STORE, data=None),
        dcc.Store(id=ids.TUNE_RUN_ACTIVE_RECORD_STORE, data=None),
        html.Div(
            [
                html.Span("Tune run", className="tune-run-title"),
                build_run_picker_row(sandbox, bound_path=bound_path),
            ],
            id=ids.TUNE_RUN_HEADER,
            className="tune-run-header",
        ),
        _build_destination_row(active=active_destination),
        html.Div(
            [
                html.Div(
                    build_setup_view(),
                    id=_nav.destination_view_id("setup"),
                    className=_nav.destination_view_class(
                        "setup", active_destination
                    ),
                ),
                html.Div(
                    build_run_view_placeholder(),
                    id=_nav.destination_view_id("run"),
                    className=_nav.destination_view_class("run", active_destination),
                ),
                html.Div(
                    html.Div(body, id=ids.TUNE_PAGE_BODY, className="tune-page-body"),
                    id=_nav.destination_view_id("monitor"),
                    className=_nav.destination_view_class(
                        "monitor", active_destination
                    ),
                ),
            ],
            className="tune-destination-views",
        ),
    ]
    # The run-picker modal lives at the page root (next to the store) so it is
    # reachable in both the empty and loaded states; ``None`` sandbox omits it.
    if sandbox is not None:
        children.append(build_run_picker_modal(sandbox))
    return html.Div(children, id=ids.TUNE_PAGE, className="tune-page")


def _build_destination_row(active: _nav.Destination) -> html.Div:
    """Render the top-level Setup / Run / Monitor destination row."""
    return html.Div(
        [
            html.Button(
                _nav.DESTINATION_LABELS[name],
                id=_nav.destination_button_id(name),
                n_clicks=0,
                className=_nav.destination_button_class(name, active),
                disabled=_nav.destination_button_disabled(
                    name, pipeline_path=None
                ),
            )
            for name in _nav.DESTINATIONS
        ],
        id=ids.TUNE_DESTINATION_DRAWER,
        className="tune-destination-row",
    )


def build_setup_view() -> html.Div:
    """Render the initial Setup destination scaffold."""
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Pipeline"),
                    dcc.Input(
                        id=ids.TUNE_SETUP_PIPELINE_INPUT,
                        type="text",
                        debounce=True,
                        placeholder="Pipeline or tuning spec path",
                        className="tune-setup-path-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_SETUP_METADATA_INPUT,
                        type="text",
                        debounce=True,
                        placeholder="Metadata layout CSV/Parquet for QC scorer",
                        className="tune-setup-path-input",
                    ),
                    html.Div(
                        "Choose a pipeline and metadata layout to author a tune spec.",
                        id=ids.TUNE_SETUP_GATE,
                        className="tune-setup-note",
                    ),
                ],
                className="tune-setup-section",
            ),
            html.Div(
                [html.H3("Search space"), html.Div("Locked until pipeline is chosen.")],
                id=ids.TUNE_SETUP_SEARCH_SPACE,
                className="tune-setup-section tune-setup-locked",
            ),
            html.Div(
                [html.H3("Scorer"), html.Div("Locked until pipeline is chosen.")],
                id=ids.TUNE_SETUP_SCORER,
                className="tune-setup-section tune-setup-locked",
            ),
            html.Div(
                html.Button(
                    "Continue",
                    id=ids.TUNE_SETUP_CONTINUE,
                    disabled=True,
                    className="tune-setup-continue",
                ),
                id=ids.TUNE_SETUP_FOOTER,
                className="tune-setup-footer",
            ),
        ],
        className="tune-setup",
    )


def build_run_view_placeholder() -> html.Div:
    """Render the Run destination."""
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Inputs"),
                    dcc.Input(
                        id=ids.TUNE_RUN_IMAGES_OVERRIDE,
                        type="text",
                        debounce=True,
                        placeholder="Image source override (blank uses shared source)",
                        className="tune-run-input tune-run-input-wide",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_OUTPUT_DIR,
                        type="text",
                        debounce=True,
                        placeholder="Output directory",
                        className="tune-run-input tune-run-input-wide",
                    ),
                ],
                className="tune-run-section",
            ),
            html.Div(
                [
                    html.H3("Strategy"),
                    dcc.Dropdown(
                        id=ids.TUNE_RUN_STRATEGY,
                        options=[
                            {"label": choice, "value": choice}
                            for choice in STRATEGY_CHOICES
                        ],
                        value="tpe",
                        clearable=False,
                        className="tune-run-field",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_N_TRIALS,
                        type="number",
                        min=1,
                        step=1,
                        value=50,
                        className="tune-run-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_STORAGE_URL,
                        type="text",
                        debounce=True,
                        placeholder="Storage URL",
                        className="tune-run-input tune-run-input-wide",
                    ),
                ],
                className="tune-run-section",
            ),
            html.Div(
                [
                    html.H3("Compute"),
                    dcc.RadioItems(
                        id=ids.TUNE_RUN_MODE,
                        options=[
                            {"label": "Local", "value": "local"},
                            {"label": "SLURM", "value": "slurm"},
                        ],
                        value="local",
                        inline=True,
                        className="tune-run-mode",
                    ),
                    dcc.Checklist(
                        id=ids.TUNE_RUN_SCREEN,
                        options=[{"label": "Two-round screening", "value": "on"}],
                        value=[],
                        className="tune-run-mode",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_N_WORKERS,
                        type="number",
                        min=1,
                        step=1,
                        placeholder="Workers",
                        className="tune-run-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_SLURM_PARTITION,
                        type="text",
                        debounce=True,
                        placeholder="SLURM partition",
                        className="tune-run-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_SLURM_MEM,
                        type="text",
                        debounce=True,
                        placeholder="SLURM mem",
                        className="tune-run-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_SLURM_TIME,
                        type="text",
                        debounce=True,
                        placeholder="SLURM time",
                        className="tune-run-input",
                    ),
                ],
                className="tune-run-section",
            ),
            html.Div(
                [
                    html.H3("Evaluation"),
                    dcc.Input(
                        id=ids.TUNE_RUN_HELD_OUT_FRACTION,
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        placeholder="Held-out fraction",
                        className="tune-run-input",
                    ),
                    dcc.Input(
                        id=ids.TUNE_RUN_CV_GROUP,
                        type="text",
                        debounce=True,
                        placeholder="CV group",
                        className="tune-run-input",
                    ),
                ],
                className="tune-run-section",
            ),
            html.Div(id=ids.TUNE_RUN_PREFLIGHT, className="tune-run-note"),
            html.Code(id=ids.TUNE_RUN_COMMAND, className="tune-launch-command"),
            html.Div(
                [
                    html.Button(
                        "Deploy",
                        id=ids.TUNE_RUN_DEPLOY,
                        n_clicks=0,
                        disabled=True,
                        className="tune-setup-continue",
                    ),
                    html.Span(id=ids.TUNE_RUN_STATUS, className="tune-run-note"),
                ],
                className="tune-setup-footer",
            ),
        ],
        className="tune-run",
    )


def build_empty_body() -> html.Div:
    """Render the pick-a-run body shown when no tune run is bound.

    A short prompt to bind a tune output via the run picker, plus the four
    sub-tab buttons — so the sub-tab surface is stable across the empty/loaded
    states and the switch callback can register against the same button IDs. It
    carries no view containers, no poll, and no figures; those arrive when
    :func:`build_loaded_body` replaces it on bind.

    Returns:
        The empty-state body.
    """
    return html.Div(
        [
            dcc.Interval(id=ids.TUNE_STUDY_POLL, interval=3000, n_intervals=0),
            _build_subtab_row(),
            _build_monitor_registry_shell(),
            html.Div(
                [
                    dcc.Graph(id=ids.TUNE_OBJECTIVE_FIGURE),
                    dcc.Graph(id=ids.TUNE_IMPORTANCE_FIGURE),
                    html.Span(
                        "--",
                        id=ids.TUNE_GAP_BADGE,
                        className="tune-gap-badge tune-gap-badge-stable",
                    ),
                    html.Div(id=ids.TUNE_TRIALS_TABLE),
                    html.Div(id=ids.TUNE_MONITOR_NOTE),
                ],
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.P(
                        "No tune run is open. Use Bind run above to select a "
                        "finished or in-flight tune output directory and monitor "
                        "its trials, curate its shortlist, and launch the winner."
                    ),
                ],
                className="tune-view tune-empty-state",
            ),
        ],
        className="tune-body-inner",
    )


def build_loaded_body(
    root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]" = None
) -> html.Div:
    """Render the four-view body for the bound run ``root``.

    The inner body the bind callback swaps into :data:`ids.TUNE_PAGE_BODY` (and
    the construction-time loaded state renders directly): the active-view store,
    the sub-tab button row, and the four view containers (only Monitor visible).
    The Monitor view carries the poll + figures (see :func:`build_monitor_view`).

    Args:
        root: The validated tune output handle the views read from.
        sandbox: The frozen-at-launch sandbox root; threaded into the Curate
            view's Image Source picker. ``None`` degrades that picker to a note.

    Returns:
        The loaded four-view body.
    """
    return html.Div(
        [
            dcc.Store(id=ids.TUNE_ACTIVE_VIEW_STORE, data=_DEFAULT_VIEW),
            _build_subtab_row(active=_DEFAULT_VIEW),
            html.Div(
                [
                    _build_view_container(name, root, sandbox=sandbox)
                    for name in ids.SUBTAB_ORDER
                ],
                className="tune-views",
            ),
        ],
        className="tune-body-inner",
    )


def _build_subtab_row(active: ids.SubTabName | None = None) -> html.Div:
    """Render the row of sub-tab buttons; ``active`` gets the active class."""
    return html.Div(
        [
            html.Button(
                ids.SUBTAB_LABELS[name],
                id=ids.subtab_button_id(name),
                n_clicks=0,
                className=ids.subtab_button_class(name, active),
            )
            for name in ids.SUBTAB_ORDER
        ],
        className="tune-subtab-row",
    )


def _build_view_container(
    name: ids.SubTabName, root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]"
) -> html.Div:
    """Render one view container; non-active containers start hidden."""
    return html.Div(
        _build_view_body(name, root, sandbox=sandbox),
        id=ids.view_container_id(name),
        className=ids.view_container_class(name, _DEFAULT_VIEW),
    )


def _build_view_body(
    name: ids.SubTabName, root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]"
) -> Component:
    """Dispatch to the per-view body builder (placeholders for Space/Launch)."""
    if name == "monitor":
        return build_monitor_view(root)
    if name == "curate":
        from phenotypic.gui.tune._curate import build_curate_view

        return build_curate_view(root, sandbox=sandbox)
    if name == "launch":
        from phenotypic.gui.tune._launch import build_launch_view

        return build_launch_view(root)
    if name == "space":
        from phenotypic.gui.tune._space import build_space_view

        return build_space_view(root)
    # Defensive: every SUBTAB_ORDER name now has a builder; an unknown name
    # (a future sub-tab added to ids without a view) renders a stub.
    return html.P(f"{name} -- view not yet implemented.")


def build_monitor_view(root: "TuneRunRoot") -> html.Div:
    """Render the Monitor view body.

    Chunk A ships the structural shell (3-second poll + figure / badge / table
    slots + the degrade note + the multi-objective Pareto card). The poll
    callback (:mod:`._callbacks`) re-reads the study and fills the figures.

    Args:
        root: The bound tune output handle.

    Returns:
        The Monitor view body.
    """
    from phenotypic.gui.tune._study_read import monitor_pareto_visible

    children: list[Component] = [
        dcc.Interval(id=ids.TUNE_STUDY_POLL, interval=3000, n_intervals=0),
        # NOTE: TUNE_RUN_ROOT_STORE is hoisted to the page root (see
        # ``build_layout``) so the Monitor poll, the overlay-poll self-heal, and
        # the Space/Launch callbacks can all reach it regardless of the active
        # sub-tab. The Monitor poll reads it as ``State`` from there.
        _build_monitor_registry_shell(),
        html.Div(
            [
                html.Span("Winner stability:"),
                html.Span(
                    "--",
                    id=ids.TUNE_GAP_BADGE,
                    className="tune-gap-badge tune-gap-badge-stable",
                ),
            ],
            className="tune-monitor-full",
        ),
        html.Div(
            [
                dcc.Graph(id=ids.TUNE_OBJECTIVE_FIGURE),
                dcc.Graph(id=ids.TUNE_IMPORTANCE_FIGURE),
            ],
            className="tune-monitor-grid",
        ),
        html.Div(id=ids.TUNE_TRIALS_TABLE, className="tune-monitor-full"),
        html.Div(
            "",
            id=ids.TUNE_MONITOR_NOTE,
            className="tune-monitor-note",
        ),
    ]

    pareto_style = {} if monitor_pareto_visible(root) else {"display": "none"}
    children.append(
        html.Div(
            html.P("Pareto front (multi-objective run)."),
            id=ids.TUNE_PARETO_CARD,
            className="tune-monitor-full",
            style=pareto_style,
        )
    )

    return html.Div(children, className="tune-monitor")


def _build_monitor_registry_shell() -> html.Div:
    """Render Monitor run-switcher, live-status, cancel, and export zones."""
    return html.Div(
        [
            dcc.Store(id=ids.TUNE_MONITOR_ACTIVE_RUN_STORE, data=None),
            html.Div(id=ids.TUNE_MONITOR_SWITCHER, className="tune-monitor-full"),
            html.Div(
                [
                    html.Div(
                        "Local log tail will appear here.",
                        id=ids.TUNE_MONITOR_LOCAL_LOG,
                        className="tune-monitor-live-card",
                    ),
                    html.Div(
                        "SLURM fleet polling state will appear here.",
                        id=ids.TUNE_MONITOR_SLURM_FLEET,
                        className="tune-monitor-live-card",
                    ),
                    dcc.ConfirmDialogProvider(
                        id=ids.TUNE_MONITOR_CANCEL_CONFIRM,
                        message="Cancel this Local tune run?",
                        children=html.Button(
                            "Cancel Local",
                            id=ids.TUNE_MONITOR_CANCEL,
                            n_clicks=0,
                            disabled=True,
                            className="tune-setup-continue",
                        ),
                    ),
                    html.Span(
                        "",
                        id=ids.TUNE_MONITOR_CANCEL_NOTE,
                        className="tune-run-note",
                    ),
                ],
                className="tune-monitor-full tune-monitor-live",
            ),
            html.Div(
                [
                    html.Button(
                        "Export best pipeline",
                        id=ids.TUNE_MONITOR_EXPORT,
                        n_clicks=0,
                        className="tune-setup-continue",
                    ),
                    html.Span(
                        "",
                        id=ids.TUNE_MONITOR_EXPORT_NOTE,
                        className="tune-run-note",
                    ),
                ],
                id=ids.TUNE_MONITOR_RESULT_ZONE,
                className="tune-monitor-full tune-monitor-result",
            ),
        ]
    )


__all__ = [
    "build_layout",
    "build_empty_body",
    "build_loaded_body",
    "build_monitor_view",
]


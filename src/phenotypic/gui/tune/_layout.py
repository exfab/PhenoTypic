"""Layout builders for the ``/tune/`` co-pilot.

The tune page is a four-view sub-tab stack — Monitor (live read), Curate
(shortlist + overlays), Space (search-space view), and Launch (apply the
tuned winner). Only the active view is shown; the switch callback
(:mod:`._callbacks`) toggles which container carries the ``tune-view-hidden``
class. Chunk A ships the Monitor view; the other three render placeholders the
later chunks fill in.

``build_layout(root)`` renders the full page (header + sub-tab button row +
the four view containers + the supporting stores/poll). ``build_empty_state_layout``
renders the pick-a-run prompt the hub shows before a tune run is bound.

Every color / font / spacing value comes from
:mod:`phenotypic.gui._design` (via class names that reference the injected
CSS custom properties); this module hard-codes none.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.tune._run_root import TuneRunRoot

#: The default active sub-tab when a run is first opened.
_DEFAULT_VIEW: ids.SubTabName = "monitor"

#: Placeholder copy for the not-yet-built views (Chunk C-ii fills Space in).
_PLACEHOLDER_COPY: dict[ids.SubTabName, str] = {
    "space": "Space — search-space view (coming in Chunk C-ii).",
}


def build_empty_state_layout() -> html.Div:
    """Render the pick-a-run prompt shown when no tune run is bound.

    The hub mounts ``/tune/`` before the user has selected a run directory, so
    the factory's ``root is None`` path renders this placeholder: a short
    prompt to pick a tune output from the sidebar. It carries no stores, no
    poll, and no callbacks — just the prompt and the four sub-tab buttons (so
    the surface is stable across the empty/loaded states and the switch
    callback can register against the same button IDs).

    Returns:
        The empty-state page body.
    """
    return html.Div(
        [
            html.Div(
                "Pick a tune run",
                id=ids.TUNE_RUN_HEADER,
                className="tune-run-header",
            ),
            _build_subtab_row(),
            html.Div(
                [
                    html.P(
                        "No tune run is open. Select a finished or in-flight "
                        "tune output directory from the sidebar to monitor its "
                        "trials, curate its shortlist, and launch the winner."
                    ),
                ],
                className="tune-view tune-empty-state",
            ),
        ],
        id=ids.TUNE_PAGE,
        className="tune-page",
    )


def build_layout(
    root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]" = None
) -> html.Div:
    """Render the full tune page for the bound run ``root``.

    Args:
        root: The validated tune output handle the page reads from.
        sandbox: The frozen-at-launch sandbox root; threaded into the Curate
            view's sandbox-bounded Image Source picker. ``None`` degrades the
            Curate picker to a note.

    Returns:
        The page body: header, sub-tab button row, the four view containers
        (only Monitor visible), and the supporting active-view store. The
        Monitor view carries the poll + figures (see :func:`build_monitor_view`).
    """
    return html.Div(
        [
            dcc.Store(id=ids.TUNE_ACTIVE_VIEW_STORE, data=_DEFAULT_VIEW),
            html.Div(
                f"Tune run: {root.path}",
                id=ids.TUNE_RUN_HEADER,
                className="tune-run-header",
                title=str(root.path),
            ),
            _build_subtab_row(active=_DEFAULT_VIEW),
            html.Div(
                [
                    _build_view_container(name, root, sandbox=sandbox)
                    for name in ids.SUBTAB_ORDER
                ],
                className="tune-views",
            ),
        ],
        id=ids.TUNE_PAGE,
        className="tune-page",
    )


def _build_subtab_row(active: ids.SubTabName | None = None) -> html.Div:
    """Render the row of sub-tab buttons; ``active`` gets the active class."""
    return html.Div(
        [
            html.Button(
                ids.SUBTAB_LABELS[name],
                id=ids.subtab_button_id(name),
                n_clicks=0,
                className=_subtab_class(name, active),
            )
            for name in ids.SUBTAB_ORDER
        ],
        className="tune-subtab-row",
    )


def _subtab_class(name: ids.SubTabName, active: ids.SubTabName | None) -> str:
    """The class string for a sub-tab button (active gets the highlight)."""
    classes = ["tune-subtab"]
    if name == active:
        classes.append("tune-subtab-active")
    return " ".join(classes)


def _build_view_container(
    name: ids.SubTabName, root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]"
) -> html.Div:
    """Render one view container; non-active containers start hidden."""
    classes = ["tune-view"]
    if name != _DEFAULT_VIEW:
        classes.append("tune-view-hidden")
    return html.Div(
        _build_view_body(name, root, sandbox=sandbox),
        id=ids.view_container_id(name),
        className=" ".join(classes),
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
    return html.P(_PLACEHOLDER_COPY[name])


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
        # Run-root descriptor the poll callback re-reads from each tick. The
        # path is the only field the callback needs to re-discover the run.
        dcc.Store(id=ids.TUNE_RUN_ROOT_STORE, data={"path": str(root.path)}),
        html.Div(
            [
                html.Span("Winner stability:"),
                html.Span(
                    "—",
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


__all__ = [
    "build_empty_state_layout",
    "build_layout",
    "build_monitor_view",
]


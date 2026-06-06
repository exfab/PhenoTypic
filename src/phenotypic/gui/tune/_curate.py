"""Curate view layout — shortlist + A/B overlays + Image Source picker (B-ii).

The Curate view lets a user audit a tuning candidate's segmentation on a chosen
plate. It is composed of:

* the **Image Source** picker (B-IMG) — a sandbox-bounded directory picker
  setting where ``<Image Source>/<plate_name>`` plates load from, pre-filled from
  the bound run's ``run.json`` ``images_dir``;
* the **shortlist** (B4) — one clickable card per
  :func:`~phenotypic.gui.tune._study_read.shortlist` trial, pinned to slot A or B;
* the **A/B segment** (B4/B5) — a Side-by-side ↔ Difference toggle over two
  ``go.Image`` graphs (side-by-side) and one (difference);
* the **plate picker** + **winner bar** — which plate to render, and "Set as
  winner" writing ``deliverables/best_pipeline.json``.

Overlays are rendered on demand on a background pool (the
:class:`~phenotypic.gui.tune._overlays.OverlayCache` singleton); the Curate
callbacks return a spinner immediately and an :class:`~dash.dcc.Interval` poll
swaps in the real figure once the render future resolves — Curate never blocks a
Werkzeug worker on a heavy ``apply``.

When the Image Source is unset (the run dir holds no input images), the view
shows a "point me at the plate images" prompt instead of attempting an overlay.

Like the rest of :mod:`phenotypic.gui.tune`, importing this module must never
drag ``optuna`` into ``sys.modules``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import plotly.graph_objects as go
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import FONT_FAMILY_BODY
from phenotypic.gui.tune import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.tune._run_root import TuneRunRoot

#: The default Curate overlay mode.
_DEFAULT_MODE: str = "side"

#: The overlay-readiness poll cadence (ms). Short so a freshly-submitted render
#: swaps in promptly, but not so tight it spins a Werkzeug worker needlessly.
_OVERLAY_POLL_MS: int = 750

#: The Image Source picker prompt shown when no Image Source is set.
_IMAGE_SOURCE_PROMPT: str = (
    "Point me at the plate images: the tune output directory holds no input "
    "images, so pick the calibration-image directory to render overlays."
)


def placeholder_figure(message: str) -> go.Figure:
    """A blank, axis-free figure carrying a centered status ``message``.

    Used as the immediate (non-blocking) return while a background overlay
    render is in flight, and for the "pick an Image Source" / "pin a candidate"
    states. The Okabe-Ito / UI colors are owned by the injected design tokens at
    the page level; this figure only sets the body font and an annotation.

    Args:
        message: The status text to center in the empty plot.

    Returns:
        A minimal :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()
    fig.update_layout(
        font={"family": FONT_FAMILY_BODY},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return fig


def _image_source_picker_row(
    sandbox: "Optional[SandboxRoot]", initial_dir: "Optional[str]"
) -> html.Div:
    """Build the Image Source picker row (button + selected-path label).

    When ``sandbox`` is ``None`` the picker is omitted and a short note explains
    that an Image Source cannot be picked (the standalone-without-sandbox path).
    """
    if sandbox is None:
        return html.Div(
            "Image Source picker unavailable (no sandbox bound).",
            className="tune-curate-note",
        )
    label = initial_dir if initial_dir else "no Image Source selected"
    return html.Div(
        [
            html.Span("Image Source:", className="tune-curate-label"),
            dbc.Button(
                "Browse...",
                id=ids.TUNE_BTN_PICK_IMAGE_SOURCE,
                color="primary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
            html.Span(
                label,
                id=ids.TUNE_IMAGE_SOURCE_LABEL,
                className="tune-curate-source-value",
            ),
        ],
        className="tune-curate-source-row",
    )


def _shortlist_cards(root: "TuneRunRoot") -> list[Component]:
    """Render one clickable card per shortlisted trial (empty list when none).

    Reads the shortlist from the run's finished journal (read-only; never
    re-optimizes). On a live, unfinished run the journal may be absent or short
    — the cards render from whatever trials exist, each tagged "in progress" via
    the run's live status the callback owns. A read failure degrades to no cards
    (the view stays usable).
    """
    from phenotypic.gui.tune._callbacks import _load_journal  # optuna-free
    from phenotypic.gui.tune._study_read import shortlist

    store = _load_journal(root)
    if store is None:
        return []
    cards: list[Component] = []
    for trial in shortlist(store, k=5):
        cards.append(
            html.Div(
                [
                    html.Div(f"Trial {trial.number}", className="tune-card-title"),
                    html.Div(
                        f"score {trial.score:.4f}", className="tune-card-score"
                    ),
                ],
                id={"type": ids.TUNE_SHORTLIST_CARD, "trial": trial.number},
                className="tune-shortlist-card",
                n_clicks=0,
            )
        )
    return cards


def _ab_segment() -> html.Div:
    """The A/B segment: the mode toggle + side-by-side + difference graphs."""
    side_by_side = html.Div(
        [
            dcc.Graph(
                id=ids.TUNE_GRAPH_A,
                figure=placeholder_figure("Pin a candidate to slot A"),
            ),
            dcc.Graph(
                id=ids.TUNE_GRAPH_B,
                figure=placeholder_figure("Pin a candidate to slot B"),
            ),
        ],
        id=ids.TUNE_SIDE_BY_SIDE,
        className="tune-curate-sidebyside",
    )
    difference = html.Div(
        dcc.Graph(
            id=ids.TUNE_GRAPH_DIFF,
            figure=placeholder_figure("Pin A and B to see the difference"),
        ),
        id=ids.TUNE_DIFFERENCE,
        className="tune-curate-difference tune-view-hidden",
    )
    toggle = dbc.RadioItems(
        id=ids.TUNE_CURATE_MODE_TOGGLE,
        options=[
            {"label": "Side-by-side", "value": "side"},
            {"label": "Difference", "value": "difference"},
        ],
        value=_DEFAULT_MODE,
        inline=True,
        className="tune-curate-mode-toggle",
    )
    return html.Div(
        [toggle, side_by_side, difference],
        className="tune-curate-ab",
    )


def _winner_bar() -> html.Div:
    """The winner bar: "Set as winner" button + status note."""
    return html.Div(
        [
            dbc.Button(
                "Set as winner",
                id=ids.TUNE_BTN_SET_WINNER,
                color="primary",
                n_clicks=0,
            ),
            html.Span(
                "",
                id=ids.TUNE_WINNER_NOTE,
                className="tune-winner-note",
            ),
        ],
        className="tune-curate-winner",
    )


def _curate_toast() -> dbc.Toast:
    """The floating Curate-view error toast (out-of-sandbox / write failure)."""
    return dbc.Toast(
        id=ids.TUNE_CURATE_TOAST,
        header="Curate",
        is_open=False,
        dismissable=True,
        duration=5000,
        icon="danger",
        style={
            "position": "fixed",
            "top": 20,
            "right": 20,
            "minWidth": 320,
            "zIndex": 1080,
        },
    )


def build_curate_view(
    root: "TuneRunRoot", *, sandbox: "Optional[SandboxRoot]" = None
) -> html.Div:
    """Render the Curate view body for the bound run ``root``.

    Args:
        root: The validated tune output handle.
        sandbox: The frozen-at-launch sandbox root; threaded into the
            Image Source picker modal. ``None`` degrades the picker to a note.

    Returns:
        The Curate view body: the Image Source picker + modal + stores, the
        shortlist, the A/B segment, the plate picker, the winner bar, and the
        overlay-readiness poll. When the Image Source is unset, a prompt is
        shown above the (empty) overlay area.
    """
    initial_source = str(root.images_dir) if root.images_dir is not None else None

    children: list[Component] = [
        dcc.Store(id=ids.TUNE_IMAGE_SOURCE_STORE, data=initial_source),
        dcc.Store(
            id=ids.TUNE_AB_STORE, data={"a": None, "b": None}
        ),
        dcc.Store(id=ids.TUNE_CURATE_MODE_STORE, data=_DEFAULT_MODE),
        dcc.Interval(
            id=ids.TUNE_OVERLAY_POLL, interval=_OVERLAY_POLL_MS, n_intervals=0
        ),
        _curate_toast(),
        _image_source_picker_row(sandbox, initial_source),
    ]

    # The "point me at the plate images" prompt — visible only when unset.
    prompt_style = {} if initial_source is None else {"display": "none"}
    children.append(
        html.Div(
            _IMAGE_SOURCE_PROMPT,
            id=ids.TUNE_CURATE_PROMPT,
            className="tune-curate-prompt",
            style=prompt_style,
        )
    )

    children.append(
        html.Div(
            _shortlist_cards(root),
            id=ids.TUNE_SHORTLIST,
            className="tune-shortlist",
        )
    )
    children.append(
        dcc.Dropdown(
            id=ids.TUNE_PLATE_PICKER,
            options=[],
            placeholder="Pick a plate to render",
            className="tune-plate-picker",
        )
    )
    children.append(_ab_segment())
    children.append(_winner_bar())

    if sandbox is not None:
        from phenotypic.gui.tune._image_source import build_image_source_modal

        children.append(
            build_image_source_modal(sandbox, initial_dir=root.images_dir)
        )

    return html.Div(children, className="tune-curate")


__all__ = ["build_curate_view", "placeholder_figure"]

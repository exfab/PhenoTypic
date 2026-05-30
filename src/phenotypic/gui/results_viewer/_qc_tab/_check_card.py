"""Per-check ``dbc.Card`` builder for the QC tab.

Each card pairs a :class:`~phenotypic.qc.QcRecipeEntry`
with a stable pattern-matching id namespace (see :mod:`._ids`) so the
card-body refresh callback can address every card by instance id.

Cards are intentionally cheap to build: every interactive element is a
``dbc.Button`` / ``dcc.Graph`` / ``html.Div`` with an empty initial
state. The card-body refresh callback fills in the figure, summary
strip, and status-badge text on its first fire (and on every
subsequent ``STORE_REMOVED_KEYS`` tick).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import plotly.graph_objects as go
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
)
from phenotypic.qc import QcRecipeEntry
from phenotypic.gui.results_viewer._qc_tab import _ids as ids


def _empty_initial_figure() -> go.Figure:
    """Build an empty placeholder figure for the card body.

    The card-body refresh callback overwrites this on its first fire; the
    placeholder keeps the ``dcc.Graph`` mounted so the figure callback's
    ``Output`` shape is stable.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=320,
    )
    return fig


def _title_row(entry: QcRecipeEntry) -> Component:
    """Build the card title row.

    Layout (left to right): status badge, class name + short id label,
    flexible spacer, then four icon-style buttons (edit / toggle /
    duplicate / delete).

    Args:
        entry: The recipe entry the card represents.

    Returns:
        A :class:`dash.html.Div` ready to drop into a
        :class:`dbc.CardHeader`.
    """
    instance_id = entry.instance_id
    # Short id: take the last 6 hex characters of the suffix so the
    # title stays readable. ``instance_id`` is shaped ``qc-<name>-<8 hex>``.
    short_id = instance_id.rsplit("-", 1)[-1][:6]

    status_badge = dbc.Badge(
        "...",
        id=ids.qc_card_status_badge_id(instance_id),
        color="secondary",
        className="me-2",
        pill=True,
    )

    title_label = html.Span(
        [
            html.Span(entry.cls.__name__, style={"fontWeight": 600}),
            html.Span(
                f" #{short_id}",
                style={
                    "color": COLOR_MUTED,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginLeft": "0.4rem",
                },
            ),
        ],
        style={"color": COLOR_NAVY, "fontSize": FONT_SIZE_LABEL},
    )

    toggle_label = "On" if entry.enabled else "Off"
    edit_btn = dbc.Button(
        "Edit",
        id=ids.qc_card_edit_id(instance_id),
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="me-1",
    )
    toggle_btn = dbc.Button(
        toggle_label,
        id=ids.qc_card_toggle_id(instance_id),
        color="primary" if entry.enabled else "secondary",
        outline=not entry.enabled,
        size="sm",
        n_clicks=0,
        className="me-1",
    )
    duplicate_btn = dbc.Button(
        "Duplicate",
        id=ids.qc_card_duplicate_id(instance_id),
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="me-1",
    )
    delete_btn = dbc.Button(
        "Delete",
        id=ids.qc_card_delete_id(instance_id),
        color="danger",
        outline=True,
        size="sm",
        n_clicks=0,
    )

    return html.Div(
        [
            status_badge,
            title_label,
            html.Div(style={"flex": "1 1 auto"}),
            edit_btn,
            toggle_btn,
            duplicate_btn,
            delete_btn,
        ],
        className="d-flex align-items-center",
        style={"gap": "0.25rem"},
    )


def _body(entry: QcRecipeEntry) -> Component:
    """Build the card body holding the figure, summary strip, and action button.

    Args:
        entry: The recipe entry the card represents.

    Returns:
        A :class:`dash.html.Div` ready to drop into a
        :class:`dbc.CardBody`.
    """
    instance_id = entry.instance_id

    figure = dcc.Graph(
        id=ids.qc_card_figure_id(instance_id),
        figure=_empty_initial_figure(),
        config={"displayModeBar": False, "responsive": True},
        style={"width": "100%", "height": "320px"},
    )

    summary_strip = html.Div(
        "",
        id=ids.qc_card_summary_id(instance_id),
        style={
            "padding": "0.5rem 0.75rem",
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": "4px",
            "marginTop": "0.5rem",
            "fontSize": FONT_SIZE_LABEL,
            "color": COLOR_NAVY,
            "fontFamily": "monospace",
        },
    )

    mark_flag_btn = dbc.Button(
        "Mark all flagged for removal",
        id=ids.qc_card_mark_flag_id(instance_id),
        color="warning",
        outline=True,
        size="sm",
        n_clicks=0,
        className="mt-2",
    )

    return html.Div(
        [figure, summary_strip, mark_flag_btn],
        style={"display": "flex", "flexDirection": "column"},
    )


def build_check_card(entry: QcRecipeEntry) -> dbc.Card:
    """Build one card per configured :class:`QualityCheck` instance.

    The card is a thin shell wrapper around the figure + summary + action
    button; the card-body refresh callback fills in the figure, summary
    strip, status-badge text, and badge color on its first fire. The
    title row carries every interactive control (edit / enable /
    duplicate / delete) so callbacks can pattern-match on the card's
    ``instance_id``.

    Args:
        entry: The recipe entry the card represents.

    Returns:
        A :class:`dash_bootstrap_components.Card` ready to drop into the
        :data:`._ids.QC_CARDS_CONTAINER_ID` parent.
    """
    instance_id = entry.instance_id
    return dbc.Card(
        [
            dbc.CardHeader(_title_row(entry)),
            dbc.CardBody(_body(entry)),
        ],
        id=ids.qc_card_root_id(instance_id),
        className="qc-card",
        style={
            "marginBottom": "1rem",
            "borderColor": COLOR_BORDER,
        },
    )


__all__ = ["build_check_card"]

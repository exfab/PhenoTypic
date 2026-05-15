"""Static layout for the QC tab body.

Top-level shape (vertical stack):

1. Top strip — ``+ Add check`` and ``Export QC report`` buttons plus a
   ``dbc.Toast`` for success/failure announcements.
2. Load-warning banner — shown only when
   :class:`~phenotypic.gui._qc_recipe.QcRecipe.load_warnings` is
   non-empty at boot.
3. Cards container — atomically rebuilt by the card-list-render
   callback so the card count tracks the recipe.
4. Shared add/edit modal — single ``dbc.Modal`` reused by the add /
   edit / duplicate flows.
5. Hidden ``dcc.Store`` carrying the instance id the modal is
   currently editing (``None`` in add mode).

Callbacks live in :mod:`._callbacks`; this module is layout-only so it
stays importable from tests and remains free of Dash state coupling.
"""

from __future__ import annotations

import logging
from typing import Iterable

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_SIZE_LABEL,
)
from phenotypic.gui._qc_recipe import QcRecipe, QcRecipeLoadWarning
from phenotypic.gui.results_viewer._qc_tab import _ids as ids
from phenotypic.gui.results_viewer._qc_tab._check_card import build_check_card

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _build_top_strip() -> Component:
    """Build the top strip: add / export buttons + success toast."""
    add_button = dbc.Button(
        "+ Add check",
        id=ids.QC_ADD_CHECK_BTN_ID,
        color="primary",
        n_clicks=0,
        className="me-2",
        style={"background": COLOR_NAVY, "borderColor": COLOR_NAVY},
    )
    export_button = dbc.Button(
        "Export QC report",
        id=ids.QC_EXPORT_BTN_ID,
        color="secondary",
        outline=True,
        n_clicks=0,
        disabled=True,
    )
    toast = dbc.Toast(
        id=ids.QC_EXPORT_TOAST_ID,
        header="QC export",
        is_open=False,
        dismissable=True,
        duration=8000,
        style={
            "position": "fixed",
            "top": "1rem",
            "right": "1rem",
            "zIndex": 1080,
            "minWidth": "320px",
        },
    )
    return html.Div(
        [
            html.Div(
                [add_button, export_button],
                className="d-flex align-items-center",
                style={"gap": "0.5rem"},
            ),
            toast,
        ],
        style={
            "padding": "0.75rem 1rem",
            "borderBottom": f"1px solid {COLOR_BORDER}",
            "background": COLOR_SURFACE,
        },
    )


def _render_load_warnings(
    warnings: Iterable[QcRecipeLoadWarning],
) -> Component:
    """Render the load-warning banner contents (or empty when none).

    Args:
        warnings: Sequence of :class:`QcRecipeLoadWarning` produced by
            :meth:`QcRecipe.load` (and appended-to by
            :meth:`QcRecipe.instantiate` when construction fails).

    Returns:
        A :class:`dash.html.Div` ready to slot into
        :data:`._ids.QC_LOAD_WARNING_BANNER_ID`. When ``warnings`` is
        empty, returns an empty ``html.Div`` so the parent's
        ``style.display`` toggle can still attach.
    """
    warnings_list = list(warnings)
    if not warnings_list:
        return html.Div()

    items: list[Component] = []
    for warning in warnings_list:
        items.append(
            html.Li(
                [
                    html.Code(
                        warning.instance_id,
                        style={"marginRight": "0.4rem", "color": COLOR_NAVY},
                    ),
                    html.Span(
                        warning.class_name or "(unknown class)",
                        style={"fontWeight": 500},
                    ),
                    html.Span(
                        f" — {warning.reason}",
                        style={"color": COLOR_MUTED},
                    ),
                ]
            )
        )

    return html.Div(
        [
            html.Strong(
                "Some QC checks could not be loaded:",
                style={"display": "block", "marginBottom": "0.25rem"},
            ),
            html.Ul(items, style={"marginBottom": 0}),
        ],
        style={"color": "#8a4d00"},
    )


def _banner_style(warnings: Iterable[QcRecipeLoadWarning]) -> dict[str, str]:
    """Pick the banner ``style`` dict based on whether warnings exist."""
    if list(warnings):
        return {
            "display": "block",
            "padding": "0.5rem 1rem",
            "background": "rgba(254,188,17,0.12)",
            "borderBottom": f"1px solid {COLOR_BORDER}",
            "fontSize": FONT_SIZE_LABEL,
        }
    return {"display": "none"}


def _initial_cards(recipe: QcRecipe) -> list[Component]:
    """Build the initial set of cards for enabled entries."""
    return [build_check_card(entry) for entry in recipe.entries if entry.enabled]


def _build_qc_modal() -> dbc.Modal:
    """Build the shared add / edit / duplicate modal.

    The class dropdown and param region are populated by callbacks on
    open. The submit button fires
    :func:`._callbacks._on_modal_submit` which dispatches between
    :meth:`QcRecipe.add` and :meth:`QcRecipe.update` based on
    :data:`._ids.STORE_QC_EDITING_INSTANCE`.
    """
    body = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label("Check class", className="fw-semibold"),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=ids.QC_MODAL_CLASS_PICKER_ID,
                            options=[],
                            value=None,
                            placeholder="Pick a quality-check class...",
                            clearable=False,
                        ),
                        width=8,
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            html.Hr(),
            html.Div(
                id=ids.QC_MODAL_PARAMS_REGION_ID,
                children=[],
            ),
        ]
    )

    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.QC_MODAL_CANCEL_BTN_ID,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Save",
                id=ids.QC_MODAL_SUBMIT_BTN_ID,
                color="primary",
                n_clicks=0,
            ),
        ]
    )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(
                    "Add QC check",
                    id=ids.QC_MODAL_TITLE_ID,
                ),
            ),
            dbc.ModalBody(body),
            footer,
        ],
        id=ids.QC_MODAL_ID,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_qc_tab_body(recipe: QcRecipe) -> Component:
    """Build the QC tab body.

    Mounts the top strip, load-warning banner, cards container, the
    shared add/edit modal, and the hidden ``editing-instance`` store.
    The cards container is seeded with one card per *enabled* entry in
    ``recipe`` so the first render does not need a callback fire; the
    card-list-render callback owns subsequent rebuilds.

    Args:
        recipe: The loaded :class:`QcRecipe` for the active output
            directory.

    Returns:
        A :class:`dash.html.Div` ready to drop into a :class:`dbc.Tab`.
    """
    return html.Div(
        [
            _build_top_strip(),
            html.Div(
                children=_render_load_warnings(recipe.load_warnings),
                id=ids.QC_LOAD_WARNING_BANNER_ID,
                style=_banner_style(recipe.load_warnings),
            ),
            html.Div(
                children=_initial_cards(recipe),
                id=ids.QC_CARDS_CONTAINER_ID,
                style={
                    "padding": "1rem",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "0.5rem",
                },
            ),
            _build_qc_modal(),
            dcc.Store(
                id=ids.STORE_QC_EDITING_INSTANCE,
                data=None,
                storage_type="memory",
            ),
        ],
        className="qc-tab-root",
        style={
            "maxHeight": "calc(100vh - 8rem)",
            "overflow": "auto",
            "background": COLOR_SURFACE,
        },
    )


__all__ = [
    "build_qc_tab_body",
]

"""Shell chrome callbacks.

Each Dash app instance has its own callback dispatch table, so chrome
callbacks must be registered on each wrapped app separately. ``wrap_in_chrome``
(in ``_layout.py``) is the only caller; it invokes
:func:`register_chrome_callbacks` once per app immediately after layout
mutation.

Callbacks registered here
    * RSS readout — ``dcc.Interval`` tick refreshes the top-right RSS label.
    * Help-modal toggle — open/close on the ``?`` button + close button.
    * Sidebar refresh — bumps the ``SHELL_CLASSIFIER_CACHE_STORE`` key, which
      is the signal Phase 5 callbacks watch to re-render the tree (Phase 3
      ships a stub that just flushes the classifier LRU server-side).

Sidebar selection, lazy expand, and release-button-driven
``ToolSession.release()`` are wired in Phase 5 once the run-console + viewer
sessions are actually mounted.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]
from dash import Input, Output, State, ctx, no_update

from phenotypic.gui.shell._classifier import invalidate_cache
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_HELP_BUTTON,
    SHELL_HELP_MODAL,
    SHELL_RSS_INTERVAL,
    SHELL_RSS_LABEL,
    SHELL_SIDEBAR_REFRESH,
)

if TYPE_CHECKING:
    import dash

    from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["register_chrome_callbacks"]


def register_chrome_callbacks(
    app: "dash.Dash",
    sandbox: "SandboxRoot",
) -> None:
    """Register the chrome callbacks on ``app``.

    Args:
        app: Dash app whose layout already contains the chrome IDs (i.e.
            ``wrap_in_chrome`` has run).
        sandbox: Sandbox root. Currently unused — wired through for Phase 5
            callbacks (sidebar lazy expand will need it).
    """
    del sandbox  # unused in Phase 3; kept on signature for Phase 5 stability

    process = psutil.Process(os.getpid())

    @app.callback(
        Output(SHELL_RSS_LABEL, "children"),
        Input(SHELL_RSS_INTERVAL, "n_intervals"),
    )
    def _update_rss(_n: int) -> str:
        try:
            rss_bytes = process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
            return "RSS ?"
        return f"RSS {rss_bytes / 1e6:.0f} MB"

    @app.callback(
        Output(SHELL_HELP_MODAL, "is_open"),
        Input(SHELL_HELP_BUTTON, "n_clicks"),
        Input(
            {"type": "shell-help-close", "scope": "modal"},
            "n_clicks",
        ),
        State(SHELL_HELP_MODAL, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_help_modal(
        open_clicks: int,
        close_clicks: int,
        is_open: bool,
    ) -> bool:
        # ``ctx.triggered_id`` distinguishes which button fired. The
        # ``?`` button toggles; the close button always closes.
        triggered = ctx.triggered_id
        if triggered == SHELL_HELP_BUTTON:
            if not open_clicks:
                return no_update  # type: ignore[return-value]
            return not is_open
        # Close button: only fire on a real click (n_clicks > 0).
        if not close_clicks:
            return no_update  # type: ignore[return-value]
        return False

    @app.callback(
        Output(SHELL_CLASSIFIER_CACHE_STORE, "data"),
        Input(SHELL_SIDEBAR_REFRESH, "n_clicks"),
        State(SHELL_CLASSIFIER_CACHE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _refresh_classifier(n_clicks: int, version: int | None) -> int:
        if not n_clicks:
            return no_update  # type: ignore[return-value]
        invalidate_cache()
        bumped = (version or 0) + 1
        logger.debug(
            "sidebar refresh: classifier cache flushed; version=%d", bumped
        )
        return bumped

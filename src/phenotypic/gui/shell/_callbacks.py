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
      is the signal the tree-render callback watches to re-paint.
    * Sidebar entry click — toggles a directory's rel-path in
      ``SHELL_SIDEBAR_EXPANDED_STORE`` AND stamps the click target into
      ``SHELL_SIDEBAR_SELECTION_STORE`` (consumed by per-tool ``[↩ from
      sidebar]`` buttons).
    * Sidebar tree re-render — fires when the expanded set, hidden /
      symlink toggles, or classifier-cache version change.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]
from dash import ALL, Input, Output, State, ctx, no_update

from phenotypic.gui.shell._classifier import classify, invalidate_cache
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_HELP_BUTTON,
    SHELL_HELP_MODAL,
    SHELL_RSS_INTERVAL,
    SHELL_RSS_LABEL,
    SHELL_SIDEBAR_EXPANDED_STORE,
    SHELL_SIDEBAR_HIDDEN_TOGGLE,
    SHELL_SIDEBAR_REFRESH,
    SHELL_SIDEBAR_SELECTION_STORE,
    SHELL_SIDEBAR_SYMLINK_TOGGLE,
    SHELL_SIDEBAR_TREE,
)
from phenotypic.gui.shell._sidebar import render_tree

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
        sandbox: Sandbox root. Used by the sidebar entry-click callback
            to resolve clicked paths and by the tree-render callback to
            walk the tree on each expansion change.
    """
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

    # ----------------------------------------------------------------------
    # Sidebar entry click — toggles expanded set + stamps selection store.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(SHELL_SIDEBAR_EXPANDED_STORE, "data"),
        Output(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        Input({"type": "shell-sidebar-entry", "path": ALL}, "n_clicks"),
        State(SHELL_SIDEBAR_EXPANDED_STORE, "data"),
        prevent_initial_call=True,
    )
    def _handle_entry_click(
        _clicks: list[int],
        expanded: list[str] | None,
    ) -> tuple[Any, Any]:
        """Handle a click on any ``shell-sidebar-entry`` button.

        Two effects:

        * **Selection** — the click target's rel-path + capability badges
          are stamped into ``SHELL_SIDEBAR_SELECTION_STORE``. Per-tool
          ``[↩ from sidebar]`` buttons read this store to know what the
          user picked.
        * **Expansion** — if the target is a directory, its rel-path is
          toggled in ``SHELL_SIDEBAR_EXPANDED_STORE``. The tree-render
          callback (below) watches this store and re-paints the listing.

        ``ctx.triggered_id`` distinguishes the pattern-matching click;
        ``ctx.triggered[0]['value']`` filters out the spurious initial
        fire that Dash emits when ``allow_duplicate=True`` is in play
        (no real click happened, ``value`` is the default ``0``/``None``).
        """
        triggered = ctx.triggered_id
        if (
            not isinstance(triggered, dict)
            or triggered.get("type") != "shell-sidebar-entry"
        ):
            return (no_update, no_update)
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update, no_update)

        rel = triggered.get("path") or ""
        if not rel:
            return (no_update, no_update)

        # Build the selection payload first; the per-tool consumers want
        # the capabilities so they can route directories vs JSON files
        # vs CLI outputs to different destinations without re-classifying.
        try:
            target = sandbox.resolve(rel)
        except ValueError:
            # Selection of an out-of-sandbox path (e.g. an external
            # symlink row that was clicked despite being disabled).
            # Stamp the rel-path with no capabilities; tools that
            # consume the store will see empty caps and decline.
            return (no_update, {"path": rel, "capabilities": None})

        try:
            caps = classify(target)
            caps_payload: dict[str, Any] | None = {
                "is_image_dir": caps.is_image_dir,
                "has_pipeline_json": caps.has_pipeline_json,
                "is_cli_output": caps.is_cli_output,
                "has_dashboard": caps.has_dashboard,
                "image_count": caps.image_count,
                "bad_perms": caps.bad_perms,
            }
        except Exception:  # pragma: no cover - defensive
            caps_payload = None

        selection: dict[str, Any] = {
            "path": rel,
            "abs_path": str(target),
            "is_dir": target.is_dir(),
            "capabilities": caps_payload,
        }

        if not target.is_dir() or target.is_symlink():
            # Files and symlinks don't expand — just update selection.
            return (no_update, selection)

        expanded_set = set(expanded or [])
        if rel in expanded_set:
            expanded_set.remove(rel)
        else:
            expanded_set.add(rel)
        # Sort for deterministic store payloads (helps test snapshots).
        return (sorted(expanded_set), selection)

    # ----------------------------------------------------------------------
    # Sidebar tree re-render — fires when expansion / toggles / cache
    # version change.
    # ----------------------------------------------------------------------

    @app.callback(
        Output(SHELL_SIDEBAR_TREE, "children"),
        Input(SHELL_SIDEBAR_EXPANDED_STORE, "data"),
        Input(SHELL_SIDEBAR_HIDDEN_TOGGLE, "value"),
        Input(SHELL_SIDEBAR_SYMLINK_TOGGLE, "value"),
        Input(SHELL_CLASSIFIER_CACHE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _render_sidebar_tree(
        expanded: list[str] | None,
        hidden_value: list[str] | None,
        symlink_value: list[str] | None,
        _cache_version: int | None,
    ) -> Any:
        """Re-render the sandbox tree on expansion / toggle / refresh.

        ``_cache_version`` is included as an Input so the Refresh button
        (which bumps that store) cascades to a tree re-paint.
        """
        return render_tree(
            sandbox,
            include_hidden=bool(hidden_value),
            include_external=bool(symlink_value),
            expanded=set(expanded or []),
        )

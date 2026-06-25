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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]
from dash import ALL, Input, Output, State, ctx, no_update

from phenotypic.gui.shell._classifier import classify, invalidate_cache
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_HELP_BUTTON,
    SHELL_HELP_MODAL,
    SHELL_METADATA_CSV_BROWSE_STORE,
    SHELL_METADATA_CSV_CANCEL,
    SHELL_METADATA_CSV_CONFIRM,
    SHELL_METADATA_CSV_ENTRY_TYPE,
    SHELL_METADATA_CSV_MODAL,
    SHELL_METADATA_CSV_MODAL_BODY,
    SHELL_METADATA_CSV_STORE,
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
    SHELL_SIDEBAR_EXPANDED_STORE,
    SHELL_SIDEBAR_HIDDEN_TOGGLE,
    SHELL_SIDEBAR_REFRESH,
    SHELL_SIDEBAR_SELECTION_STORE,
    SHELL_SIDEBAR_SYMLINK_TOGGLE,
    SHELL_SIDEBAR_TREE,
    SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE,
    SHELL_SOURCE_IMAGE_ROOT_CANCEL,
    SHELL_SOURCE_IMAGE_ROOT_CONFIRM,
    SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE,
    SHELL_SOURCE_IMAGE_ROOT_LABEL,
    SHELL_SOURCE_IMAGE_ROOT_MODAL,
    SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._sidebar import render_tree
from phenotypic.gui.shell._metadata_context import (
    metadata_csv_label,
    metadata_csv_title,
    metadata_payload_from_path,
    resolve_metadata_csv,
)
from phenotypic.gui.shell._source_context import (
    resolve_source_image_root,
    source_label,
    source_payload_from_path,
    source_title,
)
from phenotypic.gui.shell._source_picker import (
    render_metadata_csv_picker_tree,
    render_source_picker_tree,
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
        Output(SHELL_SETTINGS_POPOVER, "is_open"),
        Input(SHELL_SETTINGS_BUTTON, "n_clicks"),
        State(SHELL_SETTINGS_POPOVER, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_settings_popover(n_clicks: int, is_open: bool) -> bool:
        if not n_clicks:
            return no_update  # type: ignore[return-value]
        return not is_open

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_LABEL, "children"),
        Output(SHELL_SOURCE_IMAGE_ROOT_LABEL, "title"),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _update_source_label(payload: object) -> tuple[str, str]:
        return source_label(payload), source_title(payload)

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_duplicate=True),
        Input(SHELL_SETTINGS_INPUT_FOLDER_CLEAR, "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_source_root(n_clicks: int) -> Any:
        if not n_clicks:
            return no_update
        return None

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_MODAL, "is_open", allow_duplicate=True),
        Output(SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE, "data", allow_duplicate=True),
        Input(SHELL_SETTINGS_INPUT_FOLDER_PICK, "n_clicks"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _open_source_picker(
        n_clicks: int,
        current_payload: object,
    ) -> tuple[Any, Any]:
        if not n_clicks:
            return no_update, no_update
        current = resolve_source_image_root(sandbox, current_payload)
        browse_dir = current if current is not None else sandbox.root
        return True, str(browse_dir)

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_MODAL, "is_open", allow_duplicate=True),
        Input(SHELL_SOURCE_IMAGE_ROOT_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def _cancel_source_picker(n_clicks: int) -> Any:
        if not n_clicks:
            return no_update
        return False

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE, "data", allow_duplicate=True),
        Input(
            {
                "type": SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE,
                "kind": ALL,
                "path": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _navigate_source_picker(_clicks: list[int]) -> Any:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != SHELL_SOURCE_IMAGE_ROOT_ENTRY_TYPE:
            return no_update
        if not any(t.get("value") for t in (ctx.triggered or [])):
            return no_update
        path = triggered.get("path")
        return path if isinstance(path, str) and path else no_update

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_MODAL_BODY, "children"),
        Input(SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _render_source_picker_body(dir_value: str | None) -> Any:
        current = Path(dir_value) if dir_value else None
        return render_source_picker_tree(sandbox, current)

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_duplicate=True),
        Output(SHELL_SOURCE_IMAGE_ROOT_MODAL, "is_open", allow_duplicate=True),
        Input(SHELL_SOURCE_IMAGE_ROOT_CONFIRM, "n_clicks"),
        State(SHELL_SOURCE_IMAGE_ROOT_BROWSE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _confirm_source_picker(
        n_clicks: int,
        dir_value: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not dir_value:
            return no_update, no_update
        payload = source_payload_from_path(sandbox, dir_value, source="manual")
        if payload is None:
            return no_update, no_update
        return payload, False

    @app.callback(
        Output(SHELL_SETTINGS_METADATA_CSV_LABEL, "children"),
        Output(SHELL_SETTINGS_METADATA_CSV_LABEL, "title"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
    )
    def _update_metadata_csv_label(payload: object) -> tuple[str, str]:
        return metadata_csv_label(payload), metadata_csv_title(payload)

    @app.callback(
        Output(SHELL_METADATA_CSV_STORE, "data", allow_duplicate=True),
        Input(SHELL_SETTINGS_METADATA_CSV_CLEAR, "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_metadata_csv(n_clicks: int) -> Any:
        if not n_clicks:
            return no_update
        return None

    @app.callback(
        Output(SHELL_METADATA_CSV_MODAL, "is_open", allow_duplicate=True),
        Output(SHELL_METADATA_CSV_BROWSE_STORE, "data", allow_duplicate=True),
        Input(SHELL_SETTINGS_METADATA_CSV_PICK, "n_clicks"),
        State(SHELL_METADATA_CSV_STORE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _open_metadata_picker(
        n_clicks: int,
        current_payload: object,
        source_payload: object,
    ) -> tuple[Any, Any]:
        if not n_clicks:
            return no_update, no_update
        current = resolve_metadata_csv(sandbox, current_payload)
        if current is not None:
            return True, str(current.parent)
        source = resolve_source_image_root(sandbox, source_payload)
        browse_dir = source if source is not None else sandbox.root
        return True, str(browse_dir)

    @app.callback(
        Output(SHELL_METADATA_CSV_MODAL, "is_open", allow_duplicate=True),
        Input(SHELL_METADATA_CSV_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def _cancel_metadata_picker(n_clicks: int) -> Any:
        if not n_clicks:
            return no_update
        return False

    @app.callback(
        Output(SHELL_METADATA_CSV_BROWSE_STORE, "data", allow_duplicate=True),
        Input(
            {
                "type": SHELL_METADATA_CSV_ENTRY_TYPE,
                "kind": ALL,
                "path": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def _navigate_metadata_picker(_clicks: list[int]) -> Any:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != SHELL_METADATA_CSV_ENTRY_TYPE:
            return no_update
        if not any(t.get("value") for t in (ctx.triggered or [])):
            return no_update
        path = triggered.get("path")
        return path if isinstance(path, str) and path else no_update

    @app.callback(
        Output(SHELL_METADATA_CSV_MODAL_BODY, "children"),
        Input(SHELL_METADATA_CSV_BROWSE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _render_metadata_picker_body(path_value: str | None) -> Any:
        current = Path(path_value) if path_value else None
        return render_metadata_csv_picker_tree(sandbox, current)

    @app.callback(
        Output(SHELL_METADATA_CSV_STORE, "data", allow_duplicate=True),
        Output(SHELL_METADATA_CSV_MODAL, "is_open", allow_duplicate=True),
        Input(SHELL_METADATA_CSV_CONFIRM, "n_clicks"),
        State(SHELL_METADATA_CSV_BROWSE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _confirm_metadata_picker(
        n_clicks: int,
        path_value: str | None,
    ) -> tuple[Any, Any]:
        if not n_clicks or not path_value:
            return no_update, no_update
        payload = metadata_payload_from_path(sandbox, path_value)
        if payload is None:
            return no_update, no_update
        return payload, False

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
        # ``allow_duplicate=True`` is required because the run console's
        # hand-off-button callback also writes ``SHELL_SIDEBAR_SELECTION_STORE``
        # (clearing it on Dismiss / route). Dash requires every writer to
        # opt in explicitly; otherwise registration silently drops one of
        # the callbacks under permissive Dash versions and raises under
        # strict ones.
        Output(SHELL_SIDEBAR_SELECTION_STORE, "data", allow_duplicate=True),
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

        ``ctx.triggered_id`` resolves the pattern-matching click. The
        ``ctx.triggered`` list may contain entries with ``value=0`` from
        newly-mounted buttons whose ``n_clicks`` was just initialised; we
        require *some* entry in that list to carry a truthy value before
        treating the call as a real click.
        """
        triggered = ctx.triggered_id
        if (
            not isinstance(triggered, dict)
            or triggered.get("type") != "shell-sidebar-entry"
        ):
            return (no_update, no_update)
        # Defensive against rapid double-clicks across multiple entries:
        # ``ctx.triggered`` is a list, ``triggered[0]`` is not necessarily
        # the entry the user actually clicked. Accept the call as long as
        # at least one entry in the batch has a real (>0) ``n_clicks``.
        if not any(t.get("value") for t in (ctx.triggered or [])):
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
                "is_deliverables_bundle": caps.is_deliverables_bundle,
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

    @app.callback(
        Output(SHELL_SOURCE_IMAGE_ROOT_STORE, "data", allow_duplicate=True),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _source_from_sidebar_selection(
        selection: dict[str, Any] | None,
        current_payload: object,
    ) -> Any:
        """Promote image-directory sidebar selections to the shared source."""
        if not selection or not isinstance(selection, dict):
            return no_update
        caps = selection.get("capabilities") or {}
        if not isinstance(caps, dict) or not caps.get("is_image_dir"):
            return no_update
        path = selection.get("abs_path") or selection.get("path")
        if not isinstance(path, str) or not path:
            return no_update
        payload = source_payload_from_path(sandbox, path, source="sidebar")
        if payload is None:
            return no_update
        if (
            isinstance(current_payload, dict)
            and current_payload.get("abs_path") == payload["abs_path"]
        ):
            return no_update
        return payload

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

    # ----------------------------------------------------------------------
    # Sidebar collapse — clientside callbacks. Two callbacks form a clean
    # DAG: (1) button click flips the store; (2) store change toggles the
    # ``shell-sidebar-collapsed`` class on ``.shell-root`` and swaps the
    # button glyph. The store uses ``storage_type="local"`` so the state
    # persists across the four DispatcherMiddleware mounts.
    # ----------------------------------------------------------------------

    app.clientside_callback(
        """
        function(n, current) {
            if (!n) { return current; }
            return !current;
        }
        """,
        Output(SHELL_SIDEBAR_COLLAPSE_STORE, "data"),
        Input(SHELL_SIDEBAR_COLLAPSE_BUTTON, "n_clicks"),
        State(SHELL_SIDEBAR_COLLAPSE_STORE, "data"),
    )

    app.clientside_callback(
        """
        function(collapsed) {
            const root = document.querySelector('.shell-root');
            if (root) {
                root.classList.toggle('shell-sidebar-collapsed', !!collapsed);
            }
            return collapsed ? '»' : '«';
        }
        """,
        Output(SHELL_SIDEBAR_COLLAPSE_BUTTON, "children"),
        Input(SHELL_SIDEBAR_COLLAPSE_STORE, "data"),
    )

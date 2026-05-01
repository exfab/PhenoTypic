"""Sidebar tree component + capability badges + hidden / symlink toggles.

The sidebar is a thin Dash wrapper around the JSON ``/sandbox/api/*`` blueprint
shipped in Phase 2. Lazy expansion (one level per click) and badges are
client-driven via ``fetch()`` against the blueprint, but Phase 3 ships the
shell-only happy path: the initial root listing + the toggles + the refresh
button + the selection store. Real lazy expansion lands as a clientside
callback in Phase 5 once we have the JS asset bundle plumbed in.

For Phase 3 we render the root listing server-side once at layout build time.
This keeps the integration tests deterministic (no clientside fetch to
mock) and means the sidebar is *useful* even before Phase 5 wires up the
expand/refresh callbacks.
"""
from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.shell._classifier import Capabilities, classify
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_SIDEBAR,
    SHELL_SIDEBAR_HIDDEN_TOGGLE,
    SHELL_SIDEBAR_REFRESH,
    SHELL_SIDEBAR_SELECTION_STORE,
    SHELL_SIDEBAR_SYMLINK_TOGGLE,
    SHELL_SIDEBAR_TREE,
    sidebar_entry_id,
)
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = ["build_sidebar", "render_tree", "build_badges"]

#: Maximum number of children classified at layout-build time. Beyond this
#: limit, surplus rows are rendered with placeholder badges + a trailing
#: "+N more (use Refresh)" hint. Mirrors the JSON API's
#: ``_CHILDREN_CLASSIFY_CAP`` so the server-rendered Phase 3 sidebar and
#: the Phase 5 clientside-fetched sidebar share the same boot cost.
_SIDEBAR_CLASSIFY_CAP = 500


def build_sidebar(sandbox: SandboxRoot) -> html.Div:
    """Build the left-rail sidebar component.

    Renders:

    * Toggles (``[☐ Hidden files] [☐ External symlinks]``).
    * Refresh button — Phase 5 wires up the clientside callback that
      flushes the classifier cache.
    * Tree body — Phase 3 renders the root listing only. Phase 5 lazy-
      expands subdirectories on click.
    * The selection ``dcc.Store`` (mounted as a sibling of the tree so
      body re-renders don't re-initialise it).
    """
    return html.Div(
        [
            html.Div(
                [
                    dbc.Checklist(
                        id=SHELL_SIDEBAR_HIDDEN_TOGGLE,
                        options=[{"label": " Hidden files", "value": "on"}],
                        value=[],
                        switch=True,
                        className="shell-sidebar-toggle",
                    ),
                    dbc.Checklist(
                        id=SHELL_SIDEBAR_SYMLINK_TOGGLE,
                        options=[{"label": " External symlinks", "value": "on"}],
                        value=[],
                        switch=True,
                        className="shell-sidebar-toggle",
                    ),
                    dbc.Button(
                        "Refresh",
                        id=SHELL_SIDEBAR_REFRESH,
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                    ),
                ],
                className="shell-sidebar-controls",
            ),
            html.Div(
                render_tree(sandbox, include_hidden=False, include_external=False),
                id=SHELL_SIDEBAR_TREE,
                className="shell-sidebar-tree",
            ),
            dcc.Store(id=SHELL_SIDEBAR_SELECTION_STORE),
            dcc.Store(id=SHELL_CLASSIFIER_CACHE_STORE, data=0),
        ],
        id=SHELL_SIDEBAR,
        className="shell-sidebar",
    )


def render_tree(
    sandbox: SandboxRoot,
    *,
    include_hidden: bool,
    include_external: bool,
) -> html.Ul:
    """Render the top-level sandbox listing as a ``<ul>`` of entries.

    Used both at layout-build time (Phase 3) and as the response shape for
    the future lazy-expand callback (Phase 5).
    """
    try:
        children = list(
            sandbox.list_children(
                include_hidden=include_hidden,
                include_external_symlinks=include_external,
            )
        )
    except (PermissionError, FileNotFoundError):
        return html.Ul(
            [html.Li("(unreadable)", className="shell-sidebar-empty")],
            className="shell-sidebar-list",
        )

    sorted_children = sorted(
        children, key=lambda p: (not p.is_dir(), p.name.lower())
    )
    if not sorted_children:
        return html.Ul(
            [
                html.Li(
                    "Empty directory.",
                    className="shell-sidebar-empty text-muted",
                )
            ],
            className="shell-sidebar-list",
        )

    rows: list = []
    truncated = False
    for idx, child in enumerate(sorted_children):
        classify_this = idx < _SIDEBAR_CLASSIFY_CAP
        rows.append(
            _build_row(
                child,
                sandbox,
                include_external=include_external,
                classify_this=classify_this,
            )
        )
        if not classify_this:
            truncated = True
    if truncated:
        extra = len(sorted_children) - _SIDEBAR_CLASSIFY_CAP
        rows.append(
            html.Li(
                f"+{extra} more (Refresh after navigating)",
                className="shell-sidebar-empty text-muted",
            )
        )
    return html.Ul(rows, className="shell-sidebar-list")


def build_badges(caps: Capabilities) -> list[html.Span]:
    """Translate a :class:`Capabilities` summary to badge spans.

    Returns one or more ``html.Span`` elements suitable for inline placement
    next to a sidebar entry's label. ``img``/``cfg``/``out`` are positive
    capability badges; ``?`` surfaces ``bad_perms``.
    """
    out: list[html.Span] = []
    if caps.bad_perms:
        out.append(html.Span("?", className="shell-badge shell-badge-perm"))
        return out
    if caps.is_image_dir:
        label = "img"
        if caps.image_count is not None:
            label = f"img ({caps.image_count})"
        out.append(html.Span(label, className="shell-badge shell-badge-img"))
    if caps.has_pipeline_json:
        out.append(html.Span("cfg", className="shell-badge shell-badge-cfg"))
    if caps.is_cli_output:
        out.append(html.Span("out", className="shell-badge shell-badge-out"))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_row(
    child: Path,
    sandbox: SandboxRoot,
    *,
    include_external: bool,
    classify_this: bool = True,
) -> html.Li:
    """One sidebar tree row with badges + (optional) external-symlink mark."""
    rel_path = str(child.relative_to(sandbox.root))
    is_external = (
        child.is_symlink()
        and include_external
        and not sandbox.contains(child)
    )
    if is_external:
        # Spec: external symlinks render as disabled; never classify
        # (would read content outside the sandbox).
        badges: list[html.Span] = [
            html.Span(
                "ext",
                className="shell-badge shell-badge-external",
            )
        ]
        icon = "🔗"
        cls = "shell-sidebar-row shell-sidebar-row-external"
    elif not classify_this:
        # Beyond the classify cap — render the row without badges so the
        # sidebar still surfaces the entry without paying for the stat.
        badges = []
        icon = "📁" if child.is_dir() else "📄"
        cls = "shell-sidebar-row shell-sidebar-row-uncl"
    else:
        caps = classify(child)
        badges = build_badges(caps)
        icon = "📁" if child.is_dir() else "📄"
        cls = "shell-sidebar-row"

    return html.Li(
        html.Button(
            [
                html.Span(icon, className="shell-sidebar-icon"),
                html.Span(child.name, className="shell-sidebar-name"),
                html.Span(badges, className="shell-sidebar-badges"),
            ],
            id=sidebar_entry_id(rel_path),
            n_clicks=0,
            className=cls,
        ),
    )



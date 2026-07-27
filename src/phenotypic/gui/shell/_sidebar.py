"""Sidebar tree component + capability badges + hidden / symlink toggles.

The sidebar is a thin Dash wrapper around the JSON ``/sandbox/api/*`` blueprint
shipped in Phase 2. The initial root listing is rendered server-side at
layout build time. A pair of post-Phase-9 callbacks drives lazy expansion
(one level per click, recursively addressable via the
``SHELL_SIDEBAR_EXPANDED_STORE``) and stamps a selection payload onto
``SHELL_SIDEBAR_SELECTION_STORE`` so per-tool ``[↩ from sidebar]`` buttons
can route a chosen path to the active tab.

Server-side rendering keeps the integration tests deterministic (no
clientside ``fetch`` to mock) and matches the rest of the chrome's
"render-then-callback" idiom.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.shell._classifier import Capabilities, classify
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_RESULTS_BINDING_CANCEL,
    SHELL_RESULTS_BINDING_DETAIL,
    SHELL_RESULTS_BINDING_DIAGNOSTIC,
    SHELL_RESULTS_BINDING_JOB_STORE,
    SHELL_RESULTS_BINDING_PANEL,
    SHELL_RESULTS_BINDING_PHASE,
    SHELL_RESULTS_BINDING_POLL_INTERVAL,
    SHELL_RESULTS_BINDING_PROGRESS,
    SHELL_RESULTS_BINDING_PROGRESS_LABEL,
    SHELL_RESULTS_BINDING_STATUS,
    SHELL_SIDEBAR,
    SHELL_SIDEBAR_EXPANDED_STORE,
    SHELL_SIDEBAR_HIDDEN_TOGGLE,
    SHELL_SIDEBAR_REFRESH,
    SHELL_SIDEBAR_SELECTION_STORE,
    SHELL_SIDEBAR_SYMLINK_TOGGLE,
    SHELL_SIDEBAR_TREE,
    sidebar_entry_id,
)
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = ["build_sidebar", "render_tree", "build_badges"]

#: Cap on recursive depth so a pathological symlink loop or huge tree
#: cannot lock the render thread. Eight is enough for v1; lazy expansion
#: still works manually past this depth via Refresh.
_SIDEBAR_MAX_DEPTH = 8

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
            _build_results_binding_panel(),
            dcc.Store(id=SHELL_SIDEBAR_SELECTION_STORE),
            dcc.Store(id=SHELL_SIDEBAR_EXPANDED_STORE, data=[]),
            dcc.Store(id=SHELL_CLASSIFIER_CACHE_STORE, data=0),
            dcc.Store(
                id=SHELL_RESULTS_BINDING_JOB_STORE,
                data=None,
                storage_type="session",
            ),
            dcc.Interval(
                id=SHELL_RESULTS_BINDING_POLL_INTERVAL,
                interval=400,
                n_intervals=0,
                disabled=True,
            ),
        ],
        id=SHELL_SIDEBAR,
        className="shell-sidebar",
    )


def _build_results_binding_panel() -> html.Div:
    """Build the cross-mount Results/Analysis hand-off status card."""
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Results hand-off"),
                    html.Span("Idle", id=SHELL_RESULTS_BINDING_STATUS),
                ],
                className="shell-results-binding-heading",
            ),
            html.Div(
                "",
                id=SHELL_RESULTS_BINDING_PHASE,
                className="shell-results-binding-phase",
            ),
            html.Progress(
                id=SHELL_RESULTS_BINDING_PROGRESS,
                value="0",
                max="1",
                className="shell-results-binding-progress",
            ),
            html.Div(
                "",
                id=SHELL_RESULTS_BINDING_PROGRESS_LABEL,
                className="shell-results-binding-progress-label",
            ),
            html.Div(
                "",
                id=SHELL_RESULTS_BINDING_DETAIL,
                className="shell-results-binding-detail",
            ),
            html.Div(
                "",
                id=SHELL_RESULTS_BINDING_DIAGNOSTIC,
                className="shell-results-binding-diagnostic",
            ),
            dbc.Button(
                "Cancel",
                id=SHELL_RESULTS_BINDING_CANCEL,
                n_clicks=0,
                size="sm",
                color="danger",
                outline=True,
                disabled=True,
                className="shell-results-binding-cancel",
            ),
        ],
        id=SHELL_RESULTS_BINDING_PANEL,
        className=(
            "shell-results-binding-panel shell-results-binding-panel--hidden"
        ),
        **cast(Any, {"aria-live": "polite"}),
    )


def render_tree(
    sandbox: SandboxRoot,
    *,
    include_hidden: bool,
    include_external: bool,
    expanded: "set[str] | None" = None,
) -> html.Ul:
    """Render the sandbox tree as a nested ``<ul>``.

    The root listing always renders. When ``expanded`` contains a
    directory's sandbox-relative path, that directory's children are
    rendered as a nested ``<ul>`` directly underneath the row. Recursion
    is capped at :data:`_SIDEBAR_MAX_DEPTH` to avoid pathological deep
    trees.

    Args:
        sandbox: Frozen-at-launch sandbox root.
        include_hidden: Pass-through for :meth:`SandboxRoot.list_children`.
        include_external: Pass-through; external symlinks render as
            disabled rows when ``True``.
        expanded: Set of currently-expanded directory rel-paths. Pass
            ``None`` (default) for the chrome's first paint.

    Returns:
        A :class:`dash.html.Ul` with class ``shell-sidebar-list``.
    """
    return _render_dir_listing(
        sandbox.root,
        sandbox,
        expanded=expanded or set(),
        include_hidden=include_hidden,
        include_external=include_external,
        depth=0,
    )


def _render_dir_listing(
    directory: Path,
    sandbox: SandboxRoot,
    *,
    expanded: "set[str]",
    include_hidden: bool,
    include_external: bool,
    depth: int,
) -> html.Ul:
    """Render one directory's contents (used by :func:`render_tree` and itself).

    Recursive helper. The root call (``depth == 0``) starts at
    ``sandbox.root``; expanded sub-directories recurse with ``depth + 1``.
    """
    if depth > _SIDEBAR_MAX_DEPTH:
        return html.Ul(
            [
                html.Li(
                    f"(max depth {_SIDEBAR_MAX_DEPTH})",
                    className="shell-sidebar-empty text-muted",
                )
            ],
            className="shell-sidebar-list",
        )

    listing_arg = directory if directory != sandbox.root else None
    try:
        children = list(
            sandbox.list_children(
                listing_arg,
                include_hidden=include_hidden,
                include_external_symlinks=include_external,
            )
        )
    except (PermissionError, FileNotFoundError, ValueError):
        return html.Ul(
            [html.Li("(unreadable)", className="shell-sidebar-empty")],
            className="shell-sidebar-list",
        )

    sorted_children = sorted(
        children, key=lambda p: (not p.is_dir(), p.name.lower())
    )
    if not sorted_children:
        empty_text = "Empty directory." if depth == 0 else "(empty)"
        return html.Ul(
            [
                html.Li(
                    empty_text,
                    className="shell-sidebar-empty text-muted",
                )
            ],
            className="shell-sidebar-list",
        )

    rows: list = []
    truncated = False
    for idx, child in enumerate(sorted_children):
        classify_this = idx < _SIDEBAR_CLASSIFY_CAP
        try:
            rel_path = str(child.relative_to(sandbox.root))
        except ValueError:
            # Symlink target outside sandbox; ``rel_path`` is meaningless,
            # so this row can't be expanded — fall back to the basename.
            rel_path = child.name
        is_expanded = rel_path in expanded
        rows.append(
            _build_row(
                child,
                sandbox,
                include_external=include_external,
                classify_this=classify_this,
                is_expanded=is_expanded,
            )
        )
        if (
            is_expanded
            and child.is_dir()
            and not child.is_symlink()
        ):
            sublist = _render_dir_listing(
                child,
                sandbox,
                expanded=expanded,
                include_hidden=include_hidden,
                include_external=include_external,
                depth=depth + 1,
            )
            rows.append(
                html.Li(sublist, className="shell-sidebar-children")
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
    next to a sidebar entry's label. ``img``/``cfg``/``out``/``bundle`` are
    positive capability badges; ``?`` surfaces ``bad_perms``. A standalone
    deliverables bundle (``deliverables/master`` but no ``results/``) gets the
    distinct ``bundle`` badge so it is recognizable as a viewer-openable output
    even though it is not a full ``out`` run.
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
    elif caps.is_deliverables_bundle:
        out.append(
            html.Span("bundle", className="shell-badge shell-badge-bundle")
        )
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
    is_expanded: bool = False,
) -> html.Li:
    """One sidebar tree row with badges + (optional) external-symlink mark.

    Expanded directories render with an open-folder icon (📂) so the user
    can see at a glance which folders are showing their children. The
    closed/expanded state lives in :data:`SHELL_SIDEBAR_EXPANDED_STORE`
    and is mutated by the entry-click callback in
    :mod:`._callbacks`.
    """
    try:
        rel_path = str(child.relative_to(sandbox.root))
    except ValueError:
        rel_path = child.name
    is_external = (
        child.is_symlink()
        and include_external
        and not sandbox.contains(child)
    )

    def _dir_icon() -> str:
        return "📂" if is_expanded else "📁"

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
        icon = _dir_icon() if child.is_dir() else "📄"
        cls = "shell-sidebar-row shell-sidebar-row-uncl"
    else:
        caps = classify(child)
        badges = build_badges(caps)
        icon = _dir_icon() if child.is_dir() else "📄"
        cls = "shell-sidebar-row"

    if child.is_dir() and not is_external:
        action = "Collapse folder" if is_expanded else "Expand folder"
    else:
        action = "Select path"
    accessible_label = f"{action}: {rel_path}"

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
            title=accessible_label,
            **cast(Any, {"aria-label": accessible_label}),
        ),
    )



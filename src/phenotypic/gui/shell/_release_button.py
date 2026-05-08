"""Reusable per-tool Release button + RSS readout.

UX honesty (locked through plan review): the button is labelled
**"Release loaded data"** with a tooltip explaining that process RSS may
stay elevated. We do NOT promise RSS reduction — Python allocator behaviour
means freed objects rarely shrink RSS. The honest claim is "the next visit
re-loads from disk." Tests therefore assert *object-graph drop*, not RSS
reduction.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui.shell._ids import ToolName, release_button_id, release_status_id

__all__ = ["build_release_button"]

#: Tooltip body for the Release button. Honest about the RSS retention
#: caveat so users don't expect Python to give pages back to the OS.
_TOOLTIP = (
    "Drops Python references to this tool's loaded state (e.g. parquet, "
    "curation, intermediate caches). The next visit rebuilds from disk. "
    "NOTE: process RSS may stay elevated even after release because "
    "Python's allocators retain freed memory pages. For hard memory "
    "bounds, restart the GUI."
)


def build_release_button(
    tool: ToolName,
    *,
    label: str = "Release loaded data",
) -> html.Div:
    """Build a Release button + status line for ``tool``.

    Args:
        tool: Tool identifier (``"builder"``, ``"viewer"``, ``"run"``,
            ``"analysis"``). Used as part of the pattern-matching ID so
            a single chrome callback can dispatch across all tools.
        label: Button label. Defaults to ``"Release loaded data"`` to
            match the honest-UX wording from plan review.

    Returns:
        A ``html.Div`` wrapping the button + status line. Caller is
        responsible for placing it in the chrome (typically top-right of
        the tool's main pane).
    """
    button_id = release_button_id(tool)
    status_id = release_status_id(tool)
    btn = dbc.Button(
        label,
        id=button_id,
        color="warning",
        outline=True,
        size="sm",
        n_clicks=0,
    )
    tooltip = dbc.Tooltip(_TOOLTIP, target=button_id, placement="bottom")
    status = html.Span(
        "",
        id=status_id,
        className="shell-release-status text-muted ms-2",
    )
    return html.Div(
        [btn, tooltip, status],
        className="shell-release-control d-inline-flex align-items-center",
    )

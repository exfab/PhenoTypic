"""Pure renderer for the timeline grid (placeholders + row-major key order).

Renders a CSS grid sized to the full matrix (corner + time-column headers +
per-row [row-header, cells…]). Every data cell is a SIZE-MATCHED PLACEHOLDER
``html.Div`` carrying ``data-src`` (the thumbnail URL) and identity data-attrs
— NO ``<img>`` enters the DOM here; the virtualization JS (a later phase)
mounts/unmounts the image on scroll. Returns the component plus the row-major
list of non-empty ``(row_value, time_value)`` keys, mirroring
``colony_view.build_grid``'s ``grid_order`` so selection ranges resolve the
same way.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dash import html
from dash.development.base_component import Component

from phenotypic.gui._config import TIMELINE_GRID_GAP_PX
from phenotypic.gui._shared.timeline._matrix import TimelineCell, TimelineMatrix


def build_timeline_grid(
    matrix: TimelineMatrix,
    *,
    url_builder: Callable[[object, int], str],
    display_size: int,
    fetch_size: int,
    gap_px: int = TIMELINE_GRID_GAP_PX,
    ref_builder: Callable[[object], str] | None = None,
) -> tuple[Component, list[tuple[str, str]]]:
    """Render the timeline grid component and its row-major key order.

    Args:
        matrix: The ordered matrix from :func:`build_matrix`.
        url_builder: ``(representative_ref, fetch_size) -> thumbnail URL``,
            written into each placeholder's ``data-src``.
        display_size: CSS tile size (px) — the rendered placeholder size.
        fetch_size: Snapped thumbnail bucket (px) passed to ``url_builder``.
        gap_px: CSS gap between tiles.
        ref_builder: Optional ``representative_ref -> str`` written into each
            cell's ``data-ref`` (the surface's opaque identity for pop-out /
            deep-zoom — Browse encodes a token, Results a ``"dataset/stem"``).
            Defaults to ``str(representative)``.

    Returns:
        ``(component, grid_order)`` where ``grid_order`` is the row-major list
        of non-empty ``(row_value, time_value)`` keys.
    """
    children: list[Component] = [html.Div(className="timeline-grid-corner")]
    for col_index, time_value in enumerate(matrix.columns):
        # `data-*` HTML attributes can't be expressed in Dash's typed Div
        # kwargs, so unpack via an Any-typed dict to bypass the stub mismatch
        # (mirrors gui/_shared/tiles.py's extra_props convention).
        # Axis labels carry their value + index so the JS can match a header
        # click to its column/row of cells without fragile textContent
        # matching (Compare strip row/column triggers, §7).
        x_label_props: dict[str, Any] = {
            "data-col": time_value,
            "data-col-index": str(col_index),
        }
        children.append(
            html.Div(
                time_value,
                className="timeline-axis-label timeline-axis-label--x",
                **x_label_props,
            )
        )

    grid_order: list[tuple[str, str]] = []
    for row_index, row_value in enumerate(matrix.rows):
        y_label_props: dict[str, Any] = {
            "data-row": row_value,
            "data-row-index": str(row_index),
        }
        children.append(
            html.Div(
                row_value,
                className="timeline-axis-label timeline-axis-label--y",
                **y_label_props,
            )
        )
        for col_index, time_value in enumerate(matrix.columns):
            cell = matrix.cells.get((row_value, time_value))
            if cell is None:
                # Every grid coordinate is addressable by the focus controller.
                empty_props: dict[str, Any] = {
                    "data-row-index": str(row_index),
                    "data-col-index": str(col_index),
                }
                children.append(
                    html.Div(
                        className="timeline-cell timeline-cell--empty",
                        style={"width": f"{display_size}px", "height": f"{display_size}px"},
                        **empty_props,
                    )
                )
                continue
            grid_order.append((row_value, time_value))
            children.append(
                _build_cell(
                    cell,
                    url_builder,
                    display_size,
                    fetch_size,
                    ref_builder,
                    row_index=row_index,
                    col_index=col_index,
                )
            )

    grid = html.Div(
        children,
        className="timeline-grid",
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"minmax(0, {display_size}px) "
                + " ".join([f"{display_size}px"] * len(matrix.columns))
            ),
            "gap": f"{gap_px}px",
            "width": "max-content",
        },
    )
    return grid, grid_order


def _build_cell(
    cell: TimelineCell,
    url_builder: Callable[[object, int], str],
    display_size: int,
    fetch_size: int,
    ref_builder: Callable[[object], str] | None,
    *,
    row_index: int,
    col_index: int,
) -> Component:
    """Render one placeholder cell (no <img>; data-src drives focus-window mount)."""
    ref = ref_builder(cell.representative) if ref_builder else str(cell.representative)
    data_props: dict[str, Any] = {
        "data-src": url_builder(cell.representative, fetch_size),
        "data-ref": ref,
        "data-row": cell.row_value,
        "data-col": cell.time_value,
        "data-key": f"{cell.row_value}::{cell.time_value}",
        # Grid coordinates for the focus-navigate controller (spec §16.8).
        "data-row-index": str(row_index),
        "data-col-index": str(col_index),
    }
    inner: list[Component] = [
        # Hover-revealed via CSS (.timeline-cell:hover .timeline-cell-popout);
        # focus + Enter also opens the pop-out (spec §16.4).
        html.Button(
            "⤢",
            className="timeline-cell-popout",
            title="Open full-resolution view",
            type="button",
            n_clicks=0,
        )
    ]
    if cell.count > 1:
        inner.append(html.Span(f"N={cell.count}", className="timeline-cell-badge"))
    return html.Div(
        inner,
        className="timeline-cell",
        style={
            "width": f"{display_size}px",
            "height": f"{display_size}px",
            "position": "relative",
        },
        **data_props,
    )

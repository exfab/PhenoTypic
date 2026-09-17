"""Before/after table preview for ``PostMeasurement`` operations.

Post operations edit metadata columns (``PrependString``,
``AppendString``, ``ExpandMetadata``, ``MergeMetadata``). The analysis
GUI's stepper renders, for each authored post, a small table with one
row per *affected* column showing the column name, the top-5 values
*before* the op, and the top-5 values *after*. Authoring decisions
take effect once the user re-runs the CLI (the analysis GUI itself
does not re-apply post during ``.analyze()``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from dash import html

if TYPE_CHECKING:
    from phenotypic.abc_._post_measurement import PostMeasurement

logger = logging.getLogger(__name__)

#: Number of top values to show in each preview column.
TOP_N: int = 5


def render_post_preview(
    post_op: "PostMeasurement",
    df: pd.DataFrame,
) -> Any:
    """Build a Dash component summarising *post_op*'s effect on *df*.

    Applies ``post_op.apply`` to a copy of *df* and diffs the resulting
    columns against the input. Columns whose top-5 values changed (or
    new columns the op introduced) are shown side-by-side in a small
    table.

    Args:
        post_op: A :class:`PostMeasurement` instance. Must be safe to
            apply to *df* — if it raises, an error card is returned.
        df: The aggregate measurements frame the preview runs against.

    Returns:
        A ``dash.html.Div`` containing the preview table or an error
        card.
    """
    if df.empty:
        return html.Div(
            "No measurements available for preview.",
            className="analysis-post-preview-empty",
        )

    try:
        after = post_op.apply(df.copy(deep=True))
    except Exception as exc:  # noqa: BLE001
        logger.warning("post_op.apply raised: %s", exc)
        return html.Div(
            [
                html.Strong("Preview unavailable"),
                html.Pre(str(exc), className="analysis-error-pre"),
            ],
            className="analysis-error-card",
        )

    affected = _affected_columns(df, after)
    if not affected:
        return html.Div(
            "No metadata changes detected for this op.",
            className="analysis-post-preview-empty",
        )

    rows = [
        html.Tr(
            [
                html.Th("Column"),
                html.Th(f"Before (top {TOP_N})"),
                html.Th(f"After (top {TOP_N})"),
            ]
        )
    ]
    for col in affected:
        before_values = _top_values(df.get(col), TOP_N)
        after_values = _top_values(after.get(col), TOP_N)
        rows.append(
            html.Tr(
                [
                    html.Td(col, className="analysis-post-col"),
                    html.Td(", ".join(before_values)),
                    html.Td(", ".join(after_values)),
                ]
            )
        )

    return html.Table(rows, className="analysis-post-preview-table")


def _affected_columns(
    before: pd.DataFrame, after: pd.DataFrame
) -> list[str]:
    """Columns whose top-5 values differ between *before* and *after*.

    Includes columns that exist only in *after* (new columns the op
    introduced — e.g. :class:`ExpandMetadata`).
    """
    cols: list[str] = []
    seen: set[str] = set()
    for col in after.columns:
        if col in seen:
            continue
        seen.add(col)
        if col not in before.columns:
            cols.append(col)
            continue
        before_top = _top_values(before[col], TOP_N)
        after_top = _top_values(after[col], TOP_N)
        if before_top != after_top:
            cols.append(col)
    return cols


def _top_values(series: pd.Series | None, n: int) -> list[str]:
    """Return the *n* most common values of *series* as strings."""
    if series is None:
        return []
    if series.empty:
        return []
    try:
        counts = series.value_counts(dropna=False).head(n)
    except TypeError:
        # Unhashable elements — fall back to head() of unique values.
        try:
            return [str(v) for v in list(series.unique())[:n]]
        except Exception:
            return []
    return [_safe_str(idx) for idx in counts.index]


def _safe_str(value: object) -> str:
    """Convert a value to a string for table display, handling NA cleanly."""
    if pd.isna(value):
        return "<NA>"
    s = str(value)
    if len(s) > 40:
        return s[:37] + "..."
    return s


__all__ = ["render_post_preview", "TOP_N"]

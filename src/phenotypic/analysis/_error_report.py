"""Shared error-analysis HTML report + copy-able filter-spec renderers.

Pure pandas/json/string builders, **Dash-free and Plotly-free** so both
the Error-analysis tab's "Save analysis report" button (Phase 4) and CLI
finalize (Phase 5) render the same artifact without pulling the Dash
stack into a headless run — and without a ``gui→cli`` import inversion
(both import *down* into :mod:`phenotypic.analysis`).

The HTML is a self-contained document: a heading naming the focused error
category plus the ranked :meth:`ErrorCutoffFinder.analyze` result table.
The filter-spec helpers emit the machine-readable JSON and the
human-readable query string the user copies to filter similar bad data.
"""
from __future__ import annotations

import html as _html
import json

import pandas as pd

#: Minimal inline stylesheet so the saved report reads cleanly standalone.
_REPORT_STYLE = """
body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { font-size: 1.4rem; }
table { border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }
th, td { border: 1px solid #d0d0d0; padding: 0.35rem 0.6rem; text-align: right; }
th { background: #f3f4f6; text-align: center; }
td:first-child, th:first-child { text-align: left; }
""".strip()


def _render_section(category: str, result_df: pd.DataFrame) -> str:
    """Render one category's heading + ranked result table (a report fragment).

    Args:
        category: The error-category token (e.g. ``"debris"``).
        result_df: The ranked :meth:`ErrorCutoffFinder.analyze` frame for
            ``category`` (may be empty — a placeholder paragraph is emitted).

    Returns:
        An HTML fragment (``<h1>`` + table or placeholder), **not** a full
        document — :func:`render_error_analysis_html` and
        :func:`render_error_analysis_report` wrap one or more sections.
    """
    safe_category = _html.escape(str(category))
    if result_df.empty:
        table_html = "<p>No discriminative measurements for this category.</p>"
    else:
        table_html = result_df.to_html(
            index=False, border=0, float_format=lambda v: f"{v:.4g}"
        )
    return f"<h1>Error analysis — {safe_category}</h1>\n{table_html}\n"


def _wrap_document(title: str, body: str) -> str:
    """Wrap one or more report sections in a self-contained ``<html>`` document."""
    safe_title = _html.escape(str(title))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{safe_title}</title>\n"
        f"<style>\n{_REPORT_STYLE}\n</style>\n</head>\n<body>\n"
        f"{body}"
        "</body>\n</html>\n"
    )


def render_error_analysis_html(category: str, result_df: pd.DataFrame) -> str:
    """Render a self-contained HTML report for one error category.

    Args:
        category: The focused error-category token (e.g. ``"debris"``).
        result_df: The ranked :meth:`ErrorCutoffFinder.analyze` frame for
            ``category`` (may be empty — an empty-table report is still a
            valid document).

    Returns:
        A complete ``<html>`` document string: a heading naming the
        category plus the ranked result table. Pure string build (pandas
        ``to_html`` + a small inline ``<style>``); no Plotly/Dash import.
    """
    return _wrap_document(
        f"Error analysis — {category}", _render_section(category, result_df)
    )


def render_error_analysis_report(results_by_category: dict[str, pd.DataFrame]) -> str:
    """Render one self-contained HTML report with a section per category.

    Used by headless CLI finalize, which writes a single
    ``deliverables/error_analysis.html`` covering **every** labeled category
    (the GUI's :func:`render_error_analysis_html` is single-category and
    transient). Sections are ordered by category token for a stable artifact.

    Args:
        results_by_category: Mapping ``category token -> ErrorCutoffFinder
            result frame`` (category-free ``RESULT_COLUMNS``). An empty mapping
            yields a valid "no error categories" document.

    Returns:
        A full ``<html>`` document; a "no error categories" body when empty.
        Pure string build; no Plotly/Dash import.
    """
    if not results_by_category:
        body = "<h1>Error analysis</h1>\n<p>No error categories were labeled.</p>\n"
        return _wrap_document("Error analysis", body)
    sections = [
        _render_section(category, results_by_category[category])
        for category in sorted(results_by_category)
    ]
    return _wrap_document("Error analysis", "".join(sections))


def filter_spec_json(measurement: str, direction: str, cutoff: float) -> str:
    """Return the machine-readable filter spec as indented JSON.

    Args:
        measurement: The measurement column the cutoff applies to.
        direction: The comparison operator (``">"`` or ``"<"``).
        cutoff: The decision threshold.

    Returns:
        ``{"measurement": "...", "op": ">", "cutoff": 123.4}`` as a
        2-space-indented JSON string.
    """
    return json.dumps(
        {"measurement": measurement, "op": direction, "cutoff": cutoff},
        indent=2,
    )


def filter_spec_query(measurement: str, direction: str, cutoff: float) -> str:
    """Return the human-readable filter expression.

    Args:
        measurement: The measurement column the cutoff applies to.
        direction: The comparison operator (``">"`` or ``"<"``).
        cutoff: The decision threshold.

    Returns:
        A ``Size_Area > 123.40`` style expression (cutoff to 2 d.p.).
    """
    return f"{measurement} {direction} {cutoff:.2f}"


__all__ = [
    "filter_spec_json",
    "filter_spec_query",
    "render_error_analysis_html",
    "render_error_analysis_report",
]

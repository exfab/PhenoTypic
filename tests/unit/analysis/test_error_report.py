"""Tests for the shared error-analysis HTML + filter-spec renderers.

These live in ``phenotypic.analysis`` (R1) so both the Error tab's
"Save analysis report" button and CLI finalize (Phase 5) render the same
HTML without a gui→cli import inversion. The module is Dash-free and
Plotly-free.
"""

from __future__ import annotations

import json

import pandas as pd

from phenotypic.analysis import (
    filter_spec_json,
    filter_spec_query,
    render_error_analysis_html,
)


def _result_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measurement": ["Size_Area", "Shape_Circularity"],
            "auc": [0.98, 0.71],
            "direction": [">", "<"],
            "cutoff": [123.4, 0.5],
            "recall": [0.9, 0.6],
            "specificity": [0.95, 0.8],
        }
    )


def test_render_html_contains_category_and_measurements():
    html = render_error_analysis_html("debris", _result_df())
    assert "<html" in html.lower()
    assert "debris" in html
    assert "Size_Area" in html
    assert "Shape_Circularity" in html


def test_render_html_handles_empty_result():
    html = render_error_analysis_html("debris", _result_df().iloc[0:0])
    assert "<html" in html.lower()
    assert "debris" in html


def test_filter_spec_json_roundtrips():
    spec = filter_spec_json("Size_Area", ">", 123.4)
    parsed = json.loads(spec)
    assert parsed == {"measurement": "Size_Area", "op": ">", "cutoff": 123.4}


def test_filter_spec_query_formats_expression():
    assert filter_spec_query("Size_Area", ">", 123.4) == "Size_Area > 123.40"

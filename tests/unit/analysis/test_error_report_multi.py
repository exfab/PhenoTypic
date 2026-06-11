"""Tests for the multi-category ``render_error_analysis_report``.

Headless CLI finalize writes ONE ``error_analysis.html`` covering every labeled
category (the GUI's single-category ``render_error_analysis_html`` is transient).
The report reuses the per-category table rendering and stays Dash-free +
Plotly-free.
"""

from __future__ import annotations

import pandas as pd

from phenotypic.analysis import render_error_analysis_report


def _debris_df() -> pd.DataFrame:
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


def _noise_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measurement": ["Intensity_MeanIntensity"],
            "auc": [0.88],
            "direction": ["<"],
            "cutoff": [12.0],
            "recall": [0.8],
            "specificity": [0.9],
        }
    )


def test_report_contains_all_categories_and_their_measurements() -> None:
    html = render_error_analysis_report(
        {"debris": _debris_df(), "background_noise": _noise_df()}
    )
    assert "<html" in html.lower()
    # Both category headings present.
    assert "debris" in html
    assert "background_noise" in html
    # Measurement names from BOTH sections present.
    assert "Size_Area" in html
    assert "Shape_Circularity" in html
    assert "Intensity_MeanIntensity" in html


def test_report_handles_empty_category_frame() -> None:
    """A category with an empty result frame still gets a heading + a placeholder."""
    html = render_error_analysis_report({"debris": _debris_df().iloc[0:0]})
    assert "<html" in html.lower()
    assert "debris" in html


def test_report_empty_dict_is_valid_no_categories_document() -> None:
    html = render_error_analysis_report({})
    assert "<html" in html.lower()
    # A valid "no categories" body, not a crash.
    assert "</html>" in html.lower()
    # No stray measurement content.
    assert "Size_Area" not in html

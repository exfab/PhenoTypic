"""Unit tests for the QC tab's pure callback helpers.

The helpers live as module-level functions inside
``_qc_tab._callbacks`` so they can be tested without booting a Dash
app. Each test constructs a synthetic input inline.
"""
from __future__ import annotations

import re

import pandas as pd
import polars as pl

from phenotypic.gui.results_viewer._qc_tab._callbacks import (
    _badge_color_for_status,
    _left_join_qc_columns,
    _merge_removed_keys,
    _render_summary_strip,
    _worst_status,
)


def test_left_join_qc_columns_preserves_left_rows() -> None:
    """A left-join over partial QC results keeps every left row."""
    left = pl.DataFrame(
        {
            "Metadata_ImageFile": [f"img_{i}.tif" for i in range(10)],
            "ObjectLabel": list(range(10)),
            "Size_Area": [float(100 + i) for i in range(10)],
        }
    )
    right = pd.DataFrame(
        {
            "Metadata_ImageFile": ["img_0.tif", "img_1.tif", "img_2.tif"],
            "ObjectLabel": [0, 1, 2],
            "QC_SE_Severity": [0.01, 0.05, 0.12],
        }
    )
    result = _left_join_qc_columns(left, right)
    assert result.height == 10
    assert "QC_SE_Severity" in result.columns
    # Non-matching rows should be NaN/null.
    sev = result["QC_SE_Severity"].to_list()
    assert sev[0] is not None
    assert sev[9] is None


def test_render_summary_strip_format() -> None:
    """The summary strip text matches the documented shape."""
    summary = pd.DataFrame(
        {
            "Plate": ["P1", "P2", "P3"],
            "num_rows": [4, 4, 4],
            "num_flagged": [1, 0, 0],
            "max_severity": [0.12, 0.04, 0.02],
            "status": ["fail", "pass", "pass"],
        }
    )
    text = _render_summary_strip(summary)
    assert re.match(
        r"groups:\s+\d+\s+\|\s+flagged:\s+\d+\s+\|\s+max severity:\s+\d+\.\d+",
        text,
    ), f"unexpected summary text: {text!r}"


def test_worst_status_pass_warn_fail() -> None:
    """``fail`` wins over ``warn`` which wins over ``pass``."""
    pass_only = pd.DataFrame({"status": ["pass", "pass"]})
    pass_warn = pd.DataFrame({"status": ["pass", "warn"]})
    pass_warn_fail = pd.DataFrame({"status": ["pass", "warn", "fail"]})
    assert _worst_status(pass_only) == "pass"
    assert _worst_status(pass_warn) == "warn"
    assert _worst_status(pass_warn_fail) == "fail"


def test_badge_color_for_status() -> None:
    """Each status maps to the documented Bootstrap colour name."""
    assert _badge_color_for_status("pass") == "success"
    assert _badge_color_for_status("warn") == "warning"
    assert _badge_color_for_status("fail") == "danger"


def test_merge_removed_keys_dedupes() -> None:
    """Union of two payloads produces no duplicates."""
    current: list[list] = [["a", 1], ["b", 2]]
    new = [("b", 2), ("c", 3)]
    merged = _merge_removed_keys(current, new)
    assert merged == [["a", 1], ["b", 2], ["c", 3]]

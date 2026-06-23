"""timeline_view package public API surface."""
from __future__ import annotations

import phenotypic.gui.results_viewer.timeline_view as tv


def test_public_api_is_exported() -> None:
    expected = {
        "layout",
        "register_callbacks",
        "selectable_time_columns",
        "is_large_time_axis",
        "has_eligible_time_axis",
        "build_timeline_records",
    }
    assert expected.issubset(set(tv.__all__))
    for name in expected:
        assert hasattr(tv, name)

"""The timeline engine package exposes its public API from __init__."""
from __future__ import annotations

import phenotypic.gui._shared.timeline as timeline


def test_public_api_is_exported() -> None:
    expected = {
        "build_matrix",
        "TimelineMatrix",
        "TimelineCell",
        "downscale_to_thumb",
        "register_thumbnail_route",
        "ThumbUnavailable",
        "thumb_cache_name",
        "build_timeline_grid",
    }
    assert expected.issubset(set(timeline.__all__))
    for name in expected:
        assert hasattr(timeline, name)


def test_compare_helpers_are_exported() -> None:
    for name in ("compare_selection_plan", "ComparePlan"):
        assert name in timeline.__all__
        assert hasattr(timeline, name)

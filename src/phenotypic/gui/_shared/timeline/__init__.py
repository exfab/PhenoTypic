"""Source-agnostic timeline-view engine (matrix model, thumbnail route, grid).

Consumed by the Browse and Results timeline surfaces (later phases). Mirrors
``gui/_shared/tiles.py`` as the single owner of the matrix model, the cached
thumbnail route factory, and the placeholder-grid renderer.
"""
from __future__ import annotations

from phenotypic.gui._shared.timeline._compare import (
    ComparePlan,
    compare_selection_plan,
)
from phenotypic.gui._shared.timeline._grid import build_timeline_grid
from phenotypic.gui._shared.timeline._matrix import (
    TimelineCell,
    TimelineMatrix,
    build_matrix,
)
from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    downscale_to_thumb,
    register_thumbnail_route,
    thumb_cache_name,
)

__all__ = [
    "build_matrix",
    "TimelineMatrix",
    "TimelineCell",
    "downscale_to_thumb",
    "register_thumbnail_route",
    "ThumbUnavailable",
    "thumb_cache_name",
    "build_timeline_grid",
    "compare_selection_plan",
    "ComparePlan",
]

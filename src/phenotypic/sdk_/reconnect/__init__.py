"""Pure numerical helpers for reconnecting filamentary image structure.

This package stays independent of :mod:`phenotypic` image and operation classes.
Domain adapters belong in their callers, while public helpers are re-exported here
only after their source-fidelity review gate passes.
"""

from ._cellular_automaton import TrickTrackCAResult, tricktrack_ca
from ._gwdt import app2_gwdt_cost, grey_weighted_distance
from ._nfa import NFAResult, binomial_nfa
from ._rolling_hough import ClarkRollingHoughResult, clark_rolling_hough
from ._rorpo import RorpoResult, rorpo
from ._tensor_voting import tensor_vote
from ._colony_labeling import (
    filter_mask_by_overlap,
    markers_from_centroids,
    partition_by_grid_voronoi,
)
from ._colony_reconnect import (
    ReconnectConfig,
    build_reconnect_cost,
    compute_full_image_app2_gi_cost,
    identify_pseudo_fragments,
    reconnect_fragments_tiled,
    select_reconnect_fragments,
)

__all__ = [
    "ClarkRollingHoughResult",
    "NFAResult",
    "ReconnectConfig",
    "RorpoResult",
    "TrickTrackCAResult",
    "app2_gwdt_cost",
    "binomial_nfa",
    "build_reconnect_cost",
    "clark_rolling_hough",
    "compute_full_image_app2_gi_cost",
    "filter_mask_by_overlap",
    "grey_weighted_distance",
    "identify_pseudo_fragments",
    "markers_from_centroids",
    "partition_by_grid_voronoi",
    "reconnect_fragments_tiled",
    "rorpo",
    "select_reconnect_fragments",
    "tensor_vote",
    "tricktrack_ca",
]

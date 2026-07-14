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

__all__ = [
    "ClarkRollingHoughResult",
    "NFAResult",
    "RorpoResult",
    "TrickTrackCAResult",
    "app2_gwdt_cost",
    "binomial_nfa",
    "clark_rolling_hough",
    "grey_weighted_distance",
    "rorpo",
    "tensor_vote",
    "tricktrack_ca",
]

"""Back-compat shim — the implementation now lives in
``phenotypic.tools_.branch_pathfinding``.

This module re-exports the full legacy surface so in-tree callers do not
break. The star-import captures public names; the explicit block captures
the underscored private helpers (``_apply_*_inplace``,
``_compute_screening_envelope``) that were also imported directly by
``FilamentousFungiDetector``.

TODO: drop after downstream callers migrate to
``phenotypic.tools_.branch_pathfinding`` (track removal in a minor-version
milestone).
"""

from phenotypic.tools_.branch_pathfinding import *  # noqa: F401,F403
from phenotypic.tools_.branch_pathfinding import (  # noqa: F401
    _apply_border_penalty_inplace,
    _apply_distance_gap_penalty_inplace,
    _apply_structure_mask_inplace,
    _compute_screening_envelope,
)
from phenotypic.tools_.branch_pathfinding import __all__ as _new_all

__all__ = list(_new_all)

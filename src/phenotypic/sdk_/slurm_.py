"""SLURM headroom calculation and submission validation utilities.

.. deprecated::
    This module re-exports from ``phenotypic.sdk_.slurm._slurm_headroom``
    for backward compatibility.  Import directly from that module instead.
"""

from .slurm._slurm_headroom import (  # noqa: F401
    INFINITY_STRINGS,
    UNIT_MAP,
    fetch_slurm_context,
    get_current_footprint,
    get_headroom,
    get_partition_stats,
    get_user_association,
    parse_slurm_value,
    validate_submission,
)

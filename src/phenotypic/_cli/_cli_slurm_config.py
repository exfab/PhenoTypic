"""SLURM configuration querying and array job chunking logic.

.. deprecated::
    This module re-exports from ``phenotypic.tools_.slurm._config``
    for backward compatibility.  Import directly from that module instead.
"""

from phenotypic.tools_.slurm._config import (  # noqa: F401
    calculate_optimal_array_chunks,
    estimate_concurrent_capacity,
    get_slurm_array_limit,
    get_slurm_max_submit_jobs,
    validate_array_chunk,
)

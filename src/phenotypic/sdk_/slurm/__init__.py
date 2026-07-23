"""Shared SLURM utilities for array job chunking, sbatch submission, and dispatching.

This subpackage consolidates SLURM logic used by the main CLI
(``phenotypic._cli``).
"""

from ._config import (
    calculate_optimal_array_chunks,
    estimate_concurrent_capacity,
    get_slurm_array_limit,
    get_slurm_max_submit_jobs,
    validate_array_chunk,
)
from ._sbatch import (
    format_sbatch_directives,
    parse_job_id,
    parse_slurm_time,
    submit_script,
)
from ._dispatcher import (
    generate_dispatcher_chain,
    generate_dispatcher_script,
    submit_drip_feed_start,
)
from ._script_rendering import SlurmArrayScriptSpec, write_slurm_array_script

__all__ = [
    "SlurmArrayScriptSpec",
    "calculate_optimal_array_chunks",
    "estimate_concurrent_capacity",
    "format_sbatch_directives",
    "generate_dispatcher_chain",
    "generate_dispatcher_script",
    "get_slurm_array_limit",
    "get_slurm_max_submit_jobs",
    "parse_job_id",
    "parse_slurm_time",
    "submit_drip_feed_start",
    "submit_script",
    "validate_array_chunk",
    "write_slurm_array_script",
]

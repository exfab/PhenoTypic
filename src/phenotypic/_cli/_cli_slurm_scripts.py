"""SLURM bash script generation for the PhenoTypic CLI.

This module generates standalone bash scripts for autonomous SLURM execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from phenotypic.sdk_.slurm._sbatch import (
    format_sbatch_directives as _format_sbatch_directives,
)

logger = logging.getLogger(__name__)


def generate_slurm_directives(
    job_name: str, slurm_args: Dict[str, Any], output_log: Path, error_log: Path
) -> str:
    """Generate SBATCH directive lines for SLURM script.

    Delegates to ``phenotypic.sdk_.slurm._sbatch.format_sbatch_directives``.

    Args:
        job_name: Job name.
        slurm_args: SLURM parameters dict.
        output_log: Path for stdout log.
        error_log: Path for stderr log.

    Returns:
        String with all ``#SBATCH`` directives.
    """
    return _format_sbatch_directives(
        job_name=job_name,
        slurm_args=slurm_args,
        output_log=output_log,
        error_log=error_log,
    )

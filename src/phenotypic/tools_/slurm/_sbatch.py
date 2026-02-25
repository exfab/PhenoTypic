"""Shared sbatch submission helpers.

Provides utilities for formatting SBATCH directives, parsing job IDs
from sbatch output, and submitting scripts to SLURM.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# SBATCH directive names managed by script generators; user overrides are ignored.
_RESERVED_SBATCH_KEYS = frozenset({"array", "output", "error", "job-name"})


def format_sbatch_directives(
    job_name: str,
    slurm_args: Dict[str, Any],
    output_log: Path,
    error_log: Path,
) -> str:
    """Generate ``#SBATCH`` directive lines for a SLURM script.

    Converts CLI SLURM parameters to ``#SBATCH`` directives with proper
    formatting. Reserved keys (``array``, ``output``, ``error``,
    ``job-name``) are silently skipped because they are managed by the
    script generators.

    Args:
        job_name: Job name for ``--job-name``.
        slurm_args: SLURM parameters dict (CLI-style keys like
            ``slurm_partition``, ``mem_gb``, ``time``).
        output_log: Path for stdout log.
        error_log: Path for stderr log.

    Returns:
        String with all ``#SBATCH`` directives joined by newlines.

    Notes:
        - Time parameters (``time``, ``slurm_time``) as integers are
          treated as minutes and converted to ``HH:MM:SS``.
        - ``mem_gb`` is converted to ``--mem=<N>G``.
    """
    directives = [f"#SBATCH --job-name={job_name}"]
    directives.append(f"#SBATCH --output={output_log}")
    directives.append(f"#SBATCH --error={error_log}")

    for key, value in slurm_args.items():
        directive_name = key.replace("slurm_", "").replace("_", "-")

        if directive_name in _RESERVED_SBATCH_KEYS:
            logger.warning(
                "Ignoring user --slurm %s=%s: '%s' is managed by PhenoTypic",
                key,
                value,
                directive_name,
            )
            continue

        if key in ("time", "slurm_time"):
            if isinstance(value, int):
                hours = value // 60
                minutes = value % 60
                value = f"{hours:02d}:{minutes:02d}:00"
            directive_name = "time"
        elif key == "mem_gb":
            value = f"{value}G"
            directive_name = "mem"
        elif key == "slurm_mem":
            directive_name = "mem"
        elif key == "slurm_mem_per_cpu":
            directive_name = "mem-per-cpu"
        elif key == "slurm_cpus_per_task":
            directive_name = "cpus-per-task"
        elif key == "slurm_gpus_per_node":
            directive_name = "gpus-per-node"

        directives.append(f"#SBATCH --{directive_name}={value}")

    return "\n".join(directives)


def parse_job_id(sbatch_stdout: str) -> str:
    """Extract the SLURM job ID from sbatch output.

    Args:
        sbatch_stdout: Standard output from an ``sbatch`` command,
            typically ``"Submitted batch job 12345\\n"``.

    Returns:
        The job ID as a string.

    Raises:
        RuntimeError: If the job ID cannot be parsed from the output.

    Examples:
        >>> parse_job_id("Submitted batch job 12345\\n")
        '12345'
    """
    match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output:\n{sbatch_stdout}"
        )
    return match.group(1)


def submit_script(
    script_path: Path,
    dependency_job_id: Optional[str] = None,
) -> str:
    """Submit a script to SLURM via ``sbatch`` and return the job ID.

    Args:
        script_path: Path to the SLURM batch script.
        dependency_job_id: When set, adds
            ``--dependency=afterany:<id>`` so this job starts only
            after the dependency finishes.

    Returns:
        SLURM job ID string.

    Raises:
        RuntimeError: If ``sbatch`` is not available, the submission
            fails, or the job ID cannot be parsed.
    """
    cmd = ["sbatch", "--parsable"]

    if dependency_job_id:
        cmd.extend(["--dependency", f"afterany:{dependency_job_id}"])

    cmd.append(str(script_path))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
    except FileNotFoundError:
        raise RuntimeError(
            "sbatch command not found. SLURM does not appear to be available. "
            "Use --force-local to run locally instead."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"sbatch submission timed out for script: {script_path.name}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"sbatch submission failed for {script_path.name}:\n{e.stderr}"
        )

    # --parsable makes sbatch output just the job ID (possibly with cluster name)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        # Fallback to regex parsing for non-parsable output
        return parse_job_id(result.stdout)

    return job_id

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
_SLURM_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+)-(?P<day_hours>\d{2})|(?P<hours>\d{2,})):"
    r"(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)$"
)


def parse_slurm_time(value: object) -> str | None:
    """Validate and canonicalize a SLURM time limit.

    Args:
        value: Empty input, positive integer minutes, or a SLURM duration in
            ``HH:MM:SS`` or ``D-HH:MM:SS`` form.

    Returns:
        Canonical SLURM duration, or ``None`` when ``value`` is empty.

    Raises:
        ValueError: If ``value`` is not one of the supported forms, contains
            an invalid clock field, or represents a nonpositive duration.

    Examples:
        >>> parse_slurm_time(90)
        '01:30:00'
        >>> parse_slurm_time("00:10:00")
        '00:10:00'
        >>> parse_slurm_time("1-04:00:00")
        '1-04:00:00'
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("SLURM time must be positive minutes or a duration")

    if isinstance(value, int):
        minutes = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdecimal():
            minutes = int(text)
        else:
            match = _SLURM_DURATION_RE.fullmatch(text)
            if match is None:
                raise ValueError(
                    "SLURM time must be positive integer minutes, HH:MM:SS, "
                    "or D-HH:MM:SS"
                )
            days_text = match.group("days")
            hours = int(
                match.group("day_hours")
                if days_text is not None
                else match.group("hours")
            )
            minutes_field = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
            if days_text is not None and hours > 23:
                raise ValueError(
                    "SLURM D-HH:MM:SS time requires an HH field from 00 to 23"
                )
            days = int(days_text) if days_text is not None else 0
            total_seconds = (
                ((days * 24 + hours) * 60 + minutes_field) * 60 + seconds
            )
            if total_seconds <= 0:
                raise ValueError("SLURM time must be greater than zero")
            if days_text is None:
                return f"{hours:02d}:{minutes_field:02d}:{seconds:02d}"
            return f"{days}-{hours:02d}:{minutes_field:02d}:{seconds:02d}"
    else:
        raise ValueError(
            "SLURM time must be positive integer minutes, HH:MM:SS, "
            "or D-HH:MM:SS"
        )

    if minutes <= 0:
        raise ValueError("SLURM time in minutes must be greater than zero")
    hours, minute_field = divmod(minutes, 60)
    return f"{hours:02d}:{minute_field:02d}:00"


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
    directives.append(f"#SBATCH --output={output_log.as_posix()}")
    directives.append(f"#SBATCH --error={error_log.as_posix()}")

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
            value = parse_slurm_time(value)
            if value is None:
                continue
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
    array_index: Optional[int] = None,
) -> str:
    """Submit a script to SLURM via ``sbatch`` and return the job ID.

    Args:
        script_path: Path to the SLURM batch script.
        dependency_job_id: When set, adds
            ``--dependency=afterany:<id>`` so this job starts only
            after the dependency finishes.
        array_index: When set, overrides any script array directive and submits
            only this array index.

    Returns:
        SLURM job ID string.

    Raises:
        RuntimeError: If ``sbatch`` is not available, the submission
            fails, or the job ID cannot be parsed.
    """
    cmd = ["sbatch", "--parsable"]

    if dependency_job_id:
        cmd.extend(["--dependency", f"afterany:{dependency_job_id}"])
    if array_index is not None:
        cmd.extend(["--array", str(array_index)])

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

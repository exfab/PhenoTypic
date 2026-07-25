"""SLURM batch script generation for the sentinel job.

.. deprecated::
    The standalone sentinel job has been replaced by nonterminal checkpoint
    tasks embedded in array scripts plus a scheduler-dependent terminal
    finalizer. See :mod:`_cli_checkpoint_handler` and
    :mod:`_cli_slurm_array_scripts`. This module is retained only for
    in-flight runs that may still reference it.

This module generates the bash script that runs the sentinel Click command
as a self-resubmitting SLURM job.
"""

from __future__ import annotations

import logging
import shlex
import stat
from pathlib import Path
from typing import Any, Dict

from ._cli_slurm_scripts import generate_slurm_directives
from ._cli_utils import get_python_command
from phenotypic.sdk_ import logs_dir, slurm_scripts_dir
from phenotypic.sdk_.slurm import SLURM_PYTHONPATH_BOOTSTRAP_BASH

logger = logging.getLogger(__name__)


def generate_sentinel_script(
    output_dir: Path,
    progress_dir: Path,
    slurm_args: Dict[str, Any],
    interval: int = 60,
    max_runtime: int = 1800,
) -> Path:
    """Generate a SLURM batch script for the sentinel job.

    Args:
        output_dir: Base output directory.
        progress_dir: Directory for progress files.
        slurm_args: SLURM arguments dict (may contain ``slurm_partition``,
            ``slurm_account``, etc.).
        interval: Seconds between manifest rebuilds.
        max_runtime: Max sentinel runtime in seconds.

    Returns:
        Path to the generated sentinel script.
    """
    script_dir = slurm_scripts_dir(output_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "sentinel.sh"

    # Use the same Python command as array job scripts (sys.executable on SLURM)
    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(python_cmd)

    q_output_dir = shlex.quote(str(output_dir.as_posix()))
    q_progress_dir = shlex.quote(str(progress_dir.as_posix()))
    q_script_path = shlex.quote(str(script_path.as_posix()))

    # Override wall time: max_runtime + 15-min margin, 60-min floor
    slurm_minutes = max((max_runtime // 60) + 15, 60)
    sentinel_slurm_args = {
        k: v for k, v in slurm_args.items()
        if k not in ("time", "slurm_time")
    }
    sentinel_slurm_args["time"] = slurm_minutes
    if "slurm_partition" not in sentinel_slurm_args:
        sentinel_slurm_args["slurm_partition"] = "batch"

    log_path = logs_dir(output_dir) / "slurm" / "sentinel_%j.log"
    directives = generate_slurm_directives(
        job_name="pht-sentinel",
        slurm_args=sentinel_slurm_args,
        output_log=log_path,
        error_log=log_path,
    )

    script_content = f"""\
#!/bin/bash
{directives}

# Resubmit sentinel on SIGTERM (sent by SLURM before SIGKILL) unless
# the Python process already handled resubmission.
# Allow Polars to run on nodes without AVX2/BMI2/MOVBE CPU features
export POLARS_SKIP_CPU_CHECK=1

{SLURM_PYTHONPATH_BOOTSTRAP_BASH}

RESUBMIT_MARKER={q_progress_dir}/sentinel_resubmitted
trap 'if [ ! -f "$RESUBMIT_MARKER" ]; then
    echo "SIGTERM received — resubmitting sentinel from trap"
    sbatch --parsable --export=ALL {q_script_path}
fi
exit 0' TERM

{python_str} -m phenotypic._cli._cli_sentinel \\
    --output-dir {q_output_dir} \\
    --progress-dir {q_progress_dir} \\
    --interval {interval} \\
    --max-runtime {max_runtime} \\
    --sentinel-script {q_script_path} \\
    --slurm-partition {slurm_args.get("slurm_partition", "batch")}
"""

    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Generated sentinel script: %s", script_path)
    return script_path

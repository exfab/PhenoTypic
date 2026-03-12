"""SLURM batch script generation for the sentinel job.

This module generates the bash script that runs the sentinel Click command
as a self-resubmitting SLURM job.
"""

from __future__ import annotations

import logging
import shlex
import stat
from pathlib import Path
from typing import Any, Dict

from ._cli_utils import get_python_command

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
    partition = slurm_args.get("slurm_partition", "batch")
    account = slurm_args.get("slurm_account")

    script_dir = output_dir / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "sentinel.sh"

    # Use the same Python command as array job scripts (sys.executable on SLURM)
    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(python_cmd)

    account_line = ""
    if account:
        account_line = f"#SBATCH --account={account}\n"

    q_output_dir = shlex.quote(str(output_dir.as_posix()))
    q_progress_dir = shlex.quote(str(progress_dir.as_posix()))
    q_script_path = shlex.quote(str(script_path.as_posix()))

    # Derive SLURM wall time from max_runtime + 15-min margin, 60-min floor
    slurm_minutes = max((max_runtime // 60) + 15, 60)
    slurm_hours = slurm_minutes // 60
    slurm_mins = slurm_minutes % 60

    script_content = f"""\
#!/bin/bash
#SBATCH --job-name=pheno-sentinel
#SBATCH --partition={partition}
#SBATCH --time={slurm_hours:02d}:{slurm_mins:02d}:00
#SBATCH --mem=512M
#SBATCH --cpus-per-task=1
#SBATCH --output={progress_dir.as_posix()}/sentinel_%j.log
{account_line}
# Resubmit sentinel on SIGTERM (sent by SLURM before SIGKILL) unless
# the Python process already handled resubmission.
RESUBMIT_MARKER={q_progress_dir}/sentinel_resubmitted
trap 'if [ ! -f "$RESUBMIT_MARKER" ]; then
    echo "SIGTERM received — resubmitting sentinel from trap"
    sbatch --parsable {q_script_path}
fi
exit 0' TERM

{python_str} -m phenotypic._cli._cli_sentinel \\
    --output-dir {q_output_dir} \\
    --progress-dir {q_progress_dir} \\
    --interval {interval} \\
    --max-runtime {max_runtime} \\
    --sentinel-script {q_script_path} \\
    --slurm-partition {partition}
"""

    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Generated sentinel script: %s", script_path)
    return script_path

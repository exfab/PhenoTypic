"""Shared rendering helpers for SLURM array scripts."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._sbatch import format_sbatch_directives


@dataclass(frozen=True)
class SlurmArrayScriptSpec:
    """Specification for a bash SLURM array script.

    Args:
        job_name: SLURM job name for ``#SBATCH --job-name``.
        slurm_args: CLI-style SLURM arguments passed to the shared directive
            formatter.
        log_path: Path used for both stdout and stderr logs.
        error_log_path: Optional stderr path. Defaults to ``log_path``.
        task_indices: Values mapped from ``SLURM_ARRAY_TASK_ID`` into the
            current task variable. Strings are shell-quoted.
        body: Bash body executed once the current task variable is available.
        prelude: Optional bash block inserted after strict mode and before the
            task array.
        comments: Optional comment lines inserted after SBATCH directives.
        array_name: Bash array variable name.
        current_var: Bash variable that receives the current array entry.
        missing_task_id_message: Error printed when not running as an array job.
        bounds_error_message: Error printed when the array task id is out of
            bounds. When omitted, a message is derived from ``array_name``.
        signal_grace: Optional seconds for ``#SBATCH --signal=B:TERM@N``.
        requeue: Whether to include ``#SBATCH --requeue``.
    """

    job_name: str
    slurm_args: Mapping[str, Any]
    log_path: Path
    task_indices: Sequence[int | str]
    body: str
    error_log_path: Path | None = None
    prelude: str = ""
    comments: Sequence[str] = field(default_factory=tuple)
    array_name: str = "TASK_INDICES"
    current_var: str = "CURRENT_TASK_INDEX"
    missing_task_id_message: str = "ERROR: SLURM_ARRAY_TASK_ID not set"
    bounds_error_message: str | None = None
    signal_grace: int | None = None
    requeue: bool = False

    def render(self) -> str:
        """Render the script content."""
        if not self.task_indices:
            raise ValueError("task_indices must contain at least one entry")

        directives = format_sbatch_directives(
            job_name=self.job_name,
            slurm_args=dict(self.slurm_args),
            output_log=Path(self.log_path),
            error_log=Path(self.error_log_path or self.log_path),
        )
        array_directive = f"#SBATCH --array=0-{len(self.task_indices) - 1}"
        extra_directives = []
        if self.signal_grace:
            extra_directives.append(
                f"#SBATCH --signal=B:TERM@{self.signal_grace}"
            )
        if self.requeue:
            extra_directives.append("#SBATCH --requeue")

        directive_block = "\n".join(
            [directives, array_directive, *extra_directives]
        )
        comment_block = _line_block(self.comments)
        prelude_block = self.prelude.rstrip()
        entries = "\n".join(
            f"    {_render_task_value(entry)}" for entry in self.task_indices
        )
        bounds_message = self.bounds_error_message or (
            f"ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds "
            f"{self.array_name} size ${{#{self.array_name}[@]}}"
        )
        body = self.body.rstrip()

        return f"""#!/bin/bash
{directive_block}

{comment_block}
set -e
set -u

{prelude_block}

{self.array_name}=(
{entries}
)

if [ "${{SLURM_ARRAY_TASK_ID:-}}" = "" ]; then
    echo "{self.missing_task_id_message}"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${{#{self.array_name}[@]}}" ]; then
    echo "{bounds_message}"
    exit 1
fi

{self.current_var}="${{{self.array_name}[$SLURM_ARRAY_TASK_ID]}}"

echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID:-unknown}}"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"

set +e
{body}
EXIT_CODE=$?
set -e

echo ""
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
exit $EXIT_CODE
"""


def write_slurm_array_script(path: Path, spec: SlurmArrayScriptSpec) -> Path:
    """Write ``spec`` to ``path`` and mark it executable.

    Args:
        path: Destination script path.
        spec: Script specification to render.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.render(), encoding="utf-8")
    path.chmod(0o755)
    return path


def _render_task_value(value: int | str) -> str:
    """Render a bash array literal entry."""
    if isinstance(value, int):
        return str(value)
    return shlex.quote(value)


def _line_block(lines: Sequence[str]) -> str:
    """Return a newline-terminated block for optional lines."""
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n\n"

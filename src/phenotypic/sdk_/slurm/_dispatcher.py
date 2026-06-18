"""Drip-feed dispatcher script generation for SLURM chunk chains.

Instead of submitting all array job chunks at once (which can exceed
``MaxSubmitJobsPerUser``), this module generates lightweight dispatcher
scripts that form a chain: when chunk N finishes, its dispatcher submits
chunk N+1 and the next dispatcher.  Queue occupancy stays at ~1 chunk
(``array_limit`` jobs) + 1 dispatcher (1 job) at any time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _extract_partition(slurm_args: Dict[str, Any]) -> str:
    """Extract the partition name from SLURM args.

    Args:
        slurm_args: SLURM parameters dict with CLI-style keys.

    Returns:
        Partition name, or ``"batch"`` as fallback.
    """
    for key in ("slurm_partition", "partition"):
        if key in slurm_args:
            return str(slurm_args[key])
    return "batch"


def generate_dispatcher_script(
    next_chunk_script: Path,
    next_dispatcher_script: Optional[Path],
    output_path: Path,
    slurm_args: Dict[str, Any],
    log_dir: Path,
) -> Path:
    """Generate a dispatcher script that submits the next chunk and dispatcher.

    The dispatcher requests minimal resources (1 CPU, 100M, 5 min) and
    runs two ``sbatch`` commands: one for the next processing chunk, and
    one for the next dispatcher (with ``--dependency`` on the chunk).

    Args:
        next_chunk_script: Path to the next array job chunk script.
        next_dispatcher_script: Path to the next dispatcher script, or
            ``None`` for the last chunk (no further dispatcher needed).
        output_path: Where to write the generated dispatcher script.
        slurm_args: SLURM parameters dict (used to extract partition).
        log_dir: Directory for dispatcher log files.

    Returns:
        Path to the generated dispatcher script.
    """
    partition = _extract_partition(slurm_args)

    # Build the next-dispatcher block (empty for last chunk)
    if next_dispatcher_script is not None:
        next_dispatcher_block = (
            f'DISPATCH_JOB=$(sbatch --parsable --dependency=afterany:$CHUNK_JOB {next_dispatcher_script})\n'
            f'echo "Submitted next dispatcher: $DISPATCH_JOB (depends on $CHUNK_JOB)"'
        )
    else:
        next_dispatcher_block = 'echo "Last chunk — no further dispatcher needed"'

    script_content = f"""#!/bin/bash
#SBATCH --job-name=dispatch
#SBATCH --partition={partition}
#SBATCH --time=00:05:00
#SBATCH --mem=100M
#SBATCH --cpus-per-task=1
#SBATCH --output={log_dir}/dispatch_%j.log
#SBATCH --error={log_dir}/dispatch_%j.log

echo "Dispatcher: submitting next chunk"
echo "Timestamp: $(date)"

CHUNK_JOB=$(sbatch --parsable {next_chunk_script})
if [ $? -ne 0 ] || [ -z "$CHUNK_JOB" ]; then
    echo "ERROR: Failed to submit next chunk: {next_chunk_script}"
    exit 1
fi
echo "Submitted chunk job: $CHUNK_JOB"

# Submit next dispatcher (if not last chunk)
{next_dispatcher_block}

echo "Dispatch complete"
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_content)
    output_path.chmod(0o755)

    return output_path


def generate_dispatcher_chain(
    chunk_scripts: List[Path],
    output_dir: Path,
    slurm_args: Dict[str, Any],
    log_dir: Path,
) -> List[Path]:
    """Generate dispatcher scripts for a chain of chunk scripts.

    For *N* chunk scripts, generates *N-1* dispatcher scripts.  Each
    dispatcher submits the next chunk and (if not last) the next
    dispatcher with ``--dependency=afterany`` on that chunk.

    Args:
        chunk_scripts: Ordered list of array job chunk script paths.
        output_dir: Directory to write dispatcher scripts into.
        slurm_args: SLURM parameters dict (partition, etc.).
        log_dir: Directory for dispatcher log files.

    Returns:
        List of dispatcher script paths (one fewer than ``chunk_scripts``,
        since the last chunk does not need a dispatcher).  Empty if only
        one chunk exists.
    """
    if len(chunk_scripts) <= 1:
        return []

    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir = output_dir / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    num_dispatchers = len(chunk_scripts) - 1
    dispatcher_paths: List[Path] = []

    # Build dispatcher scripts in forward order.  Each dispatcher's path
    # is deterministic (dispatch_{idx}.sh), so we can reference the next
    # dispatcher by name without it existing on disk yet.
    for i in range(num_dispatchers):
        # Dispatcher i submits chunk_scripts[i+1]
        dispatcher_idx = i + 1  # 1-based naming: dispatch_1 submits chunk 1
        dispatcher_path = script_dir / f"dispatch_{dispatcher_idx}.sh"

        # Next dispatcher (if any)
        if i + 1 < num_dispatchers:
            next_dispatcher = script_dir / f"dispatch_{dispatcher_idx + 1}.sh"
        else:
            next_dispatcher = None

        generate_dispatcher_script(
            next_chunk_script=chunk_scripts[i + 1],
            next_dispatcher_script=next_dispatcher,
            output_path=dispatcher_path,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )
        dispatcher_paths.append(dispatcher_path)

    return dispatcher_paths


def submit_drip_feed_start(
    chunk_scripts: List[Path],
    dispatcher_scripts: List[Path],
) -> Tuple[List[str], Optional[str]]:
    """Submit the first chunk and first dispatcher to start a drip-feed chain.

    Args:
        chunk_scripts: Ordered list of chunk script paths (must be non-empty).
        dispatcher_scripts: Dispatcher scripts from
            :func:`generate_dispatcher_chain` (may be empty for single-chunk).

    Returns:
        Tuple of (job_ids, warning_message).  ``job_ids`` contains the
        submitted job IDs (1 or 2).  ``warning_message`` is ``None`` on
        success, or a string with recovery instructions if the dispatcher
        submission failed (chunk 0 was still submitted).

    Raises:
        RuntimeError: If the first chunk submission fails.
    """
    from ._sbatch import submit_script

    job_ids: List[str] = []
    warning: Optional[str] = None

    chunk0_job = submit_script(chunk_scripts[0])
    job_ids.append(chunk0_job)
    logger.info("Submitted chunk 0: Job %s", chunk0_job)

    if dispatcher_scripts:
        try:
            dispatch0_job = submit_script(
                dispatcher_scripts[0],
                dependency_job_id=chunk0_job,
            )
            job_ids.append(dispatch0_job)
            logger.info(
                "Submitted dispatcher 1: Job %s (depends on %s)",
                dispatch0_job,
                chunk0_job,
            )
        except RuntimeError as e:
            warning = (
                f"Dispatcher submission failed: {e}\n"
                f"Only chunk 0 was submitted. To resume the chain "
                f"manually, run:\n"
                f"  sbatch --dependency=afterany:{chunk0_job} "
                f"{dispatcher_scripts[0]}"
            )
            logger.warning(warning)

    return job_ids, warning

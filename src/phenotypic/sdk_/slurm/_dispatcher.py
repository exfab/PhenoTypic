"""Drip-feed dispatcher script generation for SLURM chunk chains.

Instead of submitting all array job chunks at once (which can exceed
``MaxSubmitJobsPerUser``), this module generates lightweight dispatcher
scripts that form a chain: when chunk N finishes, its dispatcher submits
chunk N+1 and the next dispatcher.  Queue occupancy stays at ~1 chunk
(``array_limit`` jobs) + 1 dispatcher (1 job) at any time.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .._io_constants import slurm_scripts_dir
from ._environment import SLURM_PYTHONPATH_BOOTSTRAP_BASH

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
    *,
    output_dir: Path | None = None,
    generation: str | None = None,
    chunk_index: int = 1,
    finalizer_script: Path | None = None,
) -> Path:
    """Generate a dispatcher script that submits the next chunk and dispatcher.

    The dispatcher requests minimal resources (1 CPU, 100M, 5 min) and
    invokes the Python lifecycle entry point. That entry point durably submits
    the next processing chunk and optional dependent dispatcher.

    Args:
        next_chunk_script: Path to the next array job chunk script.
        next_dispatcher_script: Path to the next dispatcher script, or
            ``None`` for the last chunk (no further dispatcher needed).
        output_path: Where to write the generated dispatcher script.
        slurm_args: SLURM parameters dict (used to extract partition).
        log_dir: Directory for dispatcher log files.
        output_dir: Base output directory containing the lifecycle state.
        generation: Exact scheduler generation for lifecycle submissions.
        chunk_index: Zero-based index of the chunk this dispatcher submits.
        finalizer_script: Terminal finalizer submitted after the last chunk
            becomes terminal.

    Returns:
        Path to the generated dispatcher script.
    """
    partition = _extract_partition(slurm_args)

    lifecycle_output = output_dir or _infer_output_dir(output_path)
    lifecycle_generation = generation or _ensure_generation(lifecycle_output)
    command = [
        sys.executable,
        "-m",
        "phenotypic._cli._cli_slurm_lifecycle",
        "--output",
        str(lifecycle_output),
        "--generation",
        lifecycle_generation,
        "--chunk-index",
        str(chunk_index),
        "--chunk-script",
        str(next_chunk_script),
    ]
    if next_dispatcher_script is not None:
        command.extend(["--dispatcher-script", str(next_dispatcher_script)])
    elif finalizer_script is not None:
        command.extend(["--finalizer-script", str(finalizer_script)])
    quoted_command = " ".join(shlex.quote(part) for part in command)
    final_dispatcher_message = (
        'echo "Last chunk: no further dispatcher needed"'
        if next_dispatcher_script is None
        else ""
    )

    script_content = f"""#!/bin/bash
#SBATCH --job-name=dispatch
#SBATCH --partition={partition}
#SBATCH --time=00:05:00
#SBATCH --mem=100M
#SBATCH --cpus-per-task=1
#SBATCH --output={log_dir}/dispatch_%j.log
#SBATCH --error={log_dir}/dispatch_%j.log

{SLURM_PYTHONPATH_BOOTSTRAP_BASH}

echo "Dispatcher: submitting next chunk through durable lifecycle"
echo "Timestamp: $(date)"

{quoted_command}
if [ $? -ne 0 ]; then
    echo "ERROR: Lifecycle submission failed for chunk {chunk_index}"
    exit 1
fi

{final_dispatcher_message}

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
    finalizer_script: Path | None = None,
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
        finalizer_script: Terminal finalizer passed only to the last
            dispatcher in the chain.

    Returns:
        List of dispatcher script paths (one fewer than ``chunk_scripts``,
        since the last chunk does not need a dispatcher).  Empty if only
        one chunk exists.
    """
    if len(chunk_scripts) <= 1:
        return []

    generation = _ensure_generation(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir = slurm_scripts_dir(output_dir)
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
            output_dir=output_dir,
            generation=generation,
            chunk_index=i + 1,
            finalizer_script=(
                finalizer_script if next_dispatcher is None else None
            ),
        )
        dispatcher_paths.append(dispatcher_path)

    return dispatcher_paths


def submit_drip_feed_start(
    chunk_scripts: List[Path],
    dispatcher_scripts: List[Path],
    *,
    finalizer_script: Path | None = None,
) -> Tuple[List[str], Optional[str]]:
    """Submit the first chunk and first dispatcher to start a drip-feed chain.

    Args:
        chunk_scripts: Ordered list of chunk script paths (must be non-empty).
        dispatcher_scripts: Dispatcher scripts from
            :func:`generate_dispatcher_chain` (may be empty for single-chunk).
        finalizer_script: Terminal finalizer submitted after the only chunk
            when no dispatcher is required.

    Returns:
        Tuple of (job_ids, warning_message).  ``job_ids`` contains the
        submitted job IDs (1 or 2).  ``warning_message`` is ``None`` on
        success, or a string with recovery instructions if the dispatcher
        submission failed (chunk 0 was still submitted).

    Raises:
        RuntimeError: If the first chunk submission fails.
    """
    from phenotypic._cli._cli_slurm_lifecycle import (
        cancel_generation,
        submit_with_lifecycle,
    )

    job_ids: List[str] = []
    warning: Optional[str] = None
    output_dir = _infer_output_dir(chunk_scripts[0])
    generation = _ensure_generation(output_dir)

    chunk0_job = submit_with_lifecycle(
        output_dir,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=chunk_scripts[0],
    )
    job_ids.append(chunk0_job)
    logger.info("Submitted chunk 0: Job %s", chunk0_job)

    if dispatcher_scripts:
        try:
            dispatch0_job = submit_with_lifecycle(
                output_dir,
                generation=generation,
                token="dispatcher-1",
                role="dispatcher",
                script_path=dispatcher_scripts[0],
                dependencies=(chunk0_job,),
            )
            job_ids.append(dispatch0_job)
            logger.info(
                "Submitted dispatcher 1: Job %s (depends on %s)",
                dispatch0_job,
                chunk0_job,
            )
        except RuntimeError as exc:
            cancellation = cancel_generation(output_dir, generation)
            detail = (
                "the launch was fenced and all discovered jobs were cancelled"
                if cancellation.quiescent
                else "the launch was fenced but scheduler reconciliation remains "
                "incomplete"
            )
            raise RuntimeError(
                f"Initial dispatcher submission failed; {detail}: {exc}"
            ) from exc
    elif finalizer_script is not None:
        try:
            finalizer_job = submit_with_lifecycle(
                output_dir,
                generation=generation,
                token="finalizer",
                role="finalizer",
                script_path=finalizer_script,
                dependencies=(chunk0_job,),
            )
            job_ids.append(finalizer_job)
            logger.info(
                "Submitted terminal finalizer: Job %s (depends on %s)",
                finalizer_job,
                chunk0_job,
            )
        except RuntimeError as exc:
            cancellation = cancel_generation(output_dir, generation)
            detail = (
                "the launch was fenced and all discovered jobs were cancelled"
                if cancellation.quiescent
                else "the launch was fenced but scheduler reconciliation remains "
                "incomplete"
            )
            raise RuntimeError(
                f"Terminal finalizer submission failed; {detail}: {exc}"
            ) from exc

    return job_ids, warning


def _infer_output_dir(script_path: Path) -> Path:
    """Infer the run output root from a generated script path."""
    path = Path(script_path).resolve()
    for ancestor in path.parents:
        if (
            ancestor.name == "slurm_scripts"
            and ancestor.parent.name == ".phenotypic"
        ):
            return ancestor.parent.parent
    return path.parent


def _ensure_generation(output_dir: Path) -> str:
    """Return the active generation, creating one for non-CLI SDK callers."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
        new_slurm_generation,
    )

    state = load_slurm_lifecycle(output_dir)
    if state is not None:
        generation = str(state["generation"])
        if generation_is_active(output_dir, generation):
            return generation
    generation = new_slurm_generation()
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="ordinary"
    )
    return generation

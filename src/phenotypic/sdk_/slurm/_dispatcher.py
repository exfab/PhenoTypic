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
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

from .._io_constants import slurm_scripts_dir
from ._environment import SLURM_PYTHONPATH_BOOTSTRAP_BASH
from ._generation import generation_script_key

logger = logging.getLogger(__name__)

SlurmDependencyKind = Literal["afterany", "afterok"]

# Importing the lifecycle CLI loads the package and its scheduler contracts.
# 100M is below that process's observed Linux RSS (~99 MiB before allocator
# and interpreter headroom) and was OOM-killed on the first real migration
# dispatcher. Keep this much smaller than an image worker while leaving safe
# room for the control-plane Python process.
_DISPATCHER_MEMORY = "512M"


def _validate_dependency_kind(value: str) -> SlurmDependencyKind:
    """Validate one continuation dependency before any side effects."""
    if value not in {"afterany", "afterok"}:
        raise ValueError("dependency_kind must be 'afterany' or 'afterok'")
    return cast(SlurmDependencyKind, value)


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
    dependency_kind: SlurmDependencyKind = "afterany",
) -> Path:
    """Generate a dispatcher script that submits the next chunk and dispatcher.

    The dispatcher requests control-plane resources (1 CPU, 512M, 5 min) and
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
        dependency_kind: Dependency condition for the continuation submitted
            after ``next_chunk_script``.

    Returns:
        Path to the generated dispatcher script.
    """
    validated_dependency_kind = _validate_dependency_kind(dependency_kind)
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
        "--dependency-kind",
        validated_dependency_kind,
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
#SBATCH --mem={_DISPATCHER_MEMORY}
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
    continuation_dependency_kinds: Sequence[SlurmDependencyKind] | None = None,
    generation: str | None = None,
    lifecycle_output_dir: Path | None = None,
) -> List[Path]:
    """Generate dispatcher scripts for a chain of chunk scripts.

    For *N* chunk scripts, generates *N-1* dispatcher scripts.  Each
    dispatcher submits the next chunk and (if not last) the next
    dispatcher with the corresponding dependency kind on that chunk.

    Args:
        chunk_scripts: Ordered list of array job chunk script paths.
        output_dir: Directory to write dispatcher scripts into.
        slurm_args: SLURM parameters dict (partition, etc.).
        log_dir: Directory for dispatcher log files.
        finalizer_script: Terminal finalizer passed only to the last
            dispatcher in the chain.
        continuation_dependency_kinds: Dependency kind for every
            chunk-to-continuation edge. The sequence has ``N - 1`` entries
            without a finalizer and ``N`` entries with one. Defaults to
            ``afterany`` for every edge.
        generation: Optional explicit lifecycle generation. When supplied,
            dispatcher scripts are isolated under that generation.
        lifecycle_output_dir: Optional lifecycle fence root when dispatcher
            scripts themselves are written beneath a separate control root.

    Returns:
        List of dispatcher script paths (one fewer than ``chunk_scripts``,
        since the last chunk does not need a dispatcher).  Empty if only
        one chunk exists.
    """
    dependency_kinds = _resolve_continuation_dependency_kinds(
        chunk_count=len(chunk_scripts),
        has_finalizer=finalizer_script is not None,
        dependency_kinds=continuation_dependency_kinds,
    )
    if len(chunk_scripts) <= 1:
        return []

    lifecycle_output = lifecycle_output_dir or output_dir
    lifecycle_generation = generation or _ensure_generation(lifecycle_output)
    log_dir.mkdir(parents=True, exist_ok=True)
    script_dir = slurm_scripts_dir(output_dir)
    if generation is not None:
        script_dir = script_dir / "dispatch" / generation_script_key(generation)
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
            output_dir=lifecycle_output,
            generation=lifecycle_generation,
            chunk_index=i + 1,
            finalizer_script=(
                finalizer_script if next_dispatcher is None else None
            ),
            dependency_kind=(
                dependency_kinds[i + 1]
                if i + 1 < len(dependency_kinds)
                else "afterany"
            ),
        )
        dispatcher_paths.append(dispatcher_path)

    return dispatcher_paths


def submit_drip_feed_start(
    chunk_scripts: List[Path],
    dispatcher_scripts: List[Path],
    *,
    finalizer_script: Path | None = None,
    continuation_dependency_kind: SlurmDependencyKind = "afterany",
    output_dir: Path | None = None,
    generation: str | None = None,
) -> Tuple[List[str], Optional[str]]:
    """Submit the first chunk and first dispatcher to start a drip-feed chain.

    Args:
        chunk_scripts: Ordered list of chunk script paths (must be non-empty).
        dispatcher_scripts: Dispatcher scripts from
            :func:`generate_dispatcher_chain` (may be empty for single-chunk).
        finalizer_script: Terminal finalizer submitted after the only chunk
            when no dispatcher is required.
        continuation_dependency_kind: Dependency condition for the initial
            dispatcher or single-chunk finalizer.
        output_dir: Explicit lifecycle output root for attempt-scoped chains.
        generation: Explicit lifecycle generation for attempt-scoped chains.

    Returns:
        Tuple of (job_ids, warning_message).  ``job_ids`` contains the
        submitted job IDs (1 or 2).  ``warning_message`` is ``None`` on
        success, or a string with recovery instructions if the dispatcher
        submission failed (chunk 0 was still submitted).

    Raises:
        RuntimeError: If the first chunk submission fails.
    """
    validated_dependency_kind = _validate_dependency_kind(
        continuation_dependency_kind
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        cancel_generation,
        submit_with_lifecycle,
    )

    job_ids: List[str] = []
    warning: Optional[str] = None
    lifecycle_output = output_dir or _infer_output_dir(chunk_scripts[0])
    lifecycle_generation = generation or _ensure_generation(lifecycle_output)

    chunk0_job = submit_with_lifecycle(
        lifecycle_output,
        generation=lifecycle_generation,
        token="chunk-0",
        role="chunk",
        script_path=chunk_scripts[0],
    )
    job_ids.append(chunk0_job)
    logger.info("Submitted chunk 0: Job %s", chunk0_job)

    if dispatcher_scripts:
        try:
            dispatch0_job = submit_with_lifecycle(
                lifecycle_output,
                generation=lifecycle_generation,
                token="dispatcher-1",
                role="dispatcher",
                script_path=dispatcher_scripts[0],
                dependencies=(chunk0_job,),
                dependency_kind=validated_dependency_kind,
            )
            job_ids.append(dispatch0_job)
            logger.info(
                "Submitted dispatcher 1: Job %s (depends on %s)",
                dispatch0_job,
                chunk0_job,
            )
        except RuntimeError as exc:
            cancellation = cancel_generation(
                lifecycle_output, lifecycle_generation
            )
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
                lifecycle_output,
                generation=lifecycle_generation,
                token="finalizer",
                role="finalizer",
                script_path=finalizer_script,
                dependencies=(chunk0_job,),
                dependency_kind=validated_dependency_kind,
            )
            job_ids.append(finalizer_job)
            logger.info(
                "Submitted terminal finalizer: Job %s (depends on %s)",
                finalizer_job,
                chunk0_job,
            )
        except RuntimeError as exc:
            cancellation = cancel_generation(
                lifecycle_output, lifecycle_generation
            )
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


def _resolve_continuation_dependency_kinds(
    *,
    chunk_count: int,
    has_finalizer: bool,
    dependency_kinds: Sequence[SlurmDependencyKind] | None,
) -> tuple[SlurmDependencyKind, ...]:
    """Return one validated dependency kind per continuation edge."""
    edge_count = max(0, chunk_count - 1)
    if has_finalizer and chunk_count:
        edge_count += 1
    if dependency_kinds is None:
        return cast(
            tuple[SlurmDependencyKind, ...], ("afterany",) * edge_count
        )
    resolved = tuple(dependency_kinds)
    if len(resolved) != edge_count:
        raise ValueError(
            "continuation_dependency_kinds must contain exactly "
            f"{edge_count} entries for {chunk_count} chunk script(s)"
        )
    return tuple(_validate_dependency_kind(kind) for kind in resolved)


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

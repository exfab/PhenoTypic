"""Shared SLURM script chain submission helpers for CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from phenotypic.sdk_ import logs_dir
from phenotypic.sdk_.slurm import (
    generate_dispatcher_chain,
    submit_drip_feed_start,
)
from phenotypic.sdk_.slurm._dispatcher import (
    SlurmDependencyKind,
    _resolve_continuation_dependency_kinds,
)


@dataclass(frozen=True)
class SLURMScriptChainSubmission:
    """Result from starting a drip-feed SLURM script chain.

    Args:
        job_ids: Initial SLURM job IDs returned by submission.
        warning: Recovery warning if dispatcher submission failed.
        flat_scripts: Ordered chunk scripts used for dispatch/submission.
        dispatcher_scripts: Ordered dispatcher scripts generated for chunks.
    """

    job_ids: List[str]
    warning: Optional[str]
    flat_scripts: List[Path]
    dispatcher_scripts: List[Path]


def submit_slurm_script_chain(
    *,
    flat_chunk_scripts: Sequence[Path],
    output_dir: Path,
    slurm_args: Dict[str, Any],
    console: Any,
    finalizer_script: Path | None = None,
    continuation_dependency_kinds: Sequence[SlurmDependencyKind] | None = None,
    generation: str | None = None,
) -> SLURMScriptChainSubmission:
    """Generate and start a drip-feed SLURM dispatcher chain.

    Args:
        flat_chunk_scripts: Ordered array job scripts to submit.
        output_dir: Base CLI output directory.
        slurm_args: SLURM parameters used for dispatcher scripts.
        console: Rich console-like object for status output.
        finalizer_script: Terminal publisher submitted after the final chunk
            becomes terminal.
        continuation_dependency_kinds: Dependency kind for each
            chunk-to-continuation edge. Defaults to ``afterany`` throughout.
        generation: Optional explicit lifecycle generation for this chain.

    Returns:
        Submission result with job IDs and generated script paths.

    Raises:
        RuntimeError: If no chunk scripts were provided or the initial chunk
            submission fails.
    """
    flat_scripts = list(flat_chunk_scripts)
    if not flat_scripts:
        raise RuntimeError(
            "No array job scripts were generated. "
            "Check that datasets contain images."
        )

    log_dir = logs_dir(output_dir) / "slurm"
    dependency_kinds = _resolve_continuation_dependency_kinds(
        chunk_count=len(flat_scripts),
        has_finalizer=finalizer_script is not None,
        dependency_kinds=continuation_dependency_kinds,
    )
    initial_dependency_kind: SlurmDependencyKind = (
        dependency_kinds[0] if dependency_kinds else "afterany"
    )
    generation_kwargs: dict[str, Any] = (
        {"generation": generation} if generation is not None else {}
    )
    lifecycle_kwargs: dict[str, Any] = (
        {"output_dir": output_dir, "generation": generation}
        if generation is not None
        else {}
    )
    if finalizer_script is None:
        dispatcher_scripts = generate_dispatcher_chain(
            chunk_scripts=flat_scripts,
            output_dir=output_dir,
            slurm_args=slurm_args,
            log_dir=log_dir,
            continuation_dependency_kinds=dependency_kinds,
            **generation_kwargs,
        )
    else:
        dispatcher_scripts = generate_dispatcher_chain(
            chunk_scripts=flat_scripts,
            output_dir=output_dir,
            slurm_args=slurm_args,
            log_dir=log_dir,
            finalizer_script=finalizer_script,
            continuation_dependency_kinds=dependency_kinds,
            **generation_kwargs,
        )

    console.print("[bold cyan]Submitting jobs to SLURM...[/bold cyan]")

    if finalizer_script is None:
        job_ids, warning = submit_drip_feed_start(
            chunk_scripts=flat_scripts,
            dispatcher_scripts=dispatcher_scripts,
            continuation_dependency_kind=initial_dependency_kind,
            **lifecycle_kwargs,
        )
    else:
        job_ids, warning = submit_drip_feed_start(
            chunk_scripts=flat_scripts,
            dispatcher_scripts=dispatcher_scripts,
            finalizer_script=finalizer_script,
            continuation_dependency_kind=initial_dependency_kind,
            **lifecycle_kwargs,
        )

    console.print(f"  Chunk 0: [green]Job {job_ids[0]}[/green]")
    if dispatcher_scripts and len(job_ids) > 1:
        console.print(
            f"  Dispatcher 1: [green]Job {job_ids[1]}[/green] "
            f"(depends on {job_ids[0]})"
        )
        console.print(
            f"  Remaining {len(flat_scripts) - 1} chunk(s) will be "
            f"auto-submitted as each completes"
        )
    elif finalizer_script is not None and len(job_ids) > 1:
        console.print(
            f"  Finalizer: [green]Job {job_ids[1]}[/green] "
            f"(depends on {job_ids[0]})"
        )
    if warning:
        console.print(f"  [yellow]Warning: {warning}[/yellow]")

    console.print(
        f"[green]Submitted {len(job_ids)} initial job(s) "
        f"(drip-feed dispatcher)[/green]\n"
    )

    return SLURMScriptChainSubmission(
        job_ids=job_ids,
        warning=warning,
        flat_scripts=flat_scripts,
        dispatcher_scripts=dispatcher_scripts,
    )

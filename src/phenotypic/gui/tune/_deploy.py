"""Deploy helper for launching Tune runs through the run-console runner."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

from phenotypic.gui.shell._runs_registry import RunStatus

if TYPE_CHECKING:
    from phenotypic.gui.shell._runs_registry import RunRegistry
    from phenotypic.gui.shell._sandbox import SandboxRoot


def _relative_run_path(output_dir: PurePath, root: PurePath) -> str:
    """Return the sandbox-relative run path using URL-style separators."""
    return output_dir.relative_to(root).as_posix()


def _command_digest(argv: list[str]) -> str:
    """Return a stable digest for one Tune launch command."""
    return hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest()


def deploy_tune_run(
    *,
    runner: Any,
    registry: "RunRegistry",
    sandbox: "SandboxRoot",
    argv: list[str],
    output_dir: Path,
    slurm: bool,
) -> str:
    """Register and launch a Tune run via ``LocalRunner.start``.

    SLURM Tune launches still go through the local runner: the spawned tune CLI
    owns ``--slurm`` submission. The GUI records mode/status only and does not
    parse or persist a job id in v1.
    """
    output_dir = sandbox.resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_path = _relative_run_path(output_dir, sandbox.root)
    run_id = rel_path
    initial_status: RunStatus = "submitting" if slurm else "queued"
    record = registry.allocate(
        run_id=run_id,
        mode="slurm" if slurm else "local",
        output_dir=output_dir,
        rel_path=rel_path,
        command_digest=_command_digest(argv),
        status=initial_status,
    )
    generation = record.generation
    if generation is None:  # pragma: no cover
        raise RuntimeError("allocated Tune run has no generation")
    try:
        handle = runner.start(
            run_id,
            argv,
            output_dir=output_dir,
            generation=generation,
        )
    except Exception:
        registry.compare_and_set(
            run_id,
            generation,
            expected_statuses={initial_status},
            status="failed",
        )
        raise
    pid = getattr(getattr(handle, "process", None), "pid", None)
    log_path = getattr(handle, "stdout_log_path", None)
    registry.compare_and_set(
        run_id,
        generation,
        expected_statuses={initial_status},
        status="submitting" if slurm else "running",
        pid=pid if isinstance(pid, int) else None,
        log_paths=(log_path,) if isinstance(log_path, Path) else (),
    )
    return run_id


__all__ = ["deploy_tune_run"]

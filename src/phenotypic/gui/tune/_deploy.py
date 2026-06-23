"""Deploy helper for launching Tune runs through the run-console runner."""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

from phenotypic.gui.shell._runs_registry import RunRecord

if TYPE_CHECKING:
    from phenotypic.gui.shell._runs_registry import RunRegistry
    from phenotypic.gui.shell._sandbox import SandboxRoot


def _relative_run_path(output_dir: PurePath, root: PurePath) -> str:
    """Return the sandbox-relative run path using URL-style separators."""
    return output_dir.relative_to(root).as_posix()


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
    if hasattr(runner, "reap"):
        runner.reap(run_id)
    handle = runner.start(run_id, argv, output_dir=output_dir)
    pid = getattr(getattr(handle, "process", None), "pid", None)
    log_path = getattr(handle, "stdout_log_path", None)
    registry.register(
        RunRecord(
            run_id=run_id,
            mode="slurm" if slurm else "local",
            output_dir=output_dir,
            rel_path=rel_path,
            status="submitting" if slurm else "running",
            pid=pid if isinstance(pid, int) else None,
            log_path=log_path if isinstance(log_path, Path) else None,
        )
    )
    return run_id


__all__ = ["deploy_tune_run"]

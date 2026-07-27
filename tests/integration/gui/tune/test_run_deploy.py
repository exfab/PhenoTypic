from pathlib import Path, PureWindowsPath

import pytest

from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot

from phenotypic.gui.tune._deploy import _relative_run_path, deploy_tune_run


class _Process:
    pid = 1234


class _Handle:
    process = _Process()
    stdout_log_path = Path("/tmp/stdout.log")


class _Runner:
    def __init__(self):
        self.started = []

    def start(self, run_id, argv, *, output_dir, generation):
        self.started.append((run_id, argv, output_dir, generation))
        return _Handle()


def test_deploy_local_registers_run_and_starts_runner(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = _Runner()
    output_dir = tmp_path / "tune-runs" / "run1"

    receipt = deploy_tune_run(
        runner=runner,
        registry=registry,
        sandbox=sandbox,
        argv=["python", "-m", "phenotypic.tune", "run", "spec"],
        output_dir=output_dir,
        slurm=False,
    )
    run_id = receipt["run_id"]

    assert run_id == "tune-runs/run1"
    assert runner.started[0][0] == run_id
    record = registry.get(run_id)
    assert record is not None
    assert record.mode == "local"
    assert record.status == "running"
    assert record.pid == 1234
    assert runner.started[0][3] == record.generation
    assert receipt["generation"] == str(record.generation)


def test_deploy_resolves_relative_output_inside_sandbox(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = _Runner()

    receipt = deploy_tune_run(
        runner=runner,
        registry=registry,
        sandbox=sandbox,
        argv=["python"],
        output_dir=Path("relative-out"),
        slurm=False,
    )
    run_id = receipt["run_id"]

    assert run_id == "relative-out"
    assert runner.started[0][2] == tmp_path / "relative-out"


def test_relative_run_path_uses_posix_separators_for_windows_paths():
    root = PureWindowsPath("C:/sandbox")
    output_dir = root / "tune-runs" / "run1"

    assert _relative_run_path(output_dir, root) == "tune-runs/run1"


def test_deploy_slurm_uses_runner_without_job_id(tmp_path: Path):
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = _Runner()

    receipt = deploy_tune_run(
        runner=runner,
        registry=registry,
        sandbox=sandbox,
        argv=["python", "-m", "phenotypic.tune", "run", "spec", "--slurm"],
        output_dir=tmp_path / "out",
        slurm=True,
    )
    run_id = receipt["run_id"]

    record = registry.get(run_id)
    assert record is not None
    assert record.mode == "slurm"
    assert record.status == "submitting"
    assert record.slurm_job_id is None
    assert runner.started[0][1][-1] == "--slurm"


def test_deploy_rejects_output_escape(tmp_path: Path):
    root = tmp_path / "sandbox"
    root.mkdir()
    sandbox = SandboxRoot.from_path(root)

    with pytest.raises(ValueError, match="escapes sandbox"):
        deploy_tune_run(
            runner=_Runner(),
            registry=RunRegistry(),
            sandbox=sandbox,
            argv=["python"],
            output_dir=tmp_path / "outside",
            slurm=False,
        )

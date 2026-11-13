"""DVC outs on the SLURM path (#281 spec C1, C13, C14; P2 Task 5).

A SLURM-backed DVC stage hashes its outs when ``phenotypic-wait`` exits, after
the finalizer. Workers and the finalizer only add to what the submission
created, so the outs must already exist, and be the only root entries, at
submission. These tests need the ``submissions`` scheduler stub, which lives
in ``tests/unit/cli/conftest.py``; the local-run guards are in
``tests/integration/cli/test_dvc_layout_guards.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli import _cli_staged_orchestration
from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    load_slurm_lifecycle,
)
from phenotypic.sdk_ import RunExitCode
from tests._tiny_run import run_cli

DVC_OUTS = frozenset({"deliverables", "results", ".phenotypic"})


def test_a_slurm_submission_creates_exactly_the_three_outs(
    tiny_inputs: Path, tmp_path: Path, submissions: list[str]
) -> None:
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, slurm=True)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["ordinary"]
    assert {p.name for p in out.iterdir()} == DVC_OUTS
    assert all((out / name).is_dir() for name in DVC_OUTS)


def test_a_restored_active_fence_does_not_block_a_slurm_resubmission(
    tiny_inputs: Path,
    tmp_path: Path,
    submissions: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec C13: a ``dvc checkout`` may restore a fence a dead finalizer left active."""
    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, slurm=True).exit_code == RunExitCode.SUBMITTED
    restored = load_slurm_lifecycle(out)
    assert restored is not None and restored["active"] is True
    append_lifecycle_entry(
        out, generation=str(restored["generation"]), token="finalizer",
        role="finalizer", status="submitted", job_id="77",
    )
    monkeypatch.setattr(
        _cli_staged_orchestration, "scheduler_job_is_active", lambda job_id: False
    )
    result = run_cli(tiny_inputs, out, slurm=True)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["ordinary", "ordinary"]
    current = load_slurm_lifecycle(out)
    assert current is not None and current["active"] is True
    assert current["generation"] != restored["generation"]
    assert "closed_by" not in current


def test_a_fence_that_may_still_own_a_job_is_kept(
    tiny_inputs: Path,
    tmp_path: Path,
    submissions: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retire acts on the guard's evidence only: a live job refuses, as before."""
    from phenotypic._cli._cli_staged_orchestration import retire_dead_lifecycle_fence

    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, slurm=True).exit_code == RunExitCode.SUBMITTED
    restored = load_slurm_lifecycle(out)
    append_lifecycle_entry(
        out, generation=str(restored["generation"]), token="finalizer",
        role="finalizer", status="submitted", job_id="77",
    )
    for answer in (True, None):  # live, then unverified
        monkeypatch.setattr(
            _cli_staged_orchestration, "scheduler_job_is_active",
            lambda job_id, answer=answer: answer,
        )
        assert retire_dead_lifecycle_fence(out) is None
        result = run_cli(tiny_inputs, out, slurm=True)
        assert result.exit_code == 1, result.output
        assert "may still own it" in result.output
        assert load_slurm_lifecycle(out)["generation"] == restored["generation"]
        assert load_slurm_lifecycle(out)["active"] is True
    assert submissions == ["ordinary"]

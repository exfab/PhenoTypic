"""DVC outs on the SLURM path (#281 spec C1, C13, C14; P2 Task 5).

A SLURM-backed DVC stage hashes its outs when ``phenotypic-wait`` exits, after
the finalizer. Workers and the finalizer only add to what the submission
created, so the outs must already exist, and be the only root entries, at
submission. These tests need the ``submissions`` scheduler stub, which lives
in ``tests/unit/cli/conftest.py``; the local-run guards are in
``tests/integration/cli/test_dvc_layout_guards.py``.
"""

from __future__ import annotations

import json
import sys
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


# ---- #281 P2 review I4 and the "Test coverage gaps" table ---------------------


def _dead_job(out: Path, generation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ledger one finished job for *generation*, so the guard's evidence is "dead"."""
    append_lifecycle_entry(
        out, generation=generation, token="finalizer",
        role="finalizer", status="submitted", job_id="77",
    )  # fmt: skip
    monkeypatch.setattr(
        _cli_staged_orchestration, "scheduler_job_is_active", lambda job_id: False
    )


def test_an_active_fence_with_no_ledger_rows_is_not_retired(
    tiny_inputs: Path, tmp_path: Path, submissions: list[str]
) -> None:
    """Review I4: a fence with no rows may be a submission still being prepared.

    The stubbed submitter writes no ledger row, which is exactly the state of a
    submitter between publishing its fence and writing its first ``intent``.
    A second invocation must refuse, naming the fence, rather than close it
    under the first.
    """
    from phenotypic._cli._cli_slurm_lifecycle import lifecycle_state_path
    from phenotypic._cli._cli_staged_orchestration import (
        read_job_ledger,
        retire_dead_lifecycle_fence,
    )

    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, slurm=True).exit_code == RunExitCode.SUBMITTED
    fence = load_slurm_lifecycle(out)
    assert fence is not None and fence["active"] is True
    generation = str(fence["generation"])
    assert read_job_ledger(out, epoch=generation) == []  # the premise
    assert retire_dead_lifecycle_fence(out) is None
    result = run_cli(tiny_inputs, out, slurm=True)
    assert result.exit_code == RunExitCode.ERROR, result.output
    assert generation in result.output
    assert lifecycle_state_path(out).name in result.output
    assert submissions == ["ordinary"]
    after = load_slurm_lifecycle(out)
    assert after["generation"] == generation and after["active"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="SLURM submission is POSIX-only")
def test_a_slurm_recompile_retires_a_dead_fence(
    finished_tiny_run: Path, submissions: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage gap: the retire inside ``_initialize_recompile_slurm_attempt``."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    out = finished_tiny_run
    initialize_slurm_lifecycle(out, generation="g-old", mode="ordinary")
    _dead_job(out, "g-old", monkeypatch)
    result = run_cli(None, out, mode="recompile", slurm=True)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["recompile"]
    assert "closed SLURM generation g-old" in result.output
    current = load_slurm_lifecycle(out)
    assert current["generation"] != "g-old" and current["active"] is True


def test_a_dead_staged_fence_is_retired_through_the_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage gap: a ``staged-*`` fence, closed and journalled for stragglers."""
    from phenotypic._cli._cli_staged_orchestration import (
        epoch_deactivation_journal_path,
        initialize_orchestration,
        load_orchestration_state,
        retire_dead_lifecycle_fence,
    )

    out = tmp_path / "out"
    initialize_orchestration(
        out, epoch="e-old", mode="forward", controller_config_path=out / "c.json"
    )
    assert load_slurm_lifecycle(out)["mode"] == "staged-forward"
    _dead_job(out, "e-old", monkeypatch)
    assert retire_dead_lifecycle_fence(out) == "e-old"
    fence = load_slurm_lifecycle(out)
    assert fence["active"] is False and "closed_by" not in fence
    journal = epoch_deactivation_journal_path(out).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in journal.splitlines() if line.strip()]
    assert [(row["epoch"], row["phase"]) for row in rows] == [("e-old", "cancelled")]
    assert load_orchestration_state(out)["phase"] == "cancelled"


@pytest.mark.parametrize(
    "fence",
    [
        {"mode": "ordinary", "owner_kind": "local"},
        {"mode": "migrate"},
        {"mode": "tune"},
    ],
    ids=["owned", "migrate", "tune"],
)
def test_fences_this_run_does_not_own_are_never_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fence: dict[str, str]
) -> None:
    """Coverage gap: the negative branches, each over a fence whose jobs are dead."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic._cli._cli_staged_orchestration import retire_dead_lifecycle_fence

    out = tmp_path / "out"
    out.mkdir()
    owner = fence.get("owner_kind")
    initialize_slurm_lifecycle(
        out, generation="g-other", mode=fence["mode"], owner_kind=owner,
        control_root=out if owner else None,
    )  # fmt: skip
    _dead_job(out, "g-other", monkeypatch)
    assert not _cli_staged_orchestration.live_slurm_job_ids(out)  # dead, as premised
    assert retire_dead_lifecycle_fence(out) is None
    assert load_slurm_lifecycle(out)["active"] is True


def test_a_passed_is_active_is_the_one_consulted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``phenotypic-wait`` hands in its memo; the module default is then never asked."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    out = tmp_path / "out"
    out.mkdir()
    initialize_slurm_lifecycle(out, generation="g-live", mode="ordinary")
    _dead_job(out, "g-live", monkeypatch)  # the default would answer "dead"
    asked: list[str] = []

    def _memo(job_id: str) -> bool:
        asked.append(job_id)
        return True

    jobs = _cli_staged_orchestration.live_slurm_job_ids(out, is_active=_memo)
    assert jobs.live == ("77",) and asked == ["77"]
    assert not _cli_staged_orchestration.live_slurm_job_ids(out)  # default unchanged

"""The staged SLURM controller v2: Stage-4 rounds, their counters and the worklist (#282 P5 T1).

Spec §4.3 and U7. The scheduler is faked at the controller's seams
(``fake_scheduler``); the images are real artifacts on disk
(``finish_stage3``), so ``stage4_status`` classifies them as a shard would
leave them. No torch, except the one CLI continuation test at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_failure_tracker import read_terminal_failures
from phenotypic._cli._cli_staged_controller import (
    _classify_stage4,
    _stage4_progress_digest,
    run_staged_controller,
)
from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    append_job_ledger,
    deactivate_orchestration,
    live_slurm_job_ids,
    load_orchestration_state,
    load_stage4_worklist,
    read_job_ledger,
    save_orchestration_state,
    write_stage4_worklist,
)
from phenotypic._cli import _cli_staged_orchestration as orchestration_module
from phenotypic._cli._cli_wait import RunExitCode
from tests._tiny_run import no_stage4_signals
from tests.unit.cli.conftest import (
    finish_stage3,
    no_live_jobs,
    plant_signals_for_done_image,
    record_of,
    run_round,
    write_signals,
)


def entry(stem: str) -> StagedManifestEntry:
    return StagedManifestEntry("plate", f"{stem}.tif", stem, f"/in/plate/{stem}.tif", work_id=f"work-{stem}")


_WORKLISTS: list[Path] = []


@pytest.fixture(autouse=True)
def _worklist_dir(tmp_path: Path):
    _WORKLISTS.clear()
    _WORKLISTS.append(tmp_path / "worklists")
    yield


def tmp_worklist() -> Path:
    directory = _WORKLISTS[0]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"worklist-{len(list(directory.iterdir()))}.json"


def _roles(fake_scheduler) -> list[str]:
    return [role for role, _ in fake_scheduler.submitted_roles()]


# ---------------------------------------------------------------------------
# After the last Stage-3 chunk
# ---------------------------------------------------------------------------


def test_after_the_last_stage3_chunk_stage4_is_classified(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=3)
    run_staged_controller(controller_v2.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")
    shard = load_stage4_worklist(controller_v2.worklist, 0)
    assert shard.round_index == 1 and len(shard.entries) == 3
    assert shard.manifest_path == controller_v2.manifest
    state = load_orchestration_state(controller_v2.out)
    assert state["phase"] == "stage4" and state["stage4_round"] == 1
    assert state["active_job_id"] == fake_scheduler.job_of("stage4-round-1")


def test_all_done_goes_straight_to_the_finalizer(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=0, done=2)
    run_staged_controller(controller_v2.config_path)
    roles = _roles(fake_scheduler)
    assert roles[-1] == "finalizer" and "stage4" not in roles
    assert not controller_v2.worklist.exists()


def test_the_round_retargets_the_recovery_controller(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1)
    run_staged_controller(controller_v2.config_path)
    successor = fake_scheduler.job_of("controller-after-1000")
    assert (successor, ["1000", fake_scheduler.job_of("stage4-round-1")]) in fake_scheduler.dependencies


# ---------------------------------------------------------------------------
# Counters, progress and the stall (U7)
# ---------------------------------------------------------------------------


def test_rounds_have_their_own_counters(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    state = load_orchestration_state(controller_v2.out)
    state.update(zero_progress_rounds=1, last_retryable_digest="x", round=4)  # Stage 2's
    save_orchestration_state(controller_v2.out, state)
    run_staged_controller(controller_v2.config_path)
    state = load_orchestration_state(controller_v2.out)
    assert state["stage4_zero_progress_rounds"] == 0 and state["stage4_round"] == 1
    assert state["zero_progress_rounds"] == 1 and state["last_retryable_digest"] == "x"
    assert state["round"] == 4
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")


def test_a_stage2_stall_does_not_count_against_stage4(controller_v2, fake_scheduler):
    # Stage 2 ended on two unchanged rounds; Stage 4 still gets its two.
    entries = finish_stage3(controller_v2, stage4_pending=2)
    state = load_orchestration_state(controller_v2.out)
    # Review M6: Stage 2's digest is the very digest Stage 4's first
    # transition computes, so shared counters would read "unchanged" and
    # stall at once; only Stage-4-owned counters give two rounds.
    digest, _count = _stage4_progress_digest(controller_v2.config, entries)
    state.update(zero_progress_rounds=2, last_retryable_digest=digest)
    save_orchestration_state(controller_v2.out, state)
    run_round(controller_v2, fake_scheduler)
    run_round(controller_v2, fake_scheduler)
    assert _roles(fake_scheduler).count("stage4") == 2


def test_the_progress_digest_reads_no_parquet(controller_v2, monkeypatch):
    finish_stage3(controller_v2, stage4_pending=2)
    write_signals(controller_v2, regions={"objects"})
    monkeypatch.setattr("pyarrow.parquet.read_metadata", lambda *a, **k: pytest.fail("footer read"))
    monkeypatch.setattr("pyarrow.parquet.read_table", lambda *a, **k: pytest.fail("table read"))
    monkeypatch.setattr("pyarrow.parquet.ParquetFile", lambda *a, **k: pytest.fail("file opened"))
    _stage4_progress_digest(controller_v2.config, controller_v2.entries)


def test_the_progress_digest_follows_each_key_signal(controller_v2):
    entries = finish_stage3(controller_v2, stage4_pending=2)
    base, count = _stage4_progress_digest(controller_v2.config, entries)
    assert count == 2
    assert _stage4_progress_digest(controller_v2.config, list(reversed(entries)))[0] == base
    write_signals(controller_v2, regions={"objects"}, entries=entries[:1], mtime_ns=10**18)
    one = _stage4_progress_digest(controller_v2.config, entries)[0]
    assert one != base                                                     # a written signal
    assert _stage4_progress_digest(controller_v2.config, entries)[0] == one  # nothing new
    write_signals(controller_v2, regions={"objects"}, entries=entries[:1], mtime_ns=2 * 10**18)
    rewritten = _stage4_progress_digest(controller_v2.config, entries)[0]
    assert rewritten not in {base, one}                                    # a rewritten one
    write_signals(controller_v2, regions={"grid"}, entries=entries[:1], mtime_ns=2 * 10**18)
    assert _stage4_progress_digest(controller_v2.config, entries)[0] != rewritten  # another key
    assert _stage4_progress_digest(controller_v2.config, entries[1:])[0] != rewritten  # fewer images


def test_written_signals_count_as_progress(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    run_round(controller_v2, fake_scheduler)                                  # round 1 starts
    run_round(controller_v2, fake_scheduler, write_signals_for={"objects"})   # one backbone done, nothing committed
    state = load_orchestration_state(controller_v2.out)
    assert state["stage4_zero_progress_rounds"] == 0 and state["stage4_round"] == 2
    run_round(controller_v2, fake_scheduler)                                  # round 2 writes nothing
    state = load_orchestration_state(controller_v2.out)
    assert state["stage4_zero_progress_rounds"] == 1 and state["stage4_round"] == 3


def test_a_committed_image_counts_as_progress(controller_v2, fake_scheduler):
    from tests.unit.cli.conftest import _stage4_image

    entries = finish_stage3(controller_v2, stage4_pending=2)
    run_round(controller_v2, fake_scheduler)
    _stage4_image(controller_v2.out, entries[0].stem, done=True)            # round 1 publishes one
    run_round(controller_v2, fake_scheduler)
    state = load_orchestration_state(controller_v2.out)
    assert state["stage4_zero_progress_rounds"] == 0
    assert [e.stem for e in load_stage4_worklist(controller_v2.worklist, 0).entries] == [entries[1].stem]


def test_two_unchanged_rounds_finalize_and_record_nothing(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    for _ in range(3):
        run_round(controller_v2, fake_scheduler)
    assert _roles(fake_scheduler) == ["controller", "stage4", "controller", "stage4", "controller", "finalizer"]
    assert read_terminal_failures(controller_v2.out) == []                    # U7
    assert all(
        record_of(controller_v2.out, e) is None or not record_of(controller_v2.out, e).get("artifacts")
        for e in controller_v2.entries
    )
    state = load_orchestration_state(controller_v2.out)
    assert state["phase"] == "finalizing" and state["stage4_zero_progress_rounds"] == 2


def test_one_unchanged_round_is_retried(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    run_round(controller_v2, fake_scheduler)
    run_round(controller_v2, fake_scheduler)
    assert _roles(fake_scheduler)[-1] == "stage4"
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-2")


def test_a_stall_reads_5_to_the_waiter(controller_v2, fake_scheduler, monkeypatch):
    """Review I4: from a genuinely stalled run, through the run state, to the waiter's exit 5.

    The leftovers keep a record with only ``stages.stage3`` and no journal
    row (U7), so ``resolve_run_state`` counts them unresolved, and the
    waiter, reading the real ``_run_snapshot``, ends ``ENDED_UNPUBLISHED``
    even though the finalizer published.
    """
    from phenotypic._cli import _cli_wait
    from phenotypic.sdk_ import resolve_run_state

    finish_stage3(controller_v2, stage4_pending=2)
    for _ in range(3):
        run_round(controller_v2, fake_scheduler)
    assert load_orchestration_state(controller_v2.out)["stage4_zero_progress_rounds"] == 2

    diagnostics = resolve_run_state(controller_v2.out, depth="deep").diagnostics
    assert (diagnostics.accepted, diagnostics.verified, diagnostics.failed) == (2, 0, 0)

    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: no_live_jobs())
    monkeypatch.setattr(_cli_wait, "_ledgered_jobs", lambda d, g: no_live_jobs())
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: controller_v2.epoch)
    monkeypatch.setattr(_cli_wait, "_published", lambda d, g: True)      # its finalizer published
    lines: list[str] = []
    code = _cli_wait.wait_for_run(controller_v2.out, sleep=lambda s: None, echo=lines.append)
    assert code is RunExitCode.ENDED_UNPUBLISHED and int(code) == 5
    assert any("2 accepted image(s) have neither a success nor a terminal failure record" in line
               for line in lines)


# ---------------------------------------------------------------------------
# The worklist
# ---------------------------------------------------------------------------


def test_the_worklist_partitions_only_the_retryable_set(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=5, done=3, n_shards=2)
    run_staged_controller(controller_v2.config_path)
    shards = [load_stage4_worklist(controller_v2.worklist, i).entries for i in (0, 1)]
    assert sorted(len(s) for s in shards) == [2, 3]
    assert not ({e.identity for s in shards for e in s} & controller_v2.done_identities)


@pytest.mark.parametrize("n, k", [(1, 1), (5, 2), (7, 3), (1, 4), (40, 8)])
def test_the_partition_rule(n, k):
    entries = [entry(f"img{i:02d}") for i in range(n)]
    path = write_stage4_worklist(tmp_worklist(), round_index=1, manifest_path=Path("m.json"),
                                 retryable=list(reversed(entries)), n_shards=k)
    parts = [load_stage4_worklist(path, i).entries for i in range(k)]
    assert sorted(e.identity for p in parts for e in p) == sorted(e.identity for e in entries)  # no loss
    assert sum(map(len, parts)) == n                                                          # no dup
    assert max(map(len, parts)) - min(map(len, parts)) <= 1                                   # near-even
    assert parts == [load_stage4_worklist(write_stage4_worklist(tmp_worklist(), round_index=1,
                     manifest_path=Path("m.json"), retryable=entries, n_shards=k), i).entries
                     for i in range(k)]                                                         # order-free


def test_the_worklist_names_no_shard_it_lacks():
    path = write_stage4_worklist(tmp_worklist(), round_index=1, manifest_path=Path("m.json"),
                                 retryable=[entry("a")], n_shards=2)
    with pytest.raises(ValueError, match="shard 2 does not exist"):
        load_stage4_worklist(path, 2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_stage4_worklist(path, 0)


def test_more_shards_than_work_leaves_empty_shards_harmless(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1, n_shards=4)
    run_staged_controller(controller_v2.config_path)
    assert [len(load_stage4_worklist(controller_v2.worklist, i).entries) for i in range(4)] == [1, 0, 0, 0]
    # The empty shards wrote nothing; the next classification sees the one image.
    run_round(controller_v2, fake_scheduler)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-2")
    assert [len(load_stage4_worklist(controller_v2.worklist, i).entries) for i in range(4)] == [1, 0, 0, 0]


@pytest.mark.parametrize("pending, n_shards, array", [(1, 4, "0-0"), (5, 2, "0-1"), (3, 3, "0-2")])
def test_only_filled_shards_get_a_gpu(controller_v2, fake_scheduler, pending, n_shards, array):
    """Review M13: the round's --array covers only the shards the worklist fills."""
    finish_stage3(controller_v2, stage4_pending=pending, n_shards=n_shards)
    run_staged_controller(controller_v2.config_path)
    assert fake_scheduler.extra_args["stage4-round-1"] == (f"--array={array}",)


def test_a_round_already_ledgered_keeps_its_worklist(controller_v2, fake_scheduler):
    """Review M12: a controller that died after sbatch leaves the round's partition in use.

    Its successor re-classifies a different set (one image committed
    meanwhile) but must not rewrite the worklist the running shards read.
    """
    entries = finish_stage3(controller_v2, stage4_pending=3, n_shards=2)
    write_stage4_worklist(controller_v2.worklist, round_index=1, manifest_path=controller_v2.manifest,
                          retryable=entries, n_shards=2)
    before = controller_v2.worklist.read_bytes()
    append_job_ledger(controller_v2.out, epoch=controller_v2.epoch, token="stage4-round-1",
                      role="stage4", round_index=1, status="intent")
    from tests.unit.cli.conftest import _stage4_image

    _stage4_image(controller_v2.out, entries[0].stem, done=True)            # committed meanwhile
    run_staged_controller(controller_v2.config_path)
    assert controller_v2.worklist.read_bytes() == before
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")


def test_an_unanswered_stage4_array_keeps_the_controller_waiting(controller_v2, fake_scheduler, monkeypatch):
    """Review M11: ``scheduler_job_is_active`` answering ``None`` is never read as ended."""
    finish_stage3(controller_v2, stage4_pending=2)
    run_staged_controller(controller_v2.config_path)
    active = fake_scheduler.job_of("stage4-round-1")
    fake_scheduler.become_next_controller(controller_v2.out)
    monkeypatch.setattr("phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
                        lambda job_id: None if job_id == active else False)
    before = list(fake_scheduler.calls)
    run_staged_controller(controller_v2.config_path)
    new = [call for call in fake_scheduler.calls if call not in before]
    assert [role for role, _, _ in new] == ["controller"]          # only the next recovery controller
    state = load_orchestration_state(controller_v2.out)
    assert state["active_job_id"] == active and state["scheduler_query_failed"] is True
    assert state["stage4_round"] == 1


def test_a_timeout_mid_commit_goes_back_into_the_worklist(controller_v2, fake_scheduler):
    """Review M11, the controller half: a shard killed after writing signals, before the record.

    No record means ``stage4_status`` is ``"pending"``, so the image is in
    the next round's worklist, and its signals are kept for that round to
    reuse (the worker half, publish-only with no re-embedding, is P4 T4's).
    """
    entries = finish_stage3(controller_v2, stage4_pending=2)
    signals = write_signals(controller_v2, regions={"objects", "grid"})
    run_staged_controller(controller_v2.config_path)
    worklist = {e.stem for e in load_stage4_worklist(controller_v2.worklist, 0).entries}
    assert worklist == {e.stem for e in entries}
    assert all(path.is_file() for path in signals)


# ---------------------------------------------------------------------------
# The ledger, the sidecar rule and the epoch fence
# ---------------------------------------------------------------------------


def test_stage4_rounds_are_ledgered_as_stage4(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1)
    run_staged_controller(controller_v2.config_path)
    assert ("stage4", "stage4-round-1") in fake_scheduler.submitted_roles()


def test_a_stage4_round_is_ledgered_by_the_real_submitter(controller_v2, monkeypatch):
    # The real submit_with_intent, with sbatch faked one level down.
    from phenotypic._cli import _cli_staged_controller as controller_module

    finish_stage3(controller_v2, stage4_pending=1)
    monkeypatch.setenv("SLURM_JOB_ID", "1000")
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.setattr(controller_module, "scheduler_job_is_active", lambda job_id: False)
    monkeypatch.setattr(controller_module, "update_job_dependency", lambda job_id, deps: True)
    ids = iter(range(5000, 5010))

    class _Done:
        returncode, stderr = 0, ""

        def __init__(self) -> None:
            self.stdout = f"{next(ids)}\n"

    monkeypatch.setattr(orchestration_module.subprocess, "run", lambda *a, **k: _Done())
    run_staged_controller(controller_v2.config_path)
    rows = [r for r in read_job_ledger(controller_v2.out, epoch=controller_v2.epoch)
            if r.get("status") == "submitted"]
    assert ("stage4", "stage4-round-1") in {(r["role"], r["token"]) for r in rows}


def test_a_stage4_ledger_row_is_live(tmp_path, monkeypatch):
    append_job_ledger(tmp_path, epoch="e", token="stage4-round-1", role="stage4", round_index=1,
                      job_id="77", status="submitted")
    monkeypatch.setattr(orchestration_module, "scheduler_job_is_active", lambda job_id: job_id == "77")
    assert "77" in live_slurm_job_ids(tmp_path).live


def test_no_sidecar_job(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    run_staged_controller(controller_v2.config_path)
    assert sorted(fake_scheduler.active_roles()) == ["controller", "stage4"]   # one work array plus its recovery controller


def test_no_sidecar_job_in_any_round(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=2)
    for _ in range(3):
        run_round(controller_v2, fake_scheduler)
        assert len([r for r in fake_scheduler.active_roles() if r != "controller"]) == 1
        assert fake_scheduler.active_roles().count("controller") == 1


def test_a_fenced_epoch_submits_no_round(controller_v2, fake_scheduler, monkeypatch):
    finish_stage3(controller_v2, stage4_pending=2)
    deactivate_orchestration(controller_v2.out)
    run_staged_controller(controller_v2.config_path)
    assert fake_scheduler.calls == []


def test_a_terminal_incomplete_run_ends_its_trailing_controller_quietly(controller_v2, fake_scheduler):
    """Review I3, smoke F4: the finalizer closed the run ``terminal_incomplete``.

    The fence is closed, so ``assert_active_epoch`` would raise; the
    trailing controller must exit 0 and submit nothing.
    """
    finish_stage3(controller_v2, stage4_pending=1)
    assert deactivate_orchestration(controller_v2.out, "terminal_incomplete")
    assert load_orchestration_state(controller_v2.out)["phase"] == "terminal_incomplete"
    run_staged_controller(controller_v2.config_path)
    assert fake_scheduler.calls == []


def test_a_cancel_during_the_transition_sweeps_nothing(controller_v2, fake_scheduler):
    # The cancel lands after the controller passed its own epoch check: the
    # sweep's guard refuses, so no signal is deleted and no round is submitted.
    finish_stage3(controller_v2, stage4_pending=0)
    planted = plant_signals_for_done_image(controller_v2)
    submit = fake_scheduler.submit

    def _cancel_after_prearm(output_dir, **kwargs):
        job_id = submit(output_dir, **kwargs)
        if kwargs["role"] == "controller":
            deactivate_orchestration(controller_v2.out)
        return job_id

    fake_scheduler.monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent", _cancel_after_prearm
    )
    with pytest.raises(RuntimeError, match="inactive|fenced"):
        run_staged_controller(controller_v2.config_path)
    assert all(path.is_file() for path in planted)
    assert _roles(fake_scheduler) == ["controller"]


def test_orphans_are_swept_before_classifying(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=0)
    plant_signals_for_done_image(controller_v2)
    run_staged_controller(controller_v2.config_path)
    assert no_stage4_signals(controller_v2.out)
    assert _roles(fake_scheduler)[-1] == "finalizer"


def test_a_pending_images_signals_are_kept(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1)
    kept = write_signals(controller_v2, regions={"objects"})
    plant_signals_for_done_image(controller_v2)
    run_staged_controller(controller_v2.config_path)
    assert all(path.is_file() for path in kept)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_stage4_terminal_cases(controller_v2):
    entries = controller_v2.entries_with(
        failed=1, no_stage3=1, wrong_work_id=1, reference_changed=1, pending=2, done=1)
    retryable, terminal = _classify_stage4(controller_v2.config, entries)
    assert len(terminal) == 4 and len(retryable) == 2
    assert {e.stem[:-2] for e in terminal} == {"failed", "no_stage3", "wrong_work_id", "reference_changed"}
    assert {e.stem[:-2] for e in retryable} == {"pending"}


def test_a_failure_from_another_epoch_is_retryable(controller_v2):
    # This epoch's continuation chose to retry it (--retry-failures).
    entries = controller_v2.entries_with(old_epoch_failed=1, failed=1)
    retryable, terminal = _classify_stage4(controller_v2.config, entries)
    assert [e.stem[:-2] for e in retryable] == ["old_epoch_failed"]
    assert [e.stem[:-2] for e in terminal] == ["failed"]


def test_the_terminal_images_are_counted_by_cause(controller_v2, fake_scheduler, caplog):
    """Review M5: the operator is told how many images Stage 4 skips, and which kind."""
    from phenotypic._cli._cli_staged_orchestration import write_staged_manifest
    from tests.unit.cli.conftest import _install_stage4_state

    finish_stage3(controller_v2, stage4_pending=1)
    extra = controller_v2.entries_with(failed=1, no_stage3=2)
    write_staged_manifest(controller_v2.manifest, [*controller_v2.entries, *extra])
    _install_stage4_state(controller_v2.out, [*controller_v2.entries, *extra])
    with caplog.at_level("INFO", logger="phenotypic._cli._cli_staged_controller"):
        run_staged_controller(controller_v2.config_path)
    assert any("Stage 4 skips 3 image(s): 1 failed terminally this epoch, 2 never finished" in r.message
               for r in caplog.records)


def test_terminal_images_are_not_in_the_worklist(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=1)
    extra = controller_v2.entries_with(failed=1, no_stage3=1)
    from phenotypic._cli._cli_staged_orchestration import write_staged_manifest
    from tests.unit.cli.conftest import _install_stage4_state

    write_staged_manifest(controller_v2.manifest, [*controller_v2.entries, *extra])
    _install_stage4_state(controller_v2.out, [*controller_v2.entries, *extra])
    run_staged_controller(controller_v2.config_path)
    assert [e.stem for e in load_stage4_worklist(controller_v2.worklist, 0).entries] == ["img00"]


def test_only_terminal_leftovers_go_to_the_finalizer(controller_v2, fake_scheduler):
    finish_stage3(controller_v2, stage4_pending=0)
    extra = controller_v2.entries_with(failed=2)
    from phenotypic._cli._cli_staged_orchestration import write_staged_manifest

    write_staged_manifest(controller_v2.manifest, extra)
    run_staged_controller(controller_v2.config_path)
    assert "stage4" not in _roles(fake_scheduler) and _roles(fake_scheduler)[-1] == "finalizer"


# ---------------------------------------------------------------------------
# Config versions and phase routing
# ---------------------------------------------------------------------------


def test_a_v1_config_still_runs_three_phases(v1_controller_config, fake_scheduler):
    finish_stage3(v1_controller_config, stage4_pending=2)
    run_staged_controller(v1_controller_config.config_path)
    roles = _roles(fake_scheduler)
    assert roles[-1] == "finalizer" and "stage4" not in roles


def test_a_v1_config_drives_stage1_then_stage2(v1_controller_config, fake_scheduler):
    run_staged_controller(v1_controller_config.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage1", "stage1-chunk-0")


def test_a_v1_config_in_phase_stage4_takes_the_stage3_path(v1_controller_config, fake_scheduler):
    # Unreachable: only a v2 config's submitter writes phase "stage4", so the
    # controller has no v1 branch for it. Should it happen, the generic path
    # is safe: every image of an all-Stage-4 plan has stages.stage3, so Stage
    # 2 classifies nothing retryable and the run takes the stage3 path.
    finish_stage3(v1_controller_config, stage4_pending=2)
    state = load_orchestration_state(v1_controller_config.out)
    state.update(phase="stage4", stage3_index=0)
    save_orchestration_state(v1_controller_config.out, state)
    run_staged_controller(v1_controller_config.config_path)
    assert _roles(fake_scheduler) == ["controller", "stage3"]


def test_no_detector_phases_start_at_stage3(controller_v2_no_detector, fake_scheduler):
    run_staged_controller(controller_v2_no_detector.config_path)
    assert fake_scheduler.submitted_roles()[-1][0] == "stage3"
    assert not {"stage1", "stage2"} & set(_roles(fake_scheduler))


def test_no_detector_run_goes_stage3_then_stage4(controller_v2_no_detector, fake_scheduler):
    run_staged_controller(controller_v2_no_detector.config_path)
    finish_stage3(controller_v2_no_detector, stage4_pending=1)                 # the chunk ran
    fake_scheduler.set_active(fake_scheduler.job_of("stage3-chunk-0"), False)
    run_round(controller_v2_no_detector, fake_scheduler)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")


def test_an_all_stage4_continuation_starts_at_stage4(controller_v2, fake_scheduler):
    # The CLI hands `staged_resume_phase = "stage4"`; the submitter writes it as the phase.
    finish_stage3(controller_v2, stage4_pending=2)
    state = load_orchestration_state(controller_v2.out)
    state.update(phase="stage4", stage3_index=0)
    save_orchestration_state(controller_v2.out, state)
    run_staged_controller(controller_v2.config_path)
    assert _roles(fake_scheduler) == ["controller", "stage4"]


@pytest.mark.parametrize(
    "change",
    [
        {"version": 3},
        {"phases": ["stage1", "stage2"]},
        {"phases": ["stage4", "stage3"]},
        {"phases": ["stage3", "stage3", "stage4"]},
        {"phases": "stage3"},
    ],
)
def test_an_unsupported_config_is_refused(controller_v2, fake_scheduler, change):
    controller_v2.update_config(**change)
    with pytest.raises(ValueError, match="Unsupported|unsupported phases"):
        run_staged_controller(controller_v2.config_path)


def test_a_stage4_config_without_its_script_is_refused(controller_v2, fake_scheduler):
    config = controller_v2.config
    del config["stage4_script"]
    controller_v2.config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="stage4_script"):
        run_staged_controller(controller_v2.config_path)


# ---------------------------------------------------------------------------
# The CLI hands an all-Stage-4 SLURM continuation the "stage4" phase [torch]
# ---------------------------------------------------------------------------


def test_the_cli_starts_an_all_stage4_slurm_continuation_at_stage4(
    stage4_slurm_waiting, submissions, monkeypatch
):
    """C12's interim mapping (``"stage4"`` -> ``"stage3"`` on SLURM) is gone.

    The real, lifted gate admits the run (review M8: no ``stage4_supported``
    patch), and the strategy is replaced by one that records the phase.
    """
    from datetime import datetime

    from phenotypic._cli._cli_types import ExecutionResults
    from tests._tiny_run import run_cli

    observed: dict[str, object] = {}

    class _Submitted:
        def execute(self, datasets, output_dir):
            now = datetime.now()
            return ExecutionResults(datasets={}, total_images=0, total_completed=0, total_failed=0,
                                    execution_mode="slurm", start_time=now, end_time=now, submitted=True)

    def _strategy(config, output_manager):
        observed["phase"] = config.staged_resume_phase
        return _Submitted()

    monkeypatch.setattr("phenotypic.phenotypicCLI.create_execution_strategy", _strategy)
    result = run_cli(*stage4_slurm_waiting.cli_args, **stage4_slurm_waiting.cli_kwargs, slurm=True,
                     extra=["--embed-stage", "gpu", "--skip-validation"])
    assert observed == {"phase": "stage4"}, result.output

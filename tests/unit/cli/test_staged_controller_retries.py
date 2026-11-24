"""The staged controller resubmits Stage-1 and Stage-3 tasks that died leaving nothing (#282 U15/D5).

Smoke F2: four Stage-3 tasks were OOM-killed, left no record and no journal
row, and were never resubmitted, so the run could only end unfinished. The
controller now re-classifies after a stage's last chunk, and resubmits only
the dead tasks of each chunk (``--array=<ids>``), in rounds bounded by the
zero-progress rule on that stage's own counters.

The scheduler is faked at the controller's seams (``fake_scheduler``); the
chunk scripts are written by the real script writer, so the task map the
controller reads is the one a real chunk carries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic._cli._cli_failure_tracker import (
    PerImageScientificError,
    append_terminal_failure,
    read_terminal_failures,
)
from phenotypic._cli._cli_stage2_token import write_stage2_raw, write_stage2_token
from phenotypic._cli._cli_staged_controller import (
    _chunk_task_indices,
    recorded_stall_lines,
    run_staged_controller,
)
from phenotypic._cli._cli_staged_orchestration import (
    load_orchestration_state,
    load_stage4_worklist,
    save_orchestration_state,
    write_staged_manifest,
)
from phenotypic._cli._cli_staged_resume import write_stage3_completion_marker
from tests.unit.cli.conftest import (
    STAGE2_SLOT,
    _install_stage4_state,
    _stage4_image,
    run_round,
)


def _write_chunks(fixture, stage: str, windows: list[tuple[int, int]]) -> list[Path]:
    """Real chunk scripts for *stage* over the manifest *windows*, named in the config."""
    from phenotypic._cli._cli_staged_slurm import _write_stage_script

    scripts = [
        _write_stage_script(
            fixture.out / "scripts", fixture.out / "logs", f"{stage}_chunk{i}",
            f"phenotypic-{stage}", {"slurm_partition": "intel"}, list(range(start, end)),
            "echo $CURRENT_TASK_INDEX",
        )
        for i, (start, end) in enumerate(windows)
    ]
    fixture.update_config(**{f"{stage}_scripts": [str(path) for path in scripts]})
    return scripts


def _images(fixture, *, finished: int, dead: int, failed: int = 0) -> list:
    """``finished`` images through Stage 3, then ``failed`` journalled ones, then ``dead`` ones."""
    entries = []
    for i in range(finished + failed + dead):
        kind = "fin" if i < finished else ("bad" if i < finished + failed else "dead")
        entries.append(_stage4_image(fixture.out, f"{kind}{i:02d}", stage3=kind == "fin"))
    for entry in entries[finished:finished + failed]:
        assert append_terminal_failure(
            fixture.out, work_id=entry.work_id, dataset=entry.dataset,
            relative_image_path=entry.relative_image_path, failed_stage="stage3",
            exception=PerImageScientificError("stage3", ValueError("no colonies")),
            attempt_id="attempt", lifecycle_epoch=fixture.epoch,
        )
    write_staged_manifest(fixture.manifest, entries)
    _install_stage4_state(fixture.out, entries)
    fixture.entries = entries
    return entries


def _after_the_last_stage3_chunk(fixture) -> None:
    state = load_orchestration_state(fixture.out)
    state.update(phase="stage3", stage1_index=1, stage3_index=len(fixture.config["stage3_scripts"]))
    save_orchestration_state(fixture.out, state)


def _stage3_retry_runs(fixture, scheduler, entries) -> None:
    """The active Stage-3 retry's tasks finish their images, then the next controller runs."""
    for entry in entries:
        write_stage3_completion_marker(fixture.out, entry.dataset, entry.image_name, entry.stem)
    state = load_orchestration_state(fixture.out)
    scheduler.set_active(str(state["active_job_id"]), False)
    scheduler.become_next_controller(fixture.out)
    run_staged_controller(fixture.config_path)


def _ends_without_change(fixture, scheduler) -> None:
    state = load_orchestration_state(fixture.out)
    scheduler.set_active(str(state["active_job_id"]), False)
    scheduler.become_next_controller(fixture.out)
    run_staged_controller(fixture.config_path)


def test_the_task_map_is_read_from_the_chunk_script(controller_v2_no_detector):
    (script,) = _write_chunks(controller_v2_no_detector, "stage3", [(2500, 2504)])
    assert _chunk_task_indices(script) == [2500, 2501, 2502, 2503]
    with pytest.raises(ValueError, match="cannot read"):
        _chunk_task_indices(script.with_name("missing.sh"))


def test_oom_killed_stage3_tasks_are_resubmitted_and_their_images_complete(
    controller_v2_no_detector, fake_scheduler
):
    fixture = controller_v2_no_detector
    _write_chunks(fixture, "stage3", [(0, 3), (3, 6)])
    entries = _images(fixture, finished=3, dead=3)
    # Chunk 1's three tasks died (OUT_OF_MEMORY): no marker, record or journal row.
    dead = entries[3:]
    _after_the_last_stage3_chunk(fixture)

    run_staged_controller(fixture.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage3", "stage3-retry-1-chunk-1")
    assert fake_scheduler.extra_args["stage3-retry-1-chunk-1"] == ("--array=0,1,2",)
    state = load_orchestration_state(fixture.out)
    assert state["phase"] == "stage3" and state["stage3_retry_round"] == 1

    _stage3_retry_runs(fixture, fake_scheduler, dead)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")
    assert read_terminal_failures(fixture.out) == []
    # Every image finished Stages 1-3, the retried ones included, and waits on Stage 4.
    assert {e.stem for e in load_stage4_worklist(fixture.worklist, 0).entries} == {e.stem for e in entries}


def test_only_the_dead_tasks_of_each_chunk_are_resubmitted_one_chunk_at_a_time(
    controller_v2_no_detector, fake_scheduler
):
    fixture = controller_v2_no_detector
    _write_chunks(fixture, "stage3", [(0, 3), (3, 6)])
    entries = _images(fixture, finished=2, dead=4)       # dead: 2, 3, 4, 5
    write_stage3_completion_marker(fixture.out, "plate", entries[3].image_name, entries[3].stem)
    _after_the_last_stage3_chunk(fixture)

    run_staged_controller(fixture.config_path)
    assert fake_scheduler.extra_args["stage3-retry-1-chunk-0"] == ("--array=2",)
    _stage3_retry_runs(fixture, fake_scheduler, [entries[2]])
    # Chunk 1's dead tasks are the same round's second submission.
    assert fake_scheduler.submitted_roles()[-1] == ("stage3", "stage3-retry-1-chunk-1")
    assert fake_scheduler.extra_args["stage3-retry-1-chunk-1"] == ("--array=1,2",)
    assert load_orchestration_state(fixture.out)["stage3_retry_round"] == 1


def test_a_journalled_failure_is_never_resubmitted(controller_v2_no_detector, fake_scheduler):
    fixture = controller_v2_no_detector
    _write_chunks(fixture, "stage3", [(0, 3)])
    _images(fixture, finished=2, failed=1, dead=0)
    _after_the_last_stage3_chunk(fixture)
    run_staged_controller(fixture.config_path)
    roles = [role for role, _ in fake_scheduler.submitted_roles()]
    assert "stage3" not in roles and roles[-1] == "stage4"
    assert len(load_stage4_worklist(fixture.worklist, 0).entries) == 2      # the failure is not Stage 4's


def test_an_always_failing_image_ends_with_the_stall(controller_v2_no_detector, fake_scheduler):
    """Two unchanged rounds stop retrying; nothing is recorded, so the waiter reads 5."""
    from phenotypic._cli._cli_wait import RunExitCode, decide_wait_outcome
    from phenotypic.sdk_ import resolve_run_state
    from tests.unit.cli.conftest import no_live_jobs

    fixture = controller_v2_no_detector
    _write_chunks(fixture, "stage3", [(0, 2)])
    entries = _images(fixture, finished=1, dead=1)
    _after_the_last_stage3_chunk(fixture)

    run_staged_controller(fixture.config_path)                 # round 1
    _ends_without_change(fixture, fake_scheduler)              # round 2: one unchanged round is retried
    _ends_without_change(fixture, fake_scheduler)              # the second unchanged: stall
    tokens = [token for role, token in fake_scheduler.submitted_roles() if role == "stage3"]
    assert tokens == ["stage3-retry-1-chunk-0", "stage3-retry-2-chunk-0"]
    state = load_orchestration_state(fixture.out)
    assert state["stage3_zero_progress_rounds"] == 2
    assert fake_scheduler.submitted_roles()[-1][0] == "stage4"   # the finished image goes on
    assert any("stage3" in line and "died without a result" in line
               for line in recorded_stall_lines(state))
    assert read_terminal_failures(fixture.out) == []           # no verdict on the image

    diagnostics = resolve_run_state(fixture.out, depth="deep").diagnostics
    assert diagnostics.failed == 0 and diagnostics.accepted - diagnostics.verified - diagnostics.failed == 2
    assert decide_wait_outcome(
        "incomplete", no_live_jobs(), published=True, process_run=False,
        unresolved=diagnostics.accepted - diagnostics.verified - diagnostics.failed,
    ) is RunExitCode.ENDED_UNPUBLISHED
    del entries


def test_a_staged_role_retries_only_an_image_stage3_can_replay(controller_v2, fake_scheduler):
    """With a GpuDetector, a dead Stage-3 task's image needs its Stage-2 signal (both halves)."""
    fixture = controller_v2
    _write_chunks(fixture, "stage3", [(0, 3)])
    entries = _images(fixture, finished=1, dead=2)
    write_stage2_raw(fixture.out, "plate", entries[1].stem, np.zeros((4, 4), np.uint16), STAGE2_SLOT)
    write_stage2_token(fixture.out, "plate", entries[1].stem, STAGE2_SLOT, objmap_shape=(4, 4))
    _after_the_last_stage3_chunk(fixture)
    run_staged_controller(fixture.config_path)
    assert fake_scheduler.extra_args["stage3-retry-1-chunk-0"] == ("--array=1",)


def test_a_dead_stage1_task_is_resubmitted_before_stage2(controller_v2, fake_scheduler):
    from phenotypic.sdk_ import zarr_store_path

    fixture = controller_v2
    _write_chunks(fixture, "stage1", [(0, 3)])
    entries = _images(fixture, finished=3, dead=0)
    import shutil

    shutil.rmtree(zarr_store_path(fixture.out, "plate", entries[2].stem))   # its task was killed
    state = load_orchestration_state(fixture.out)
    state.update(phase="stage1", stage1_index=1)
    save_orchestration_state(fixture.out, state)
    run_staged_controller(fixture.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage1", "stage1-retry-1-chunk-0")
    assert fake_scheduler.extra_args["stage1-retry-1-chunk-0"] == ("--array=2",)
    assert load_orchestration_state(fixture.out)["phase"] == "stage1"


def test_a_v1_config_of_an_in_flight_281_run_still_works(v1_controller_config, fake_scheduler):
    """A #281 state carries none of the retry counters; they start at zero, and the run finishes."""
    fixture = v1_controller_config
    _write_chunks(fixture, "stage3", [(0, 2)])
    entries = _images(fixture, finished=1, dead=1)
    write_stage2_raw(fixture.out, "plate", entries[1].stem, np.zeros((4, 4), np.uint16), STAGE2_SLOT)
    write_stage2_token(fixture.out, "plate", entries[1].stem, STAGE2_SLOT, objmap_shape=(4, 4))
    _after_the_last_stage3_chunk(fixture)
    state = load_orchestration_state(fixture.out)
    assert not any(key.startswith(("stage3_retry", "stage3_zero", "stage4_")) for key in state)
    run_staged_controller(fixture.config_path)
    assert fake_scheduler.submitted_roles()[-1] == ("stage3", "stage3-retry-1-chunk-0")
    _stage3_retry_runs(fixture, fake_scheduler, [entries[1]])
    roles = [role for role, _ in fake_scheduler.submitted_roles()]
    assert roles[-1] == "finalizer" and "stage4" not in roles


def test_unreadable_chunk_scripts_fall_back_to_no_retry(controller_v2_no_detector, fake_scheduler):
    fixture = controller_v2_no_detector                         # its stage3.sh was never written
    _images(fixture, finished=1, dead=1)
    _after_the_last_stage3_chunk(fixture)
    run_staged_controller(fixture.config_path)
    assert "stage3" not in [role for role, _ in fake_scheduler.submitted_roles()]


def test_run_round_helper_is_unaffected(controller_v2, fake_scheduler):
    """Every image of ``finish_stage3`` has its marker, so no retry precedes Stage 4."""
    from tests.unit.cli.conftest import finish_stage3

    finish_stage3(controller_v2, stage4_pending=2)
    run_round(controller_v2, fake_scheduler)
    assert fake_scheduler.submitted_roles()[-1] == ("stage4", "stage4-round-1")

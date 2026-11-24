"""A SLURM ``--mode measure`` pass ends by its own leftovers (#282 U16, U18). Torch-free.

U16: a measure pass that leaves any forward-published manifest image neither
republished under its epoch nor journalled ends unfinished. The image keeps
its forward record, which verifies, so the run state alone reads complete;
the measure finalizer counts those images, names them in its log, and
closes its fence failed, so ``phenotypic-wait`` returns 5. Exit 0 means every
image really was re-measured and re-embedded.

U18: a measure step leaves a per-epoch measure-done marker after its table
rewrite. An image without one was not re-measured by this pass: the
controller resubmits it (U15), Stage 4 never takes it, and U16 counts it.

Every exit code here is read from the real fence, the real run state and
the real journal, through ``_run_finalize`` and ``wait_for_run``; only the
scheduler is replaced, by "nothing live".
"""

from __future__ import annotations

import logging

import pytest

from phenotypic._cli import _cli_wait
from phenotypic._cli._cli_checkpoint_handler import _run_finalize
from phenotypic._cli._cli_slurm_lifecycle import FENCE_CLOSED_BY_FINALIZER
from phenotypic._cli._cli_staged_controller import _classify_stage4, _dead_stage3_indices
from phenotypic._cli._cli_staged_orchestration import LiveSlurmJobs, load_orchestration_state
from phenotypic._cli._cli_staged_resume import (
    measure_done_marker_exists,
    measure_done_marker_path,
)
from phenotypic.sdk_ import RunExitCode, progress_dir
from tests._tiny_run import (
    build_measure_epoch_tree,
    fence_of,
    journal_measure_failure,
)


def _forward_pipeline(out):
    """The forward run's own pipeline file: a same-pipeline measure (the C17 case)."""
    return out.parent / "tiny_run.pht-pipe.json"


def _finalize_and_wait(tree, monkeypatch) -> tuple[RunExitCode, list[str]]:
    """The finalizer, then ``phenotypic-wait`` over the real fence and run state."""
    _run_finalize(tree.out, progress_dir(tree.out), epoch=tree.epoch)
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: LiveSlurmJobs())
    monkeypatch.setattr(_cli_wait, "_ledgered_jobs", lambda d, g: LiveSlurmJobs())
    lines: list[str] = []
    code = _cli_wait.wait_for_run(tree.out, sleep=lambda s: None, echo=lines.append)
    return code, lines


def _assert_unfinished_close(tree) -> None:
    fence = fence_of(tree.out)
    assert fence["active"] is False and fence["closed_by"] == FENCE_CLOSED_BY_FINALIZER
    assert fence["terminal_status"] == "failed"
    assert load_orchestration_state(tree.out)["phase"] == "terminal_incomplete"


# ---------------------------------------------------------------------------
# U16: the three ways a pass leaves an image unfinished, each exit 5
# ---------------------------------------------------------------------------


def test_a_same_pipeline_stage4_stall_exits_5(tmp_path, finished_tiny_run, monkeypatch, caplog):
    """Deviation 1: Stage 4 republished image 0 and stalled on image 1 (U7)."""
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, prior_pass=True,
                                    pipeline=_forward_pipeline(finished_tiny_run))
    tree.republish(0)
    retryable, _terminal = _classify_stage4(tree.config, tree.entries)
    assert [e.stem for e in retryable] == [tree.entries[1].stem]       # left by the stall
    with caplog.at_level(logging.WARNING):
        code, lines = _finalize_and_wait(tree, monkeypatch)
    assert code is RunExitCode.ENDED_UNPUBLISHED, lines
    _assert_unfinished_close(tree)
    # The finalizer names the image, and the waiter points at that log.
    assert any(tree.entries[1].image_name in r.getMessage() for r in caplog.records)
    assert not any(f"unfinished: ds/{tree.entries[0].image_name}" in r.getMessage()
                   for r in caplog.records)
    assert any("ended unfinished" in line and "1 image(s)" in line for line in lines), lines


@pytest.mark.parametrize("same_pipeline", [True, False], ids=["same-pipeline", "differing-pipeline"])
def test_dead_measure_steps_after_u15_gives_up_exit_5(tmp_path, finished_tiny_run, monkeypatch, same_pipeline):
    """Row 3 of C1: both measure steps died before writing anything, and U15's retries gave up.

    No marker (U18), so they are dead tasks for U15, and Stage 4 never takes
    them; once U15 stops resubmitting, the finalizer counts both.
    """
    kwargs = {"prior_pass": True, "pipeline": _forward_pipeline(finished_tiny_run)} if same_pipeline else {}
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, measure=False, **kwargs)
    assert _dead_stage3_indices(tree.config, tree.entries) == [0, 1]
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    assert not retryable and len(terminal) == 2
    code, lines = _finalize_and_wait(tree, monkeypatch)
    assert code is RunExitCode.ENDED_UNPUBLISHED, lines
    _assert_unfinished_close(tree)


def test_a_differing_pipeline_stall_exits_5(tmp_path, finished_tiny_run, monkeypatch):
    """Exit 5 here no longer rests on the root having changed: the finalizer's count says so."""
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    tree.republish(0)
    code, lines = _finalize_and_wait(tree, monkeypatch)
    assert code is RunExitCode.ENDED_UNPUBLISHED, lines
    _assert_unfinished_close(tree)


# ---------------------------------------------------------------------------
# The controls: every image republished is 0; a journalled one is 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("same_pipeline", [True, False], ids=["same-pipeline", "differing-pipeline"])
def test_every_image_republished_exits_0(tmp_path, finished_tiny_run, monkeypatch, same_pipeline):
    kwargs = {"prior_pass": True, "pipeline": _forward_pipeline(finished_tiny_run)} if same_pipeline else {}
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, **kwargs)
    for index in range(len(tree.entries)):
        tree.republish(index)
    code, lines = _finalize_and_wait(tree, monkeypatch)
    assert code is RunExitCode.COMPLETE, lines
    fence = fence_of(tree.out)
    assert fence["closed_by"] == FENCE_CLOSED_BY_FINALIZER and not fence.get("terminal_status")
    assert load_orchestration_state(tree.out)["phase"] == "complete"


def test_a_journalled_image_and_a_republished_one_exit_3(tmp_path, finished_tiny_run, monkeypatch):
    """Image 1 failed in Stage 4 as U8 has it: journalled, then its record republished over what was carried."""
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    tree.republish(0)
    journal_measure_failure(tree, image=1)
    tree.republish(1)
    code, lines = _finalize_and_wait(tree, monkeypatch)
    assert code is RunExitCode.COMPLETE_WITH_FAILURES, lines


def test_an_image_with_no_forward_record_is_not_the_pass_s_to_count(tmp_path, finished_tiny_run, monkeypatch):
    """U16 counts forward-published images only: one the forward run never certified is its own state's."""
    from phenotypic.sdk_ import image_record_path

    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    tree.republish(0)
    entry = tree.entries[1]
    image_record_path(tree.out, entry.dataset, entry.stem).unlink()
    from phenotypic._cli._cli_staged_controller import unfinished_measure_entries

    assert unfinished_measure_entries(tree.out, tree.entries, pipeline_sha256=tree.sha,
                                      epoch=tree.epoch) == []


# ---------------------------------------------------------------------------
# U18: the measure-done marker
# ---------------------------------------------------------------------------


def test_a_step_killed_after_its_table_rewrite_is_re_measured(tmp_path, finished_tiny_run):
    """The core rewrote the table, and the step died before its marker: it is dead, and Stage 4 skips it.

    The same-pipeline case, where the run folder alone cannot tell (P6
    review I1); a real ``run_measure_step`` then writes the marker, and the
    image is Stage 4's.
    """
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_process_single import process_single_store_measure_core
    from phenotypic._cli._cli_staged_slurm_worker import run_measure_step

    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, measure=False, prior_pass=True,
                                    pipeline=_forward_pipeline(finished_tiny_run))
    manager = OutputManager.from_config(tree.out, ".tiff", save_overlays=False, durable_writes=False)
    for entry in tree.entries:                       # the table rewrite, then the kill
        process_single_store_measure_core(
            pipeline_path=tree.pipeline, store_path=tree.store(tree.entries.index(entry)),
            output_dir=tree.out, dataset_name=entry.dataset, image_type="GridImage",
            output_manager=manager, defer_embeddings=True,
        )
    assert _dead_stage3_indices(tree.config, tree.entries) == [0, 1]
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    assert not retryable and len(terminal) == 2      # Stage 4 never takes an unmarked image

    run_measure_step(tree.pipeline, tree.out, "GridImage", tree.entries, 0, epoch=tree.epoch,
                     durable_writes=False)
    assert measure_done_marker_exists(tree.out, tree.epoch, tree.entries[0].dataset, tree.entries[0].stem)
    assert _dead_stage3_indices(tree.config, tree.entries) == [1]
    retryable, _terminal = _classify_stage4(tree.config, tree.entries)
    assert [e.stem for e in retryable] == [tree.entries[0].stem]


def test_a_marker_of_another_epoch_does_not_count(tmp_path, finished_tiny_run):
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    entry = tree.entries[0]
    marker = measure_done_marker_path(tree.out, tree.epoch, entry.dataset, entry.stem)
    other = measure_done_marker_path(tree.out, "an-earlier-pass", entry.dataset, entry.stem)
    other.parent.mkdir(parents=True, exist_ok=True)
    marker.rename(other)
    assert _dead_stage3_indices(tree.config, tree.entries) == [0]


def test_the_marker_lives_under_progress_never_in_the_store(tmp_path, finished_tiny_run):
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    entry = tree.entries[0]
    marker = measure_done_marker_path(tree.out, tree.epoch, entry.dataset, entry.stem)
    assert marker.is_file() and marker.is_relative_to(progress_dir(tree.out))
    assert not marker.is_relative_to(tree.store(0))


def test_a_new_measure_submission_clears_old_markers(tmp_path, finished_tiny_run):
    from phenotypic._cli._cli_staged_resume import clear_measure_done_markers

    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    entry = tree.entries[0]
    assert measure_done_marker_exists(tree.out, tree.epoch, entry.dataset, entry.stem)
    clear_measure_done_markers(tree.out)
    assert not measure_done_marker_exists(tree.out, tree.epoch, entry.dataset, entry.stem)

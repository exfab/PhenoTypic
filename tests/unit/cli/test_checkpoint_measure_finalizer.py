"""The measure-mode finalizer (#282 U10, plan review I9). Torch-free.

A SLURM ``--mode measure`` resubmission on the staged controller has no
forward run proof to write: its finalizer aggregates, takes
``measure_failed`` from the journal under its epoch, closes the fence as its
finalizer and marks the orchestration complete, and never calls
``publish_run_completion_evidence``.
"""

from __future__ import annotations

import pytest

from phenotypic._cli._cli_checkpoint_handler import _run_finalize
from phenotypic._cli._cli_slurm_lifecycle import FENCE_CLOSED_BY_FINALIZER
from phenotypic._cli._cli_staged_orchestration import (
    load_orchestration_state,
    staged_completion_path,
)
from phenotypic.sdk_ import JobMetadataKey, progress_dir
from tests._tiny_run import (
    build_measure_epoch_tree,
    fence_of,
    journal_measure_failure,
    metrics,
    record_slurm_job_metadata,
    spy,
)


@pytest.fixture
def measure_epoch_tree(tmp_path, finished_tiny_run):
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run)
    for index in range(len(tree.entries)):
        tree.republish(index)        # Stage 4 of this pass republished both records
    return tree


def test_the_measure_finalizer_publishes_no_run_proof_and_counts_measure_failures(measure_epoch_tree, monkeypatch):
    proof = spy(monkeypatch, "publish_run_completion_evidence")
    journal_measure_failure(measure_epoch_tree, image=0)
    _run_finalize(measure_epoch_tree.out, progress_dir(measure_epoch_tree.out), epoch=measure_epoch_tree.epoch)
    assert proof.calls == []
    assert metrics(measure_epoch_tree.out)["failed"] == 1                       # measure_failed from the journal
    fence = fence_of(measure_epoch_tree.out)
    assert fence["active"] is False and fence["closed_by"] == FENCE_CLOSED_BY_FINALIZER
    assert not fence.get("terminal_status")
    assert load_orchestration_state(measure_epoch_tree.out)["phase"] == "complete"
    # The forward run's staged-completion marker is not this pass's to write.
    assert not staged_completion_path(measure_epoch_tree.out).is_file()


def test_a_failure_from_another_epoch_is_not_counted(measure_epoch_tree, monkeypatch):
    journal_measure_failure(measure_epoch_tree, image=0)
    measure_epoch_tree.epoch, epoch = "another-epoch", measure_epoch_tree.epoch
    try:
        journal_measure_failure(measure_epoch_tree, image=1)
    finally:
        measure_epoch_tree.epoch = epoch
    _run_finalize(measure_epoch_tree.out, progress_dir(measure_epoch_tree.out), epoch=epoch)
    assert metrics(measure_epoch_tree.out)["failed"] == 1


def test_a_forward_staged_finalizer_still_publishes_its_run_proof(measure_epoch_tree, monkeypatch):
    """The control: the same tree finalized as a forward run takes the forward branch."""
    record_slurm_job_metadata(measure_epoch_tree.out, **{JobMetadataKey.RUN_MODE: "full"})
    proof = spy(monkeypatch, "publish_run_completion_evidence")
    _run_finalize(measure_epoch_tree.out, progress_dir(measure_epoch_tree.out), epoch=measure_epoch_tree.epoch)
    assert len(proof.calls) == 1
    assert staged_completion_path(measure_epoch_tree.out).is_file()

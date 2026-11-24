"""SLURM ``--mode measure`` on the staged controller (#282 P6 T2, U10). Torch-free.

A SLURM measure with ``--gpu-slurm`` resolves ``--embed-stage auto`` to
``gpu`` and goes through ``StagedSlurmStrategy``: phases ``["stage3",
"stage4"]``, the ``"measure"`` Stage-3 role, a version-5 manifest flagged
*republish* (the mode's one carrier), and job metadata that tells the
finalizer it serves a measure pass. The controller then classifies Stage 4
in republish mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli import _cli_embed_stage as embed_stage_module
from phenotypic._cli._cli_staged_controller import _classify_stage4, _dead_stage3_indices
from phenotypic._cli._cli_staged_orchestration import (
    load_staged_manifest,
    staged_manifest_republishes,
)
from phenotypic.sdk_ import JobMetadataKey, read_image_record
from tests._tiny_run import (
    build_measure_epoch_tree,
    controller_config_path,
    journal_measure_failure,
    pipeline_sha,
    read_job_metadata,
    run_cli,
    stores_of,
    with_stage4_dino_objects,
)

GPU_SLURM = ["--gpu-slurm", "slurm_partition=exfab", "--gpu-slurm", "slurm_account=exfab"]
#: The run preflight imports-checks torch and asks the scheduler; neither is this file's subject.
SKIP = ["--skip-validation"]


def test_a_slurm_measure_with_gpu_slurm_uses_the_controller(tmp_path, finished_tiny_run, submissions, monkeypatch):
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    pipeline = with_stage4_dino_objects(tmp_path)
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True, pipeline=pipeline, extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert submissions == ["staged"]
    config = json.loads(controller_config_path(finished_tiny_run).read_text(encoding="utf-8"))
    assert config["phases"] == ["stage3", "stage4"] and config["stage3_role"] == "measure"
    assert config["pipeline_sha256"] == pipeline_sha(pipeline)
    assert "republish" not in config and staged_manifest_republishes(Path(config["manifest_path"]))
    metadata = read_job_metadata(finished_tiny_run)
    assert metadata["run_mode"] == "measure" and metadata[JobMetadataKey.EMBED_STAGE] == "gpu"
    # The measure call's initiation, which measure workers read (figures spec §1a).
    for key in (JobMetadataKey.FIGURES_RUN_DATE, JobMetadataKey.INITIATED_AT_UTC, JobMetadataKey.INITIATED_PID):
        assert metadata.get(key) is not None
    assert metadata[JobMetadataKey.EMBED_REVISIONS] == {"facebook/dinov2-base": "c" * 40}
    # One entry per store, the store itself, under its measure work id.
    entries = load_staged_manifest(Path(config["manifest_path"]))
    assert sorted(Path(entry.input_path) for entry in entries) == stores_of(finished_tiny_run)
    stage3 = Path(config["stage3_scripts"][0]).read_text(encoding="utf-8")
    assert "--stage3-role measure" in stage3


def test_a_slurm_measure_without_gpu_slurm_stays_inline(tmp_path, finished_tiny_run, submissions):
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=SKIP)
    assert result.exit_code == 4, result.output
    assert submissions == ["ordinary"] and not controller_config_path(finished_tiny_run).exists()


@pytest.fixture
def measure_epoch_tree(tmp_path, finished_tiny_run):
    return build_measure_epoch_tree(tmp_path, finished_tiny_run)


def test_classify_stage4_in_republish_mode(measure_epoch_tree):
    """Measured but not republished: retryable; republished under this epoch: done; journalled: terminal."""
    tree = measure_epoch_tree
    assert len(tree.entries) == 2
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    assert [e.stem for e in retryable] == [e.stem for e in tree.entries] and not terminal
    tree.republish(0)
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    assert [e.stem for e in retryable] == [tree.entries[1].stem] and not terminal
    journal_measure_failure(tree, image=1)
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    assert not retryable and [e.stem for e in terminal] == [tree.entries[1].stem]


def test_a_record_republished_under_another_epoch_is_still_pending(measure_epoch_tree):
    """Done means *this pass* republished: a record that verifies under another epoch does not count."""
    tree = measure_epoch_tree
    tree.republish(0)
    other = {**tree.config, "epoch": "another-epoch"}
    retryable, _terminal = _classify_stage4(other, tree.entries)
    assert len(retryable) == 2


def _status(tree, index, *, epoch=None):
    from phenotypic._cli._cli_staged_resume import stage4_status

    entry = tree.entries[index]
    return stage4_status(tree.out, entry.dataset, entry.stem, entry.work_id, republish=True,
                         pipeline_sha256=tree.sha, epoch=tree.epoch if epoch is None else epoch)


def test_a_same_pipeline_pass_reads_done_only_once_this_generation_republished(tmp_path, finished_tiny_run):
    """The predicate's first half (lead ruling on the C17 finding): the epoch, not verification, says done.

    A same-pipeline measure core over a root last written by a table
    rewrite (here an earlier measure; in the field a Stage-4 commit) leaves
    the root byte-identical, so the record, stamped with an earlier
    generation, still verifies and certifies the measure pipeline's own
    folder. It must read pending, or no Stage-4 round re-embeds it; once
    this generation republished it, done, so no round re-embeds it twice.
    """
    from phenotypic._cli._cli_completion import valid_image_success

    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, prior_pass=True,
                                    pipeline=finished_tiny_run.parent / "tiny_run.pht-pipe.json")
    entry = tree.entries[0]
    forward_record = read_image_record(tree.out, entry.dataset, entry.stem)
    assert valid_image_success(tree.out, dataset=entry.dataset, image_stem=entry.stem,
                               work_id=forward_record["work_id"])          # still verifies: the finding
    assert forward_record["lifecycle_epoch"] != tree.epoch
    assert [_status(tree, i) for i in range(2)] == ["pending", "pending"]
    tree.republish(0)
    assert [_status(tree, i) for i in range(2)] == ["done", "pending"]
    assert _status(tree, 0, epoch="an-earlier-generation") == "pending"
    retryable, _terminal = _classify_stage4(tree.config, tree.entries)
    assert [e.stem for e in retryable] == [tree.entries[1].stem]


def test_a_record_of_this_generation_over_another_folder_is_pending(measure_epoch_tree):
    """The predicate's second half: stamped with this epoch, verifying, but certifying the forward folder."""
    from phenotypic.sdk_._table_runs import list_table_runs

    tree = measure_epoch_tree
    forward = [run for run in list_table_runs(tree.store(0)) if run.pipeline_sha256 != tree.sha]
    assert len(forward) == 1
    tree.republish(0, run=forward[0])
    assert _status(tree, 0) == "pending"
    tree.republish(0)
    assert _status(tree, 0) == "done"


def test_a_forward_manifest_classifies_in_the_forward_sense(measure_epoch_tree):
    """The flag is read from the manifest: without it the same tree is classified forward."""
    from phenotypic._cli._cli_staged_orchestration import write_staged_manifest

    tree = measure_epoch_tree
    write_staged_manifest(tree.manifest, tree.entries, republish=False)
    retryable, terminal = _classify_stage4(tree.config, tree.entries)
    # Forward: no Stage-3 marker for these images, so nothing is pending.
    assert not retryable and len(terminal) == 2


def test_a_killed_measure_step_is_resubmitted(tmp_path, finished_tiny_run):
    """U15 for the measure role: a record and no run folder of the measure pipeline is a dead task."""
    tree = build_measure_epoch_tree(tmp_path, finished_tiny_run, measure=False)
    assert _dead_stage3_indices(tree.config, tree.entries) == [0, 1]
    journal_measure_failure(tree, image=0)
    assert _dead_stage3_indices(tree.config, tree.entries) == [1]


def test_a_measured_store_is_not_a_dead_task(measure_epoch_tree):
    assert _dead_stage3_indices(measure_epoch_tree.config, measure_epoch_tree.entries) == []

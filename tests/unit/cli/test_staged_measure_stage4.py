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
    _mark_measured(tree, "another-epoch")        # that pass measured both, too (U18)
    retryable, _terminal = _classify_stage4(other, tree.entries)
    assert len(retryable) == 2


def _mark_measured(tree, epoch):
    """Every entry's measure-done marker under *epoch* (U18), as that pass's measure steps write it."""
    from phenotypic._cli._cli_staged_resume import write_measure_done_marker

    for entry in tree.entries:
        write_measure_done_marker(tree.out, entry.dataset, entry.stem, epoch=epoch, work_id=entry.work_id)


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
    _mark_measured(tree, "an-earlier-generation")
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


# ---------------------------------------------------------------------------
# P6 deep review: U17 on SLURM, M1, M6, M11
# ---------------------------------------------------------------------------


def _record_forward_revisions(out, revisions):
    """Write *revisions* into the finished run's ``processing_state.json`` as a Stage-4 run records them (U5)."""
    from phenotypic.sdk_ import resolve_processing_state_path

    path = resolve_processing_state_path(out)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["config"]["embed_revisions"] = revisions
    path.write_text(json.dumps(state), encoding="utf-8")


def test_a_slurm_measure_reuses_the_runs_recorded_commit(tmp_path, finished_tiny_run, submissions, monkeypatch):
    """U17: the measure submission records the forward run's commit, never the hub's current one."""
    _record_forward_revisions(finished_tiny_run, {"facebook/dinov2-base": "c" * 40})
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("re-resolved the hub"))
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert read_job_metadata(finished_tiny_run)[JobMetadataKey.EMBED_REVISIONS] == {
        "facebook/dinov2-base": "c" * 40}


def test_a_slurm_measure_resolves_only_a_model_the_run_never_recorded(
    tmp_path, finished_tiny_run, submissions, monkeypatch
):
    _record_forward_revisions(finished_tiny_run, {"facebook/dinov2-base": None})        # recorded offline
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "d" * 40)
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert read_job_metadata(finished_tiny_run)[JobMetadataKey.EMBED_REVISIONS] == {
        "facebook/dinov2-base": "d" * 40}


def test_a_measure_pipeline_naming_another_model_resolves_it(
    tmp_path, finished_tiny_run, submissions, monkeypatch
):
    """U17: a complete record of other models is not a measure's whole answer; its own model is resolved."""
    _record_forward_revisions(finished_tiny_run, {"facebook/dinov2-small": "c" * 40})
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "d" * 40)
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert read_job_metadata(finished_tiny_run)[JobMetadataKey.EMBED_REVISIONS] == {
        "facebook/dinov2-base": "d" * 40}


def test_an_explicit_revision_is_never_overridden_by_the_recorded_one(
    tmp_path, finished_tiny_run, submissions, monkeypatch
):
    """U17: a pinned embedder is not unpinned, so the recorded commit never reaches it."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize
    from tests._tiny_run import _write_pipeline_file

    _record_forward_revisions(finished_tiny_run, {"facebook/dinov2-base": "c" * 40})
    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: pytest.fail("asked the hub"))
    pinned = _write_pipeline_file(tmp_path, "pinned_dino", ImagePipeline(
        name="pinned", ops=[OtsuDetector()],
        meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(revision="e" * 40)},
    ))
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True, pipeline=pinned,
                     extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert read_job_metadata(finished_tiny_run)[JobMetadataKey.EMBED_REVISIONS] == {}


def test_a_measure_submission_keeps_the_forward_completion_marker(
    tmp_path, finished_tiny_run, submissions, monkeypatch
):
    """M11: the forward run's staged-completion marker is not a measure pass's to delete."""
    from phenotypic._cli._cli_staged_orchestration import staged_completion_path

    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    marker = staged_completion_path(finished_tiny_run)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"epoch": "the-forward-epoch"}), encoding="utf-8")
    before = marker.read_bytes()
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert marker.read_bytes() == before


def test_a_measure_submission_clears_an_earlier_passs_markers(
    tmp_path, finished_tiny_run, submissions, monkeypatch
):
    """U18: markers go with their pass; the next submission removes the old ones before its epoch exists."""
    from phenotypic._cli._cli_staged_resume import measure_done_marker_path, write_measure_done_marker

    monkeypatch.setattr(embed_stage_module, "_hub_commit", lambda hf_id: "c" * 40)
    write_measure_done_marker(finished_tiny_run, "ds", "plate", epoch="an-old-pass", work_id="w")
    assert measure_done_marker_path(finished_tiny_run, "an-old-pass", "ds", "plate").is_file()
    result = run_cli(None, finished_tiny_run, mode="measure", slurm=True,
                     pipeline=with_stage4_dino_objects(tmp_path), extra=[*GPU_SLURM, *SKIP])
    assert result.exit_code == 4, result.output
    assert not measure_done_marker_path(finished_tiny_run, "an-old-pass", "ds", "plate").exists()


def _plant_signal(out, entry):
    from phenotypic._cli._cli_stage4_signal import stage4_signal_path

    path = stage4_signal_path(out, entry.dataset, "dinov2-vitb14/rgb/objects", entry.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"a leftover signal")
    return path


def test_the_republish_sweep_removes_only_what_this_pass_republished(measure_epoch_tree):
    """M1: an image republished and then killed before its signals were consumed loses them; a pending one keeps them."""
    from phenotypic._cli._cli_stage4_worker import sweep_republished_stage4_signals

    tree = measure_epoch_tree
    done, pending = (_plant_signal(tree.out, entry) for entry in tree.entries)
    tree.republish(0)
    swept = sweep_republished_stage4_signals(tree.out, tree.entries, pipeline_sha256=tree.sha, epoch=tree.epoch)
    assert swept == 1 and not done.exists() and pending.exists()


def test_the_controller_sweeps_in_republish_mode(measure_epoch_tree, monkeypatch):
    """M1: the controller's Stage-4 advance runs the republish sweep, never the forward one."""
    from phenotypic._cli import _cli_stage4_worker, _cli_staged_controller

    tree = measure_epoch_tree
    calls = []
    monkeypatch.setattr(_cli_stage4_worker, "sweep_run_orphan_stage4_signals",
                        lambda *a, **k: pytest.fail("the forward sweep ran in republish mode"))
    monkeypatch.setattr(_cli_stage4_worker, "sweep_republished_stage4_signals",
                        lambda *a, **k: calls.append(k) or 0)
    monkeypatch.setattr(_cli_staged_controller, "_submit_stage4_round", lambda *a, **k: None)
    _cli_staged_controller._advance_stage4(
        {**tree.config, "stage4_worklist": str(tree.out / "w.json"), "n_shards": 1},
        {"phase": "stage4"}, "successor", "current",
    )
    assert calls and calls[0]["epoch"] == tree.epoch and calls[0]["pipeline_sha256"] == tree.sha


def test_a_measure_step_refuses_a_pipeline_edited_since_submission(measure_epoch_tree):
    """M6: the step takes the submitter's sha; an edited file fails the task, journalling and marking nothing."""
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures
    from phenotypic._cli._cli_process_single import MeasurePipelineChangedError
    from phenotypic._cli._cli_staged_resume import measure_done_marker_path
    from phenotypic._cli._cli_staged_slurm_worker import run_measure_step

    tree = measure_epoch_tree
    entry = tree.entries[0]
    marker = measure_done_marker_path(tree.out, tree.epoch, entry.dataset, entry.stem)
    marker.unlink()
    tree.pipeline.write_text(tree.pipeline.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(MeasurePipelineChangedError, match="changed since this pass began"):
        run_measure_step(tree.pipeline, tree.out, "GridImage", tree.entries, 0, epoch=tree.epoch,
                         durable_writes=False)
    assert read_terminal_failures(tree.out) == [] and not marker.exists()


def test_a_stage4_shard_refuses_a_pipeline_edited_since_submission(measure_epoch_tree):
    """M6: Stage 4 checks the recorded sha before any load: node state, nothing journalled."""
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_stage4_worker import Stage4ShardError, run_stage4

    tree = measure_epoch_tree
    tree.pipeline.write_text(tree.pipeline.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Stage4ShardError, match="changed since this pass began"):
        run_stage4(
            tree.pipeline, tree.out, tree.entries, image_type="GridImage", epoch=tree.epoch,
            republish=True, republish_epoch=tree.epoch, pipeline_sha256=tree.sha, num_workers=0,
            output_manager=OutputManager.from_config(tree.out, ".tiff", save_overlays=False),
        )

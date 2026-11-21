"""Stages 1-3 defer embeddings and the record under ``--embed-stage gpu`` (#282 P4 T2). [torch]

One predicate, one rule (plan review C1): under deferral an image whose
record holds ``stages.stage3``, whose store matches its work id and which has
a measurement table has finished Stages 1-3 (``stages_1_to_3_done``). Every
shortcut that used to publish it returns early and publishes nothing, every
stage loop skips it, and every core that re-runs it drops that state first.
Completion is record-based (plan review C2).

Stage 4 itself is P4 T3: here it is stubbed out (``no_stage4``), so every image
of a ``--embed-stage gpu`` run ends waiting on it. Run in the foundation venv
(``TPT``); in the shared env the module skips.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from phenotypic._cli._cli_completion import valid_image_success  # noqa: E402
from phenotypic._cli._cli_staged_orchestration import staged_completion_path  # noqa: E402
from phenotypic._cli._cli_staged_resume import stages_1_to_3_done  # noqa: E402
from phenotypic._cli._cli_update_state import parse_event_line  # noqa: E402
from phenotypic.sdk_ import event_log_path, journal_measurement_table  # noqa: E402
from phenotypic.sdk_._image_record import read_image_record  # noqa: E402
from tests._tiny_run import (  # noqa: E402
    count_calls,
    dataset_and_stem,
    invalidate_store,
    publish_waiting_records,
    record_embed_stage_in_job_metadata,
    run_cli,
    run_entry,
    run_items,
    signal_files,
    store_root_bytes,
    with_stage4_dino_objects,
    zarr_store_path_of,
)

GPU = ["--embed-stage", "gpu"]
_RUN_KINDS = pytest.mark.parametrize("fake_gpu", [True, False], ids=["staged", "single-pass"])


def _stores(out):
    stores = sorted(out.glob("results/*/zarr/*.ome.zarr"))
    assert len(stores) == 2, stores  # tiny_inputs: two plates
    return stores


def read_events(out):
    """Every event of the run's log, parsed."""
    return [
        parse_event_line(line)
        for line in event_log_path(out).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@_RUN_KINDS
def test_deferral_writes_no_record_and_no_embeddings(tmp_path, tiny_inputs, stage4_cpu, no_stage4, fake_gpu):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu), extra=GPU)
    for store in _stores(out):
        record = read_image_record(out, *dataset_and_stem(store))
        assert record is not None and "stage3" in record["stages"], record
        assert not record.get("artifacts"), record          # no record: Stage 4 publishes it
        assert "embeddings" not in run_entry(store)          # Stage 4 commits them
        assert journal_measurement_table(store) is not None  # the measurements are final
    for item in run_items(out):
        assert stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)
        assert not valid_image_success(out, dataset=item.dataset, image_stem=item.stem, work_id=item.work_id)


def test_stages_1_to_3_done_needs_all_three_conjuncts(stage4_pending_with_signals, monkeypatch):
    """The marker, a store bound to the work id, and a table: drop any one and it is not done."""
    from phenotypic._cli import _cli_staged_resume
    from phenotypic._cli._cli_staged_resume import remove_stage3_completion_marker

    out, item = stage4_pending_with_signals
    other = stage4_pending_with_signals.items[1]
    assert stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)
    assert not stages_1_to_3_done(out, item.dataset, item.stem, "another-work-id")
    with monkeypatch.context() as patched:
        patched.setattr(_cli_staged_resume, "journal_measurement_table", lambda store: None)
        assert not stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)
    invalidate_store(out, item)
    assert not stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)
    remove_stage3_completion_marker(out, other.dataset, other.stem)
    assert not stages_1_to_3_done(out, other.dataset, other.stem, other.work_id)


@_RUN_KINDS
def test_a_waiting_run_is_not_staged_complete(tmp_path, tiny_inputs, stage4_cpu, no_stage4, fake_gpu):
    """C2: Stage-3 markers are not completion; a record-less run never reads complete."""
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu), extra=GPU)
    assert result.exit_code == 1, result.output
    assert "STAGED FINALIZATION FAILED" in result.output, result.output
    assert not staged_completion_path(out).is_file()


@_RUN_KINDS
def test_a_published_deferral_run_finalizes_and_is_staged_complete(
    tmp_path, tiny_inputs, stage4_cpu, no_stage4, fake_gpu
):
    """The positive control of the predicate above; the single-pass case pins re-review item 1.

    Once every record is published (here by hand, standing in for Stage 4),
    the same command finalizes. A no-detector run reaches the finalization's
    Stage-2 reconcile, which must not ask ``staged_detector_slot`` for a slot
    its pipeline does not have.
    """
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert publish_waiting_records(out) == 2
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code == 0, result.output
    assert staged_completion_path(out).is_file()
    assert (out / "deliverables" / "master_measurements.parquet").is_file()


def test_a_waiting_image_is_neither_redetected_nor_remeasured(tmp_path, tiny_inputs, stage4_cpu, no_stage4, monkeypatch):
    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path, fake_gpu=True)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    roots = store_root_bytes(out)
    assert len(roots) == 2
    calls = count_calls(
        monkeypatch,
        ["stage1_preprocess_core", "stage2_detect_core", "stage3_merge_measure_core", "publish_image_success"],
    )
    result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    assert result.exit_code != 2, result.output
    assert calls == {
        "stage1_preprocess_core": 0,
        "stage2_detect_core": 0,
        "stage3_merge_measure_core": 0,
        "publish_image_success": 0,
    }
    assert store_root_bytes(out) == roots


@pytest.mark.parametrize(
    "embed_stage, resume",
    [("gpu", True), ("gpu", False), ("inline", True)],
    ids=["deferring-resume", "deferring-fresh", "inline-control"],
)
def test_the_slurm_stage3_step_returns_early_for_a_waiting_image(
    stage4_slurm_waiting, monkeypatch, embed_stage, resume
):
    """C1: the SLURM Stage-3 step publishes nothing and never ``SystemExit``\\s.

    Deferral is read from ``job_metadata.json``. The ``inline`` control is
    the old shortcut, which publishes the record: it shows the test can fail.
    """
    from phenotypic._cli._cli_staged_slurm_worker import run_stage3_step

    out = stage4_slurm_waiting.out
    record_embed_stage_in_job_metadata(out, embed_stage)
    item = stage4_slurm_waiting.items[0]
    roots = store_root_bytes(out)
    calls = count_calls(monkeypatch, ["stage3_merge_measure_core"])
    run_stage3_step(**stage4_slurm_waiting.stage3_kwargs(0, resume=resume))  # no SystemExit
    published = valid_image_success(out, dataset=item.dataset, image_stem=item.stem, work_id=item.work_id)
    if embed_stage == "gpu":
        assert not published
        assert calls == {"stage3_merge_measure_core": 0}
        assert store_root_bytes(out) == roots
    else:
        assert published


@pytest.mark.parametrize("embed_stage", ["gpu", "inline"], ids=["deferring", "inline-control"])
def test_the_slurm_stage2_shard_skips_a_waiting_image(stage4_slurm_waiting, monkeypatch, embed_stage):
    """C1: a waiting image's Stage-2 signal is consumed; a shard must not re-detect it."""
    from phenotypic._cli._cli_staged_slurm_worker import run_stage2_shard

    out = stage4_slurm_waiting.out
    record_embed_stage_in_job_metadata(out, embed_stage)
    calls = count_calls(monkeypatch, ["stage2_detect_core"])
    run_stage2_shard(
        stage4_slurm_waiting.pipeline, out, "Image", stage4_slurm_waiting.manifest(), 0, 1, resume=True
    )
    assert calls == {"stage2_detect_core": 0 if embed_stage == "gpu" else 2}


def test_a_core_rerun_deletes_the_images_signals(stage4_pending_with_signals):
    out, item = stage4_pending_with_signals
    assert signal_files(out, item)
    invalidate_store(out, item)                                    # forces the core to re-run it
    stage4_pending_with_signals.rerun()
    assert not signal_files(out, item)
    assert stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)   # it re-ran to the end


def test_a_core_rerun_drops_the_old_stage3_first(stage4_pending_with_signals, monkeypatch):
    """A crash inside a re-running core leaves an image that is not done.

    The crash comes after the core's final save, the one write that makes
    the store match the work id with a table again, so only the old
    ``stages.stage3`` could still make the image read done.
    """
    from phenotypic.plotting._pipeline import PlotCoordinator

    out, item = stage4_pending_with_signals
    invalidate_store(out, item)
    original = PlotCoordinator.publish_store_figures

    def _crash_after_the_final_save(self, saved_store, *args, **kwargs):
        if saved_store.name.startswith(item.stem + "."):
            raise RuntimeError("crash after the final save")
        return original(self, saved_store, *args, **kwargs)

    monkeypatch.setattr(PlotCoordinator, "publish_store_figures", _crash_after_the_final_save)
    stage4_pending_with_signals.rerun()
    store = zarr_store_path_of(out, item)
    assert journal_measurement_table(store) is not None                     # the final save landed
    assert not stages_1_to_3_done(out, item.dataset, item.stem, item.work_id)
    assert not signal_files(out, item)


def test_a_no_detector_local_run_finalizes(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path, fake_gpu=False), extra=GPU)
    assert result.exit_code == 0, result.output                 # no staged_detector_slot refusal
    assert (out / "deliverables" / "embeddings" / "manifest.json").is_file()


def test_inline_events_stay_untagged(tmp_path, tiny_inputs, stage4_cpu):
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path))       # local auto: inline
    assert result.exit_code == 0, result.output
    assert {event.stage for event in read_events(out)} == {None}


def test_a_deferring_single_pass_core_tags_its_events_stage3(tmp_path, tiny_inputs, stage4_cpu, no_stage4):
    from phenotypic._cli._cli_state_management import load_processing_state

    out = tmp_path / "out"
    run_cli(tiny_inputs, out, pipeline=with_stage4_dino_objects(tmp_path), extra=GPU)
    events = read_events(out)
    assert events and {event.stage for event in events} == {"stage3"}
    state = load_processing_state(out)
    assert state is not None and not state.datasets["ds"].completed        # Stage 3 is not completion


def test_a_gpu_run_keeps_its_embed_stage_across_continuations(tmp_path, tiny_inputs, stage4_cpu, no_stage4):
    """The resume rewrite of ``state.config`` keeps ``embed_stage``.

    Before it did, the first continuation dropped the key, and the second
    read ``"inline"``, so R3 refused the run's own command (exit 2).
    """
    import json

    from phenotypic.sdk_ import processing_state_path

    out, pipe = tmp_path / "out", with_stage4_dino_objects(tmp_path)
    run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
    for _ in range(2):
        result = run_cli(tiny_inputs, out, pipeline=pipe, extra=GPU)
        assert result.exit_code != 2, result.output
        config = json.loads(processing_state_path(out).read_text(encoding="utf-8"))["config"]
        assert config["embed_stage"] == "gpu"


def test_process_mode_ignores_embed_stage(tmp_path, tiny_inputs, stage4_cpu):
    """H5: process mode embeds inline. Only its grid embedder runs (objects are skipped there)."""
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, mode="process", pipeline=with_stage4_dino_objects(tmp_path, grid=True),
                     extra=["--layer", "rgb", "--process-format", "zarr"])
    assert result.exit_code == 0, result.output
    stores = sorted(p for p in out.rglob("*.ome.zarr") if ".phenotypic" not in p.parts)
    assert len(stores) == 2, stores
    for store in stores:
        entry = run_entry(store)["embeddings"]["dinov2-vitb14/rgb/grid"]
        assert entry["compute"]["stage"] == "inline"
    assert {event.stage for event in read_events(out)} <= {None}

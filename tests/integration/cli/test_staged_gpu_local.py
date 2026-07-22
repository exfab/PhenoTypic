"""End-to-end tests for the local staged GPU engine (Spec 1, Plan 2)."""

import phenotypic
import numpy as np
import pytest
from click.testing import CliRunner
from datetime import datetime

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic._cli._cli_sidecar import sidecar_exists, write_sidecar
from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    append_stage2_terminal_failure,
    initialize_orchestration,
)
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy
from phenotypic._cli._cli_staged_resume import (
    build_staged_resume_plan,
    reconcile_stage3_publications,
    stage3_completion_exists,
    stage3_completion_marker_path,
)
from phenotypic._cli._cli_state_management import (
    load_processing_state,
    save_processing_state,
)
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset, ExecutionConfig, ExecutionResults
from phenotypic._cli._cli_update_state import (
    aggregate_stage_state_from_events,
    parse_event_line,
)
from phenotypic.sdk_ import dataset_hdf_dir, dataset_overlays_dir, event_log_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from phenotypic.phenotypicCLI import phenotypic_cli


def _config(out, pipe_path, resume=False):
    return ExecutionConfig(
        pipeline_json=pipe_path,
        input_path=out,
        output_dir=out,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.5,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=resume,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
    )


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    """Make ``FakeGpuDetector`` resolvable by ``ImagePipeline.from_json``.

    The staged strategy loads its pipeline from ``pipeline.json``, and the
    deserializer resolves op classes from the ``phenotypic`` namespace. The
    fake is a test-only class, so register it for the duration of each test
    (``monkeypatch`` auto-reverts — no global namespace pollution).
    """
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )


def _write_image(tmp_path):
    img = load_synth_yeast_plate()
    p = tmp_path / "img.tiff"
    img.rgb.imsave(filepath=p)
    return p


def test_three_stage_cores_end_to_end(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
    )
    pipe_path = out / "pipeline.json"
    pipe_path.parent.mkdir(parents=True)
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path))
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])

    # Stage 1: preprocess -> HDF
    stage1_preprocess_core(
        plan, image_path, "ds", "img", out, om, image_type="Image"
    )
    assert (dataset_hdf_dir(out, "ds") / "img.h5").is_file()

    # Stage 2: resident detector -> sidecar
    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(plan.gpu_detector, out, "ds", "img")
    assert sidecar_exists(out, "ds", "img")

    # Stage 3: merge + measure -> parquet, re-save HDF, delete sidecar
    stage3_merge_measure_core(plan, out, "ds", "img", om, image_type="Image")
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert not sidecar_exists(out, "ds", "img")  # mandatory cleanup


def test_staged_strategy_runs_all_stages(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=True)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])

    strat = StagedGpuStrategy(_config(out, pipe_path), om)
    results = strat.execute([Dataset("ds", [image_path], tmp_path, out)], out)

    assert results.total_completed == 1
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert (dataset_overlays_dir(out, "ds") / "img.png").is_file()
    assert not sidecar_exists(out, "ds", "img")


def test_staged_strategy_resume_backfills_missing_overlay_without_gpu(tmp_path, monkeypatch):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=True)
    datasets = [Dataset("ds", [image_path], tmp_path, out)]
    om.create_structure(datasets)
    StagedGpuStrategy(_config(out, pipe_path), om).execute(datasets, out)
    overlay = dataset_overlays_dir(out, "ds") / "img.png"
    overlay.unlink()
    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    result = StagedGpuStrategy(
        _config(out, pipe_path, resume=True), om
    ).execute(datasets, out)

    assert result.total_completed == 1
    assert overlay.is_file()


def test_staged_strategy_resumes_skipping_done_stages(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    ds = [Dataset("ds", [image_path], tmp_path, out)]

    StagedGpuStrategy(_config(out, pipe_path), om).execute(ds, out)
    parquet = out / "results" / "ds" / "measurements" / "img.parquet"
    mtime = parquet.stat().st_mtime_ns

    # second run with resume=True: every stage skips -> parquet untouched and
    # no orphan sidecar is recreated.
    StagedGpuStrategy(_config(out, pipe_path, resume=True), om).execute(
        ds, out
    )
    assert parquet.stat().st_mtime_ns == mtime
    assert not sidecar_exists(out, "ds", "img")


def test_process_objmap_runs_stages_1_2_then_exports(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])

    cfg = _config(out, pipe_path)
    cfg.process_only_layer = "objmap"
    StagedGpuStrategy(cfg, om).execute(
        [Dataset("ds", [image_path], tmp_path, out)], out
    )

    # objmap layer exported (mirrored); no measurement parquet; sidecar cleaned
    expected = process_only_output_path(out, image_path, out, "objmap")
    assert expected.is_file()
    assert not (
        out / "results" / "ds" / "measurements" / "img.parquet"
    ).exists()
    assert not sidecar_exists(out, "ds", "img")


def test_stage_tagged_events_emitted(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    StagedGpuStrategy(_config(out, pipe_path), om).execute(
        [Dataset("ds", [image_path], tmp_path, out)], out
    )

    per_stage = aggregate_stage_state_from_events(event_log_path(out))
    for stage in ("stage1", "stage2", "stage3"):
        assert "img.tiff" in per_stage["ds"][stage].completed


def _count_stage_started(log_text, stage):
    n = 0
    for line in log_text.splitlines():
        if not line.strip():
            continue
        ev = parse_event_line(line)
        if ev.status == "started" and ev.stage == stage:
            n += 1
    return n


def test_process_objmap_resume_skips_gpu_stage2(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    ds = [Dataset("ds", [image_path], tmp_path, out)]

    cfg1 = _config(out, pipe_path)
    cfg1.process_only_layer = "objmap"
    StagedGpuStrategy(cfg1, om).execute(ds, out)
    log = event_log_path(out)
    stage2_before = _count_stage_started(
        log.read_text(encoding="utf-8"), "stage2"
    )
    assert stage2_before == 1

    # resume: the exported objmap PNG is the durable Stage-2 done-marker, so the
    # GPU stage must NOT re-run (no new stage2 'started' event).
    cfg2 = _config(out, pipe_path, resume=True)
    cfg2.process_only_layer = "objmap"
    StagedGpuStrategy(cfg2, om).execute(ds, out)
    stage2_after = _count_stage_started(
        log.read_text(encoding="utf-8"), "stage2"
    )
    assert stage2_after == stage2_before  # no re-detection on resume


def test_stage2_shard_worker_processes_its_shard(tmp_path):
    # Stage 1 first (reuse the core), then run the Stage-2 shard worker over shard 0.
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    stage1_preprocess_core(
        split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path)),
        image_path,
        "ds",
        "img",
        out,
        om,
        image_type="Image",
    )

    from phenotypic._cli._cli_staged_slurm_worker import run_stage2_shard

    run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )
    assert sidecar_exists(out, "ds", "img")


def test_stage2_publication_is_fenced_for_stale_epoch(tmp_path):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    plan = split_pipeline_at_gpu(pipe)
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    stage1_preprocess_core(
        plan, image_path, "ds", "img", out, om, image_type="Image"
    )
    plan.gpu_detector._ensure_model_loaded()

    with pytest.raises(RuntimeError, match="stale"):
        stage2_detect_core(
            plan.gpu_detector,
            out,
            "ds",
            "img",
            "Image",
            active_check=lambda: (_ for _ in ()).throw(
                RuntimeError("stale epoch")
            ),
        )
    assert not sidecar_exists(out, "ds", "img")


def test_plain_resume_reuses_stage1_hdf_after_stage2_failure(
    tmp_path, monkeypatch
):
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    datasets = [Dataset("ds", [image_path], tmp_path, out)]

    original_infer = FakeGpuDetector._infer_one
    monkeypatch.setattr(
        FakeGpuDetector,
        "_infer_one",
        lambda self, sample: (_ for _ in ()).throw(RuntimeError("GPU failed")),
    )
    failed = StagedGpuStrategy(_config(out, pipe_path), om).execute(
        datasets, out
    )
    assert failed.total_failed == 1
    assert (dataset_hdf_dir(out, "ds") / "img.h5").is_file()
    monkeypatch.setattr(FakeGpuDetector, "_infer_one", original_infer)
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_strategy.stage1_preprocess_core",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage 1 reran")
        ),
    )
    resumed = StagedGpuStrategy(
        _config(out, pipe_path, resume=True), om
    ).execute(datasets, out)

    assert resumed.total_completed == 1
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert stage3_completion_exists(out, "ds", "img")


def test_cli_plain_resume_includes_recorded_stage2_failure(
    tmp_path, monkeypatch
):
    images = tmp_path / "images"
    images.mkdir()
    image_path = _write_image(images)
    second_image = images / "img2.tiff"
    second_image.write_bytes(image_path.read_bytes())
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    args = [
        "--pipeline",
        str(pipe_path),
        "--input",
        str(images),
        "--output",
        str(out),
        "--image-type",
        "Image",
        "--force-local",
        "--skip-validation",
        "--njobs",
        "1",
    ]
    original_infer = FakeGpuDetector._infer_one
    infer_calls = 0

    def fail_second_inference(self, sample):
        nonlocal infer_calls
        infer_calls += 1
        if infer_calls == 2:
            raise RuntimeError("GPU failed")
        return original_infer(self, sample)

    monkeypatch.setattr(FakeGpuDetector, "_infer_one", fail_second_inference)
    first = CliRunner().invoke(phenotypic_cli, args)
    assert first.exit_code == 1
    from phenotypic._cli._cli_staged_orchestration import staged_completion_path

    assert not staged_completion_path(out).is_file()
    assert (dataset_hdf_dir(out, images.name) / f"{image_path.stem}.h5").is_file()

    monkeypatch.setattr(FakeGpuDetector, "_infer_one", original_infer)
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_strategy.stage1_preprocess_core",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage 1 reran")
        ),
    )
    aggregated_images: list[str] = []
    original_aggregate = OutputManager.aggregate_master_csv

    def capture_full_inventory(self, datasets, *args, **kwargs):
        aggregated_images.extend(
            image.name for dataset in datasets for image in dataset.images
        )
        return original_aggregate(self, datasets, *args, **kwargs)

    monkeypatch.setattr(
        OutputManager, "aggregate_master_csv", capture_full_inventory
    )
    resumed = CliRunner().invoke(phenotypic_cli, [*args, "--resume"])

    assert resumed.exit_code == 0, resumed.output
    assert "Resuming staged GPU processing" in resumed.output
    assert "Stage 2 1" in resumed.output
    assert sorted(aggregated_images) == ["img.tiff", "img2.tiff"]


@pytest.mark.parametrize("legacy_markerless", [False, True])
def test_cli_resume_backfills_completed_overlay_before_early_exit(
    tmp_path, monkeypatch, legacy_markerless
):
    images = tmp_path / "images"
    images.mkdir()
    image_path = _write_image(images)
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    args = [
        "--pipeline",
        str(pipe_path),
        "--input",
        str(images),
        "--output",
        str(out),
        "--image-type",
        "Image",
        "--force-local",
        "--skip-validation",
        "--njobs",
        "1",
        "--overlay-alpha",
        "0.65",
    ]
    first = CliRunner().invoke(phenotypic_cli, args)
    assert first.exit_code == 0, first.output
    overlay = dataset_overlays_dir(out, images.name) / "img.png"
    overlay.unlink()
    if legacy_markerless:
        stage3_completion_marker_path(
            out, images.name, image_path.stem
        ).unlink()
        state = load_processing_state(out)
        assert state is not None
        state.config["staged_stage3_markers"] = False
        save_processing_state(state, out)
    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    observed_alpha = []
    original_save_overlay = OutputManager.save_overlay

    def _save_overlay_with_alpha(manager, *args, **kwargs):
        observed_alpha.append(manager.overlay_alpha)
        return original_save_overlay(manager, *args, **kwargs)

    monkeypatch.setattr(OutputManager, "save_overlay", _save_overlay_with_alpha)

    resumed = CliRunner().invoke(phenotypic_cli, [*args, "--resume"])

    assert resumed.exit_code == 0, resumed.output
    assert "All images already processed" in resumed.output
    assert overlay.is_file()
    assert observed_alpha == [0.65]
    assert stage3_completion_exists(out, images.name, image_path.stem)


def test_cli_resumes_finalizer_only_when_image_stages_are_complete(
    tmp_path, monkeypatch
):
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images)
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    base_args = [
        "--pipeline",
        str(pipe_path),
        "--input",
        str(images),
        "--output",
        str(out),
        "--image-type",
        "Image",
        "--skip-validation",
        "--njobs",
        "1",
    ]
    first = CliRunner().invoke(phenotypic_cli, [*base_args, "--force-local"])
    assert first.exit_code == 0, first.output
    from phenotypic._cli._cli_staged_orchestration import staged_completion_path

    staged_completion_path(out).unlink()
    observed: dict[str, object] = {}

    class SubmittedStrategy:
        def execute(self, datasets, output_dir):
            observed["datasets"] = datasets
            now = datetime.now()
            return ExecutionResults(
                datasets={},
                total_images=1,
                total_completed=0,
                total_failed=0,
                execution_mode="slurm",
                start_time=now,
                end_time=now,
                submitted=True,
                remote_managed=True,
            )

    def fake_strategy(config, output_manager):
        observed["finalizer_only"] = config.staged_finalizer_only
        observed["phase"] = config.staged_resume_phase
        return SubmittedStrategy()

    monkeypatch.setattr(
        "phenotypic.phenotypicCLI.create_execution_strategy", fake_strategy
    )
    resumed = CliRunner().invoke(
        phenotypic_cli,
        [
            *base_args,
            "--resume",
            "--slurm",
            "slurm_partition=test",
        ],
    )

    assert resumed.exit_code == 0, resumed.output
    assert "Resuming staged GPU finalization" in resumed.output
    assert observed == {
        "finalizer_only": True,
        "phase": "stage3",
        "datasets": [],
    }


def test_cli_local_resume_republishes_missing_final_outputs(
    tmp_path, monkeypatch
):
    from phenotypic._cli._cli_readme_generator import READMEGenerator

    images = tmp_path / "images"
    images.mkdir()
    _write_image(images)
    out = tmp_path / "out"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    args = [
        "--pipeline",
        str(pipe_path),
        "--input",
        str(images),
        "--output",
        str(out),
        "--image-type",
        "Image",
        "--skip-validation",
        "--njobs",
        "1",
        "--force-local",
    ]
    original_generate = READMEGenerator.generate
    monkeypatch.setattr(
        READMEGenerator,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("README publication failed")
        ),
    )
    first = CliRunner().invoke(phenotypic_cli, args)
    assert first.exit_code == 1
    assert "STAGED FINALIZATION FAILED" in first.output
    assert "PROCESSING COMPLETE" not in first.output

    from phenotypic._cli._cli_staged_orchestration import staged_completion_path

    marker = staged_completion_path(out)
    assert not marker.is_file()
    monkeypatch.setattr(READMEGenerator, "generate", original_generate)
    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    resumed = CliRunner().invoke(phenotypic_cli, [*args, "--resume"])

    assert resumed.exit_code == 0, resumed.output
    assert "Resuming staged GPU finalization" in resumed.output
    assert marker.is_file()


def test_stage3_resume_does_not_load_gpu_model(tmp_path, monkeypatch):
    import phenotypic._cli._cli_staged_slurm_worker as worker

    out, pipe_path = _stage1_only(tmp_path)
    worker.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )
    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    image_path = tmp_path / "img.tiff"
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)

    result = StagedGpuStrategy(
        _config(out, pipe_path, resume=True), om
    ).execute([Dataset("ds", [image_path], tmp_path, out)], out)

    assert result.total_completed == 1


def test_stage3_partial_publication_keeps_sidecar_and_is_resumable(
    tmp_path, monkeypatch
):
    import phenotypic._cli._cli_staged_slurm_worker as worker

    out, pipe_path = _stage1_only(tmp_path)
    worker.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path))
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    monkeypatch.setattr(
        om,
        "save_image_hdf",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="Stage 3 HDF publication failed"):
        stage3_merge_measure_core(
            plan,
            out,
            "ds",
            "img",
            om,
            image_type="Image",
            image_name="img.tiff",
        )

    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert sidecar_exists(out, "ds", "img")
    assert not stage3_completion_exists(out, "ds", "img")
    resume_plan = build_staged_resume_plan(
        datasets=[Dataset("ds", [tmp_path / "img.tiff"], tmp_path, out)],
        output_dir=out,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )
    assert resume_plan.initial_stage == "stage3"


def test_stage3_reconciliation_cleans_sidecar_and_excludes_partial_parquet(
    tmp_path,
):
    out = tmp_path / "out"
    complete = out / "results" / "ds" / "measurements" / "done.parquet"
    partial = out / "results" / "ds" / "measurements" / "partial.parquet"
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_bytes(b"complete")
    partial.write_bytes(b"partial")
    write_sidecar(out, "ds", "done", np.zeros((2, 2)))
    from phenotypic._cli._cli_staged_resume import (
        write_stage3_completion_marker,
    )

    write_stage3_completion_marker(out, "ds", "done.tiff", "done")

    moved = reconcile_stage3_publications(
        out,
        {"ds": ["done.tiff", "partial.tiff"]},
        namespace="test",
    )

    assert moved == 1
    assert not sidecar_exists(out, "ds", "done")
    assert complete.is_file()
    assert not partial.exists()
    assert (
        out
        / ".phenotypic"
        / "progress"
        / "unpublished_stage3"
        / "test"
        / "ds"
        / "partial.parquet"
    ).is_file()


def _stage1_only(tmp_path):
    """Shared setup: write an image and run Stage 1 so a staged HDF exists."""
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])
    stage1_preprocess_core(
        split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path)),
        image_path,
        "ds",
        "img",
        out,
        om,
        image_type="Image",
    )
    return out, pipe_path


def test_slurm_stage3_worker_writes_and_backfills_overlay(tmp_path, monkeypatch):
    import phenotypic._cli._cli_staged_slurm_worker as worker

    out, pipe_path = _stage1_only(tmp_path)
    dataset_overlays_dir(out, "ds").mkdir(parents=True)
    manifest = [("ds", "img")]
    observed_alpha = []
    original_save_overlay = OutputManager.save_overlay

    def _save_overlay_with_alpha(manager, *args, **kwargs):
        observed_alpha.append(manager.overlay_alpha)
        return original_save_overlay(manager, *args, **kwargs)

    monkeypatch.setattr(OutputManager, "save_overlay", _save_overlay_with_alpha)
    worker.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=manifest,
        shard_index=0,
        n_shards=1,
    )

    worker.run_stage3_step(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=manifest,
        index=0,
        overlay_alpha=0.65,
    )

    overlay = dataset_overlays_dir(out, "ds") / "img.png"
    assert overlay.is_file()
    overlay.unlink()
    monkeypatch.setattr(
        worker,
        "split_pipeline_at_gpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pipeline reloaded")
        ),
    )

    worker.run_stage3_step(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=manifest,
        index=0,
        resume=True,
        overlay_alpha=0.65,
    )

    assert overlay.is_file()
    assert observed_alpha == [0.65, 0.65]


def test_completed_shard_exits_before_loading_model(tmp_path, monkeypatch):
    import phenotypic._cli._cli_staged_slurm_worker as W

    out, pipe_path = _stage1_only(tmp_path)
    W.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )
    assert sidecar_exists(out, "ds", "img")  # shard complete

    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    W.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[("ds", "img")],
        shard_index=0,
        n_shards=1,
    )


def test_terminal_failure_shard_exits_before_loading_model(tmp_path, monkeypatch):
    import phenotypic._cli._cli_staged_slurm_worker as W

    out, pipe_path = _stage1_only(tmp_path)
    epoch = "test-epoch"
    initialize_orchestration(
        out,
        epoch=epoch,
        mode="full",
        controller_config_path=out / "controller.json",
    )
    entry = StagedManifestEntry(
        dataset="ds",
        image_name="img.tiff",
        stem="img",
        input_path=str(tmp_path / "img.tiff"),
    )
    append_stage2_terminal_failure(
        out,
        epoch=epoch,
        round_index=0,
        entry=entry,
        error_type="DetectorError",
        error_message="deterministic failure",
    )
    monkeypatch.setattr(
        FakeGpuDetector,
        "_ensure_model_loaded",
        lambda self: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    W.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=[entry],
        shard_index=0,
        n_shards=1,
        epoch=epoch,
    )
    assert not sidecar_exists(out, "ds", "img")


def test_shard_worker_records_missing_hdf_without_requeue(tmp_path):
    import phenotypic._cli._cli_staged_slurm_worker as W

    out, pipe_path = _stage1_only(tmp_path)  # "img" has a staged HDF
    manifest = [("ds", "img"), ("ds", "ghost")]  # "ghost" has no HDF
    W.run_stage2_shard(
        pipeline_path=pipe_path,
        output_dir=out,
        image_type="Image",
        manifest=manifest,
        shard_index=0,
        n_shards=1,
    )
    assert sidecar_exists(out, "ds", "img")
    assert not sidecar_exists(out, "ds", "ghost")
    assert "staged HDF missing" in event_log_path(out).read_text(
        encoding="utf-8"
    )

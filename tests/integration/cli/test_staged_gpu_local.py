"""End-to-end tests for the local staged GPU engine (Spec 1, Plan 2)."""

import phenotypic
import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic._cli._cli_sidecar import sidecar_exists
from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    append_stage2_terminal_failure,
    initialize_orchestration,
)
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic._cli._cli_update_state import (
    aggregate_stage_state_from_events,
    parse_event_line,
)
from phenotypic.sdk_ import dataset_hdf_dir, event_log_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector


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
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", [image_path], tmp_path, out)])

    strat = StagedGpuStrategy(_config(out, pipe_path), om)
    results = strat.execute([Dataset("ds", [image_path], tmp_path, out)], out)

    assert results.total_completed == 1
    assert (out / "results" / "ds" / "measurements" / "img.parquet").is_file()
    assert not sidecar_exists(out, "ds", "img")


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

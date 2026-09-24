"""The calibration overlay in every CLI mode (figures spec §1a, §3a).

The overlay can only be drawn where ``CalibrateColorRpcc.apply()`` ran, so
full, process and staged Stage 1 draw it into this run's folder. Measure mode
and staged Stage 3 cannot draw it: they keep it when their run folder already
holds it, and otherwise list it as ``unavailable`` without touching the folder
that has it. Every run date is pinned.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    plot_failures_jsonl_path,
    plots_dir,
    zarr_store_path,
)
from phenotypic.sdk_._image_figures import (
    FigureRun,
    RunInitiation,
    read_image_figures_descriptor,
)
from tests.unit.correction._checker_frames import frozen_op, render_frame

DAY = "2026-09-22"
LATER = "2026-09-30"


def _write_inputs(root: Path, *, with_plot: bool = True, gpu: bool = False, variant: int = 0):
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    root.mkdir(parents=True, exist_ok=True)
    image = root / "in" / "plate.tiff"
    image.parent.mkdir(exist_ok=True)
    imsave(str(image), render_frame(), check_contrast=False)
    cal = frozen_op()
    if gpu:
        from tests._fakes.fake_gpu_detector import FakeGpuDetector

        detector = FakeGpuDetector(output_kind="instance", threshold=0.3)
    else:
        detector = OtsuDetector()
    meas = {"size": MeasureSize()}
    if variant:
        meas[f"size{variant}"] = MeasureSize()
    pipeline = ImagePipeline(
        ops={"cal": cal, "detect": detector},
        meas=meas,
        plots=[cal] if with_plot else [],
    )
    path = root / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return image, path


def _manager(out: Path, date: str = DAY, *, overlays: bool = False) -> OutputManager:
    return OutputManager.from_config(
        out, ".tiff", save_overlays=overlays,
        run_initiation=RunInitiation(date, f"{date}T12:00:00.000Z", 7),
    )


def _run_of(pipeline: Path, date: str = DAY) -> str:
    sha = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    return FigureRun(date=date, pipeline_sha256=sha).run_id


def _full(out: Path, pipeline: Path, image: Path) -> Path:
    from phenotypic._cli._cli_process_single import process_single_image_core

    process_single_image_core(pipeline, image, out, "ds", "Image", {}, _manager(out))
    return zarr_store_path(out, "ds", "plate")


def _measure(out: Path, store: Path, pipeline: Path, date: str) -> None:
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    process_single_store_measure_core(pipeline, store, out, "ds", "Image", _manager(out, date))


def _runs(store: Path) -> dict:
    return read_image_figures_descriptor(store)["runs"]


def _overlay(store: Path, run_id: str) -> tuple[dict, bytes]:
    """The overlay's descriptor entry in one run folder, and its PNG."""
    entry = _runs(store)[run_id]["bindings"]["cal"]
    [page] = entry["pages"]
    [stored] = page["files"]
    return entry, (store / stored["path"]).read_bytes()


def _deliverable(out: Path) -> bytes:
    [copy] = (out / "deliverables" / "plots" / "cal" / "ds").glob("plate-*.png")
    return copy.read_bytes()


def test_full_mode_stores_the_overlay_png_and_copies_it_out(tmp_path):
    image, pipeline = _write_inputs(tmp_path)
    out = tmp_path / "out"
    store = _full(out, pipeline, image)
    run_id = _run_of(pipeline)
    run = _runs(store)[run_id]
    assert run["failed"] == [] and run["unavailable"] == []
    entry, data = _overlay(store, run_id)
    assert entry["class"] == "CalibrateColorRpcc"
    [page] = entry["pages"]
    assert (page["key"], page["backend"]) == ("default", "mpl")
    assert [(f["format"], f["path"]) for f in page["files"]] == [
        ("png", f"figures/{run_id}/cal/default.png")
    ]
    assert data.startswith(b"\x89PNG")
    assert _deliverable(out) == data


def test_process_mode_zarr_stores_the_overlay(tmp_path):
    from phenotypic._cli._cli_process_only import process_single_apply_only_core

    image, pipeline = _write_inputs(tmp_path)
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline, image_path=image, input_root=image.parent,
        output_dir=out, image_type="Image", layer="rgb", read_kwargs={},
        process_format="zarr", run_initiation=RunInitiation(DAY, f"{DAY}T12:00:00.000Z", 7),
    )
    store = out / "plate.ome.zarr"
    run_id = _run_of(pipeline)
    assert _runs(store)[run_id]["failed"] == []
    _entry, data = _overlay(store, run_id)
    assert data.startswith(b"\x89PNG")


def test_measure_with_the_same_pipeline_keeps_the_overlay_in_its_folder(tmp_path):
    """Revision 14: the same pipeline reuses the folder; §3a keeps the overlay."""
    image, pipeline = _write_inputs(tmp_path)
    out = tmp_path / "out"
    store = _full(out, pipeline, image)
    run_id = _run_of(pipeline)
    entry, data = _overlay(store, run_id)
    table_inode = os.stat(store / MEASUREMENT_TABLE_RELATIVE_PATH).st_ino
    # So the copy below can only be measure mode's own copy-out.
    shutil.rmtree(out / "deliverables" / "plots")

    _measure(out, store, pipeline, LATER)

    assert os.stat(store / MEASUREMENT_TABLE_RELATIVE_PATH).st_ino != table_inode
    assert list(_runs(store)) == [run_id]
    run = _runs(store)[run_id]
    assert run["failed"] == [] and run["unavailable"] == []
    assert _overlay(store, run_id) == (entry, data)
    assert _deliverable(out) == data


def test_measure_with_another_pipeline_lists_the_overlay_unavailable(tmp_path):
    """§3a: no copy between run folders; the folder holding it is untouched."""
    image, first = _write_inputs(tmp_path)
    out = tmp_path / "out"
    store = _full(out, first, image)
    first_run = _run_of(first)
    before = _overlay(store, first_run)

    _, second = _write_inputs(tmp_path / "second", variant=1)
    _measure(out, store, second, LATER)

    new = _runs(store)[_run_of(second, LATER)]
    assert new["unavailable"] == ["cal"]
    assert "cal" not in new["bindings"] and new["failed"] == []
    assert _overlay(store, first_run) == before
    assert not (store / "figures" / _run_of(second, LATER) / "cal").exists()


@pytest.fixture
def fake_gpu(monkeypatch):
    """Make ``FakeGpuDetector`` resolvable by ``ImagePipeline.from_json``."""
    import phenotypic
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)


def test_staged_stage1_draws_the_overlay_and_stage3_keeps_it(tmp_path, fake_gpu, monkeypatch):
    """One run id across stages, whatever the wall clock says (spec §1a)."""
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
    from phenotypic._cli._cli_stage2_token import detector_slot
    from phenotypic._cli._cli_staged_workers import (
        stage1_preprocess_core,
        stage2_detect_core,
        stage3_merge_measure_core,
    )
    from phenotypic._cli._cli_types import Dataset
    from phenotypic.sdk_ import _image_figures

    image, pipeline = _write_inputs(tmp_path, gpu=True)
    out = tmp_path / "out"
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline))
    assert [b.id for b in plan.pre_pipeline.get_plots()] == ["cal"]
    assert plan.post_pipeline.get_plots() == []
    run_id = _run_of(pipeline)

    monkeypatch.setattr(_image_figures, "_utc_now", lambda: datetime(2026, 12, 1, tzinfo=timezone.utc))
    manager = _manager(out)
    manager.create_structure([Dataset("ds", [image], image.parent, out)])
    stage1_preprocess_core(
        plan, image, "ds", "plate", out, manager, image_type="Image",
        pipeline_path=pipeline,
    )
    store = zarr_store_path(out, "ds", "plate")
    assert list(_runs(store)) == [run_id]
    assert _runs(store)[run_id]["failed"] == []
    stage1 = _overlay(store, run_id)
    assert stage1[1].startswith(b"\x89PNG")

    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(plan.gpu_detector, out, "ds", "plate", detector_slot(plan.gpu_path))
    monkeypatch.setattr(_image_figures, "_utc_now", lambda: datetime(2027, 1, 15, tzinfo=timezone.utc))
    stage3_merge_measure_core(plan, out, "ds", "plate", _manager(out), image_type="Image")

    assert (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    assert list(_runs(store)) == [run_id]
    run = _runs(store)[run_id]
    assert run["failed"] == [] and run["unavailable"] == []
    assert _overlay(store, run_id) == stage1
    assert _deliverable(out) == stage1[1]


def _failure_lines(out: Path) -> list[dict]:
    path = plot_failures_jsonl_path(plots_dir(out))
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _staged_manager(out: Path, image: Path) -> OutputManager:
    from phenotypic._cli._cli_types import Dataset

    manager = _manager(out)
    manager.create_structure([Dataset("ds", [image], image.parent, out)])
    return manager


def test_stage3_records_a_failed_stage1_binding_with_its_class(tmp_path, fake_gpu, monkeypatch):
    """MINOR-1: `plot_class` comes from the pipeline's binding (spec §3). A
    Stage-1 binding that failed outright has no class in the descriptor, and
    Stage 3's own pipeline does not carry it."""
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
    from phenotypic._cli._cli_stage2_token import detector_slot
    from phenotypic._cli._cli_staged_workers import (
        stage1_preprocess_core,
        stage2_detect_core,
        stage3_merge_measure_core,
    )

    image, pipeline = _write_inputs(tmp_path, gpu=True)
    out = tmp_path / "out"
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline))
    [binding] = plan.pre_pipeline.get_plots()

    def _cannot_draw(self, subject=None, *, for_save=False, **overrides):
        raise RuntimeError("the overlay could not be drawn")

    monkeypatch.setattr(type(binding.plot), "inspect", _cannot_draw)
    stage1_preprocess_core(
        plan, image, "ds", "plate", out, _staged_manager(out, image),
        image_type="Image", pipeline_path=pipeline,
    )
    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(plan.gpu_detector, out, "ds", "plate", detector_slot(plan.gpu_path))
    stage3_merge_measure_core(plan, out, "ds", "plate", _manager(out), image_type="Image")

    run = _runs(zarr_store_path(out, "ds", "plate"))[_run_of(pipeline)]
    assert run["bindings"] == {}
    assert [(f["binding"], f["page"], f["format"]) for f in run["failed"]] == [
        ("cal", None, None)
    ]
    [line] = [line for line in _failure_lines(out) if line["binding_id"] == "cal"]
    assert line["plot_class"] == "CalibrateColorRpcc"


def test_a_run_folder_that_cannot_be_named_fails_no_image(tmp_path):
    """MINOR-2 (spec §3): a malformed run date -- here handed straight to the
    worker -- costs the image its figures and one record, never the image."""
    from phenotypic._cli._cli_process_single import process_single_image_core
    from phenotypic.plotting._pipeline._store_figures import RUN_FOLDER_FAILURE

    image, pipeline = _write_inputs(tmp_path)
    out = tmp_path / "out"
    manager = OutputManager.from_config(
        out, ".tiff", save_overlays=False, run_initiation=RunInitiation("2026-13-45")
    )
    assert process_single_image_core(pipeline, image, out, "ds", "Image", {}, manager)
    store = zarr_store_path(out, "ds", "plate")
    assert (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    assert read_image_figures_descriptor(store) is None
    [line] = _failure_lines(out)
    assert (line["binding_id"], line["dataset"], line["image_stem"]) == (
        RUN_FOLDER_FAILURE, "ds", "plate"
    )
    assert "YYYY-MM-DD" in line["error"]


def test_stage1_without_a_pipeline_digest_completes_without_figures(tmp_path, fake_gpu):
    """MINOR-2: Stage 1 names its run from the journal. Without a pipeline
    file the journal records no digest; the image gets no figures, not a
    terminal failure."""
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
    from phenotypic._cli._cli_staged_workers import stage1_preprocess_core
    from phenotypic.plotting._pipeline._store_figures import RUN_FOLDER_FAILURE
    from phenotypic.sdk_.ngff_ import valid_staged_store

    image, pipeline = _write_inputs(tmp_path, gpu=True)
    out = tmp_path / "out"
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline))
    stage1_preprocess_core(
        plan, image, "ds", "plate", out, _staged_manager(out, image), image_type="Image"
    )
    store = zarr_store_path(out, "ds", "plate")
    assert valid_staged_store(store)
    assert read_image_figures_descriptor(store) is None
    [line] = _failure_lines(out)
    assert line["binding_id"] == RUN_FOLDER_FAILURE
    assert "no pipeline digest" in line["error"]

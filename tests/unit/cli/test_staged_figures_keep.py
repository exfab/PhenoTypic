"""Stage 3 keeps a post-GPU §3a figure from its own run's folder (spec §3a).

A §3a provider bound AFTER the detector runs in Stage 3, so it is Stage 3's to
draw; when it cannot (``FigureInputUnavailable``), the entry an earlier Stage 3
of the same run already wrote -- a retry, or a same-day rerun -- must be kept
byte for byte, not dropped and not listed as unavailable. Stage 1 never builds
it, so the entry is seeded into the store's run folder the way that earlier
Stage 3 would have left it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_stage2_token import detector_slot
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.abc_.plotting import FigureInputUnavailable, PlotImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import (
    FigureRun,
    RunInitiation,
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_image_figures_descriptor,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector

DAY = "2026-09-22"
SEEDED = b"\x89PNG drawn by an earlier Stage 3 of this run"


class PostApplyState(BaseModel, PlotImage):
    """A §3a provider bound after the detector; this process cannot draw it."""

    def inspect(self, subject=None, *, for_save=False, **overrides):
        raise FigureInputUnavailable("the apply-state lived in an earlier Stage 3")


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)


def _seed_entry(store: Path, run: FigureRun) -> dict:
    """Write the run folder an earlier Stage 3 of this run would have left."""
    page = StoredFigurePage(
        "default", None, "mpl", {},
        (StoredFigureFile("png", "image/png", "default.png", SEEDED),),
    )
    seed = StoredFigures(
        run,
        (StoredFigureBinding("PostApplyState", "PostApplyState", "PostApplyState", (page,)),),
        (),
    )
    Image.load_zarr(store).save2zarr(store, figures=seed)
    return read_image_figures_descriptor(store)["runs"][run.run_id]["bindings"]["PostApplyState"]


def test_stage3_keeps_a_post_gpu_apply_state_figure_from_its_run_folder(tmp_path):
    image_path = tmp_path / "img.tiff"
    load_synth_yeast_plate().rgb.imsave(filepath=image_path)
    out = tmp_path / "out"
    out.mkdir()
    pipeline = ImagePipeline(
        ops=[FakeGpuDetector(output_kind="instance", threshold=0.3)],
        meas=[MeasureSize()],
        plots=[PostApplyState()],
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(pipeline.to_json(), encoding="utf-8")
    plan = split_pipeline_at_gpu(pipeline)
    assert plan.pre_pipeline.get_plots() == []
    assert [b.id for b in plan.post_pipeline.get_plots()] == ["PostApplyState"]

    call = RunInitiation(DAY, f"{DAY}T12:00:00.000Z", 7)
    manager = OutputManager.from_config(
        out, ".tiff", save_overlays=False, run_initiation=call
    )
    manager.create_structure([Dataset("ds", [image_path], tmp_path, out)])

    stage1_preprocess_core(
        plan, image_path, "ds", "img", out, manager, image_type="Image",
        pipeline_path=pipeline_path,
    )
    store = zarr_store_path(out, "ds", "img")
    run = FigureRun(
        date=DAY,
        pipeline_sha256=hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    )
    seeded = _seed_entry(store, run)

    plan.gpu_detector._ensure_model_loaded()
    stage2_detect_core(plan.gpu_detector, out, "ds", "img", detector_slot(plan.gpu_path))
    stage3_merge_measure_core(plan, out, "ds", "img", manager, image_type="Image")

    final = read_image_figures_descriptor(store)["runs"][run.run_id]
    assert final["unavailable"] == [] and final["failed"] == []
    assert final["bindings"]["PostApplyState"] == seeded
    [stored] = final["bindings"]["PostApplyState"]["pages"][0]["files"]
    assert (store / stored["path"]).read_bytes() == SEEDED

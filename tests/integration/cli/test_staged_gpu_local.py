"""End-to-end tests for the local staged GPU engine (Spec 1, Plan 2)."""

import phenotypic
import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_sidecar import sidecar_exists
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.tools_ import dataset_hdf_dir
from tests._fakes.fake_gpu_detector import FakeGpuDetector


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

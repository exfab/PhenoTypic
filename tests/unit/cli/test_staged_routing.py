"""create_execution_strategy routes local GPU runs to the staged engine."""

import phenotypic
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic._cli._cli_execution_strategies import (
    LocalParallelStrategy,
    create_execution_strategy,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_staged_strategy import StagedGpuStrategy
from phenotypic._cli._cli_types import ExecutionConfig
from tests._fakes.fake_gpu_detector import FakeGpuDetector


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )


def _minimal_local_config(
    pipe_path, out, *, process_only_layer=None, measure_only=False
):
    return ExecutionConfig(
        pipeline_json=pipe_path, input_path=out, output_dir=out,
        image_type="Image", nrows=None, ncols=None, bit_depth=None,
        n_jobs=1, slurm_args={}, force_local=True, wait=False, ext=".tiff",
        overlay_alpha=0.5, include_dataset_column=False, dry_run=False,
        sample=None, resume=False, retry_failures=False, skip_validation=True,
        save_overlays=False, measure_only=measure_only,
        process_only_layer=process_only_layer,
    )


def _write_pipe(tmp_path, pipe):
    p = tmp_path / "pipe.json"
    p.write_text(pipe.to_json(), encoding="utf-8")
    return p


def test_gpu_local_routes_to_staged(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), StagedGpuStrategy)


def test_cpu_local_routes_to_local_parallel(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[OtsuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), LocalParallelStrategy)


def test_gpu_measure_only_routes_to_local_parallel(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path, measure_only=True)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), LocalParallelStrategy)


def test_gpu_objmap_process_routes_to_staged(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(
        pipe_path, tmp_path, process_only_layer="objmap"
    )
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), StagedGpuStrategy)

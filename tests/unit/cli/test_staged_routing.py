"""create_execution_strategy routes local GPU runs to the staged engine."""

import phenotypic
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic._cli._cli_execution_strategies import (
    AutonomousSLURMStrategy,
    LocalParallelStrategy,
    create_execution_strategy,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_staged_slurm import StagedSlurmStrategy
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


def _minimal_slurm_config(pipe_path, out, *, measure_only=False):
    return ExecutionConfig(
        pipeline_json=pipe_path, input_path=out, output_dir=out,
        image_type="Image", nrows=None, ncols=None, bit_depth=None,
        n_jobs=1, slurm_args={"slurm_partition": "batch"}, force_local=False,
        wait=False, ext=".tiff", overlay_alpha=0.5, include_dataset_column=False,
        dry_run=False, sample=None, resume=False, retry_failures=False,
        skip_validation=True, save_overlays=False, measure_only=measure_only,
    )


def test_gpu_slurm_routes_to_staged_slurm(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_slurm_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), StagedSlurmStrategy)


def test_cpu_slurm_routes_to_autonomous(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[OtsuDetector()]))
    cfg = _minimal_slurm_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), AutonomousSLURMStrategy)


def test_gpu_slurm_measure_only_routes_to_autonomous(tmp_path):
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_slurm_config(pipe_path, tmp_path, measure_only=True)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), AutonomousSLURMStrategy)


# ---------------------------------------------------------------------------
# Shared run setup: durability log + orphaned-part sweep (Task 3.5)
#
# Both live in ``create_execution_strategy`` rather than in the staged
# strategy, because spec §3.7 and §3.2 are unqualified and a plain --mode full
# CPU run writes its stores through the same promote (OPEN-QUESTIONS G6/P21).
# ---------------------------------------------------------------------------


def _stale(path, hours=24):
    """Backdate *path* past ``SWEEP_MIN_AGE_SECONDS`` (6 h)."""
    import os
    import time

    old = time.time() - hours * 60 * 60
    os.utime(path, (old, old))
    return path


def test_run_start_logs_the_resolved_durability_mode(tmp_path, caplog):
    """The same command carries different guarantees in different places."""
    import logging

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    with caplog.at_level(logging.INFO):
        create_execution_strategy(cfg, om)
    assert any("durable writes:" in record.message for record in caplog.records)


def test_the_logged_durability_mode_tracks_the_environment(
    tmp_path, caplog, monkeypatch
):
    """A hard-coded sentence would pass the previous test and be a lie."""
    import logging

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    with caplog.at_level(logging.INFO):
        create_execution_strategy(cfg, om)
    assert "durable writes: off (local)" in caplog.text

    caplog.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    with caplog.at_level(logging.INFO):
        create_execution_strategy(cfg, om)
    assert "durable writes: on (SLURM)" in caplog.text


def test_run_start_sweeps_stale_orphaned_part_directories(tmp_path, caplog):
    """An interrupted promote leaves a .part; nothing else ever removes it."""
    import logging

    from phenotypic.sdk_ import dataset_zarr_dir

    orphan = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.deadbeef.part"
    orphan.mkdir(parents=True)
    (orphan / "zarr.json").write_text("{}", encoding="utf-8")
    trash = dataset_zarr_dir(tmp_path, "ds") / "img.ome.zarr.trash"
    trash.mkdir(parents=True)
    _stale(orphan)
    _stale(trash)

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    with caplog.at_level(logging.INFO):
        create_execution_strategy(cfg, om)

    assert not orphan.exists()
    assert not trash.exists()
    assert "swept 2 orphaned" in caplog.text


def test_the_sweep_spares_a_recent_part(tmp_path):
    """A uuid says nothing about liveness; under a SLURM array a sibling task
    may be mid-write into exactly this directory."""
    from phenotypic.sdk_ import dataset_zarr_dir

    live = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.cafef00d.part"
    live.mkdir(parents=True)

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    create_execution_strategy(cfg, om)

    assert live.is_dir()


def test_the_sweep_never_touches_a_promoted_store(tmp_path):
    """Only .part/.trash leftovers are orphans; a real store is the output."""
    from phenotypic.sdk_ import zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "img")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    _stale(store)

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    create_execution_strategy(cfg, om)

    assert (store / "zarr.json").is_file()


def test_a_plain_cpu_run_also_logs_durability_and_sweeps(tmp_path, caplog):
    """Spec §3.7 and §3.2 are unqualified; the CPU path uses the same promote.

    Wiring either into the staged strategy alone leaves the common case with
    no durability log and no sweep (OPEN-QUESTIONS G6/P21).
    """
    import logging

    from phenotypic.sdk_ import dataset_zarr_dir

    orphan = dataset_zarr_dir(tmp_path, "ds") / ".img.ome.zarr.deadbeef.part"
    orphan.mkdir(parents=True)
    _stale(orphan)

    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[OtsuDetector()]))
    cfg = _minimal_local_config(pipe_path, tmp_path)
    om = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    with caplog.at_level(logging.INFO):
        strategy = create_execution_strategy(cfg, om)

    assert isinstance(strategy, LocalParallelStrategy)
    assert "durable writes:" in caplog.text
    assert not orphan.exists()


def test_sidecar_module_is_gone():
    """The concept is dead; a surviving module is how it comes back."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phenotypic._cli._cli_sidecar")

"""Live SLURM dispatch of the staged GPU engine (Plan 3 Task 10).

Runs on a real SLURM cluster: submits the 3-link ``afterany`` chain and waits
for it to finish. Uses the CPU-only ``FakeGpuDetector`` with
``slurm_gpus_per_node=0`` (OQ4), so Stage 2 needs no GPU allocation — this
exercises submission + the 3-stage chain + shard distribution + the sidecar
lifecycle without GPU nodes. Gated on ``sbatch`` and marked ``slow``.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import phenotypic
from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_staged_slurm import StagedSlurmStrategy
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from tests._fakes.fake_gpu_detector import FakeGpuDetector

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("sbatch") is None, reason="requires a SLURM environment"
    ),
]

# A real CPU partition; FakeGpuDetector needs no GPU. Defaults to the modern
# EPYC partition (the heterogeneous ``short`` partition includes ancient
# abu_dhabi AMD nodes whose lack of AVX SIGILLs the modern numpy/scipy wheels).
# Override via PHENOTYPIC_TEST_SLURM_PARTITION.
TEST_PARTITION = os.environ.get("PHENOTYPIC_TEST_SLURM_PARTITION", "epyc")
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    # In-process resolution (the submitting test) + tell the fresh SLURM worker
    # processes to preload the fake (sbatch --export=ALL propagates the env).
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )
    monkeypatch.setenv(
        "PHENOTYPIC_PRELOAD_MODULES", "tests._fakes.register_fake_gpu"
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )


def _stage_images(tmp_path, n):
    img = load_synth_yeast_plate()
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.tiff"
        img.rgb.imsave(filepath=p)
        paths.append(p)
    return paths


def _wait_for_parquets(out, stems, timeout_s):
    deadline = time.monotonic() + timeout_s
    meas = out / "results" / "ds" / "measurements"
    while time.monotonic() < deadline:
        if all((meas / f"{s}.parquet").is_file() for s in stems):
            return True
        time.sleep(5)
    return False


def _scancel(job_ids):
    for jid in job_ids:
        if jid:
            subprocess.run(["scancel", str(jid)], check=False)


def _staged_config(out, tmp_path, pipe_path):
    return ExecutionConfig(
        pipeline_json=pipe_path, input_path=tmp_path, output_dir=out,
        image_type="Image", nrows=None, ncols=None, bit_depth=None, n_jobs=1,
        slurm_args={"slurm_partition": TEST_PARTITION, "slurm_time": "00:15:00"},
        force_local=False, wait=False, ext=".tiff", overlay_alpha=0.5,
        include_dataset_column=False, dry_run=False, sample=None, resume=False,
        retry_failures=False, skip_validation=True, save_overlays=False,
        # CPU partition for the GPU stage too: gpus_per_node=0 omits the GPU
        # directive (OQ4), so FakeGpuDetector runs without a GPU allocation.
        gpu_slurm_args={
            "slurm_partition": TEST_PARTITION, "slurm_gpus_per_node": 0,
        },
        gpu_shards=2,
    )


def test_live_3stage_dispatch_completes(tmp_path):
    images = _stage_images(tmp_path, 3)
    out = tmp_path / "out"
    out.mkdir()
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
    )
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, ".tiff", save_overlays=False)
    om.create_structure([Dataset("ds", images, tmp_path, out)])

    strat = StagedSlurmStrategy(_staged_config(out, tmp_path, pipe_path), om)
    strat.execute([Dataset("ds", images, tmp_path, out)], out)
    job_ids = list(getattr(strat, "submitted_job_ids", []))
    assert job_ids and all(job_ids), "expected three submitted SLURM job ids"

    try:
        stems = [p.stem for p in images]
        assert _wait_for_parquets(out, stems, timeout_s=900), (
            "3-stage SLURM afterany chain did not finish within the timeout"
        )
        # Stage 3 deletes every sidecar on completion.
        assert not list((out / "results" / "ds" / "objmap").glob("*.npy"))
    finally:
        _scancel(job_ids)  # C5: never orphan submitted jobs

"""The SLURM finalizer chain, end to end, through the CLI and a fake scheduler.

The CLI submits for real; ``sbatch`` is a fake that runs each job once its
``afterany`` dependency is terminal (``tests/_fakes/fake_slurm.py``). So the
image array, the finalizer chain's ``prepare`` job, and every job ``prepare``
submits in turn all execute the production code, and this suite asserts what
the scheduler saw: one job per finalization stage, each waiting on the one
before, and a run certified only when the last of them finishes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_ import (
    CONFIG_SUFFIX_PIPELINE,
    ensure_typed_json_suffix,
    master_measurements_parquet_path,
    measurements_parquet_path,
)
from tests._fakes.fake_slurm import write_fake_slurm_bin

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake scheduler uses fcntl"
)

_CHAIN_TOKENS = (
    "finalize-shards",
    "finalize-master",
    "finalize-outputs",
    "finalize-qc",
    "finalize-publish",
)
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}


def _wait_for_chain_terminal(state_path: Path, timeout: float = 240.0) -> dict:
    """Wait until the publish job exists and every job is terminal."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = json.loads(state_path.read_text(encoding="utf-8"))
        jobs = last.get("jobs", {})
        has_publish = any(
            str(job.get("comment", "")).endswith(":finalize-publish")
            for job in jobs.values()
        )
        if has_publish and all(
            str(job.get("state")) in _TERMINAL for job in jobs.values()
        ):
            return last
        time.sleep(0.2)
    raise AssertionError(f"the finalizer chain did not finish: {last!r}")


def _job_by_token(state: dict, token: str) -> tuple[str, dict]:
    matches = [
        (job_id, job)
        for job_id, job in state["jobs"].items()
        if str(job.get("comment", "")).endswith(f":{token}")
    ]
    assert len(matches) == 1, (token, matches)
    return matches[0]


def _submit_full_run(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    state_path = tmp_path / "fake-slurm-state.json"
    bin_dir = write_fake_slurm_bin(tmp_path, state_path)
    pipeline_base = tmp_path / "pipeline.json"
    ImagePipeline(ops=[OtsuDetector()]).to_json(pipeline_base)
    pipeline_path = ensure_typed_json_suffix(
        pipeline_base, CONFIG_SUFFIX_PIPELINE
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, box in (("plate1.tiff", (16, 16, 48, 48)), ("plate2.tiff", (8, 8, 40, 40))):
        plate = PILImage.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(plate).ellipse(box, fill=(255, 255, 255))
        plate.save(input_dir / name)
    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "PATH": os.pathsep.join((str(bin_dir), env.get("PATH", ""))),
            "PHENOTYPIC_FAKE_SLURM_AUTORUN": "1",
            "PHENOTYPIC_FAKE_SLURM_STATE": str(state_path),
        }
    )
    env.pop("PYTEST_CURRENT_TEST", None)
    submitted = subprocess.run(
        [
            sys.executable,
            "-m",
            "phenotypic",
            "--pipeline",
            str(pipeline_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--slurm",
            "slurm_partition=compute",
            "--skip-validation",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=300,
    )
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    return output_dir, state_path, env


def test_the_finalizer_runs_as_one_scheduler_job_per_stage(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_completion import valid_run_completion
    from phenotypic.sdk_ import resolve_run_state

    output_dir, state_path, _ = _submit_full_run(tmp_path)
    state = _wait_for_chain_terminal(state_path)

    prepare_id, prepare = _job_by_token(state, "finalizer")
    chain = [_job_by_token(state, token) for token in _CHAIN_TOKENS]

    # prepare waits on the image array; each stage waits on the one before.
    assert prepare["dependency_kind"] == "afterany"
    assert prepare["dependencies"], "prepare did not wait on the image array"
    assert chain[0][1]["dependencies"] == []
    for (previous_id, _), (_, job) in zip(chain, chain[1:]):
        assert job["dependency_kind"] == "afterany"
        assert job["dependencies"] == [previous_id]
    # Nothing in the chain started before the image array was terminal.
    image_ids = set(prepare["dependencies"])
    image_done = max(state["jobs"][job_id]["completed_at"] for job_id in image_ids)
    for _, job in chain:
        assert job["started_at"] >= image_done

    assert all(job["state"] == "COMPLETED" for job in state["jobs"].values()), {
        job_id: job["state"] for job_id, job in state["jobs"].items()
    }
    assert master_measurements_parquet_path(output_dir).is_file()
    assert measurements_parquet_path(output_dir).is_file()
    assert valid_run_completion(output_dir) is not None
    assert resolve_run_state(output_dir, depth="deep").completion == "complete"
    assert prepare_id not in {job_id for job_id, _ in chain}


def test_a_slurm_recompile_finalizes_through_the_same_chain(
    tmp_path: Path,
) -> None:
    """Recompile is the recovery path for a finalizer that died, so it must
    not be the one finalizer left running everything in one job."""
    from phenotypic._cli._cli_completion import valid_run_completion

    output_dir, state_path, env = _submit_full_run(tmp_path)
    first = _wait_for_chain_terminal(state_path)
    first_jobs = set(first["jobs"])

    recompiled = subprocess.run(
        [
            sys.executable,
            "-m",
            "phenotypic",
            "--output",
            str(output_dir),
            "--mode",
            "recompile",
            "--slurm",
            "slurm_partition=compute",
            "--wait",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=300,
    )
    assert recompiled.returncode == 0, recompiled.stdout + recompiled.stderr

    state = _wait_for_chain_terminal(state_path)
    new_jobs = {
        job_id: job
        for job_id, job in state["jobs"].items()
        if job_id not in first_jobs
    }
    tokens = sorted(
        str(job["comment"]).rsplit(":", 1)[-1] for job in new_jobs.values()
    )
    # Recompile's measurement tasks are its shards, so its chain has no
    # shard stage of its own.
    assert "finalize-shards" not in tokens
    for token in ("finalizer", *_CHAIN_TOKENS[1:]):
        assert token in tokens, tokens
    recompile_arrays = [
        job
        for job in new_jobs.values()
        if str(job["comment"]).rsplit(":", 1)[-1].startswith("chunk-")
    ]
    assert recompile_arrays, tokens
    assert all(job["state"] == "COMPLETED" for job in new_jobs.values()), {
        job["comment"]: job["state"] for job in new_jobs.values()
    }
    assert valid_run_completion(output_dir) is not None


def test_a_staged_gpu_run_hands_its_controller_to_the_chain(
    tmp_path: Path,
) -> None:
    """The staged controller submits ``prepare``; the chain does the rest.

    The recovery controller must wait for the chain's last job, or it would
    read the run as finished while the chain is still queued -- so the run
    can only reach ``complete`` here if the hand-off worked.
    """
    from phenotypic._cli._cli_completion import valid_run_completion
    from phenotypic._cli._cli_staged_orchestration import (
        load_orchestration_state,
    )
    from phenotypic.measure import MeasureSize
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    repo_root = Path(__file__).resolve().parents[3]
    state_path = tmp_path / "fake-slurm-state.json"
    bin_dir = write_fake_slurm_bin(tmp_path, state_path)
    pipeline_path = tmp_path / "staged.json"
    import tests._fakes.register_fake_gpu  # noqa: F401  (import side effect)

    pipeline_path.write_text(
        ImagePipeline(
            ops=[FakeGpuDetector(threshold=0.3)], meas=[MeasureSize()]
        ).to_json(),
        encoding="utf-8",
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    plate = PILImage.new("RGB", (64, 64), (0, 0, 0))
    ImageDraw.Draw(plate).ellipse((16, 16, 48, 48), fill=(255, 255, 255))
    plate.save(input_dir / "plate.tiff")
    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "PATH": os.pathsep.join((str(bin_dir), env.get("PATH", ""))),
            "PHENOTYPIC_FAKE_SLURM_AUTORUN": "1",
            "PHENOTYPIC_FAKE_SLURM_STATE": str(state_path),
            "PHENOTYPIC_PRELOAD_MODULES": "tests._fakes.register_fake_gpu",
            "PYTHONPATH": os.pathsep.join(
                (str(repo_root), env.get("PYTHONPATH", ""))
            ),
        }
    )
    env.pop("PYTEST_CURRENT_TEST", None)
    submitted = subprocess.run(
        [
            sys.executable,
            "-m",
            "phenotypic",
            "--pipeline",
            str(pipeline_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--slurm",
            "slurm_partition=compute",
            "--gpu-slurm",
            "slurm_gpus_per_node=0",
            "--skip-validation",
            "--wait",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=300,
    )
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr

    state = json.loads(state_path.read_text(encoding="utf-8"))
    prepare_id, _ = _job_by_token(state, "finalizer")
    publish_id, _ = _job_by_token(state, "finalize-publish")
    for token in _CHAIN_TOKENS:
        _job_by_token(state, token)
    orchestration = load_orchestration_state(output_dir)
    assert orchestration["phase"] == "complete"
    assert valid_run_completion(output_dir) is not None
    assert prepare_id != publish_id

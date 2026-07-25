"""Opt-in live SLURM acceptance for the controller-driven staged GPU engine.

The CPU-only ``FakeGpuDetector`` exercises dynamic controller, recovery,
Stage 1/2/3, and finalizer roles without requesting a GPU. The shared GUI live
harness enforces the exact reviewed SHA, protected remote paths, one small
image, generation-scoped output, and fail-closed scheduler cleanup. Cleanup
retains the exact case plus fd-bound evidence; the dedicated root may be moved
to trash manually only after external scheduler verification.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from uuid import UUID

import pytest

import phenotypic
from phenotypic import ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_staged_orchestration import (
    read_job_ledger,
    staged_completion_path,
)
from phenotypic._cli._cli_staged_slurm import StagedSlurmStrategy
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import job_metadata_path
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from tests._support.live_slurm import (
    cleanup_case,
    prepared_case,
    require_live_environment,
    validate_retained_case_evidence,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PHENOTYPIC_RUN_LIVE_SLURM") != "1",
        reason="set PHENOTYPIC_RUN_LIVE_SLURM=1 for real scheduler tests",
    ),
    pytest.mark.skipif(
        any(
            shutil.which(tool) is None
            for tool in ("sbatch", "squeue", "sacct", "scancel")
        ),
        reason="requires sbatch, squeue, sacct, and scancel",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_TIMEOUT_SECONDS = 10 * 60.0


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the test detector in-process and in fresh SLURM workers."""
    monkeypatch.setattr(
        phenotypic,
        "FakeGpuDetector",
        FakeGpuDetector,
        raising=False,
    )
    monkeypatch.setenv(
        "PHENOTYPIC_PRELOAD_MODULES",
        "tests._fakes.register_fake_gpu",
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{_REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )


def _staged_config(
    output_dir: Path,
    input_dir: Path,
    pipeline_path: Path,
    partition: str,
    image_name: str,
) -> ExecutionConfig:
    """Return a one-image, one-shard, bounded CPU-only staged profile."""
    cpu_profile: dict[str, object] = {
        "slurm_partition": partition,
        "slurm_time": "00:10:00",
        "slurm_mem": "4G",
        "slurm_cpus_per_task": 1,
    }
    account = os.environ.get("PHENOTYPIC_TEST_SLURM_ACCOUNT", "").strip()
    if account:
        cpu_profile["slurm_account"] = account
    return ExecutionConfig(
        pipeline_json=pipeline_path,
        input_path=input_dir,
        output_dir=output_dir,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args=cpu_profile,
        force_local=False,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.5,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
        gpu_slurm_args={
            **cpu_profile,
            "slurm_gpus_per_node": 0,
        },
        gpu_shards=1,
        full_dataset_inventory={"ds": [image_name]},
    )


def _wait_for_staged_completion(output_dir: Path) -> None:
    """Wait for the exact staged completion marker and one measurement."""
    deadline = time.monotonic() + _TERMINAL_TIMEOUT_SECONDS
    measurement_dir = output_dir / "results" / "ds" / "measurements"
    while time.monotonic() < deadline:
        parquets = tuple(measurement_dir.glob("*.parquet"))
        if len(parquets) == 1 and staged_completion_path(output_dir).is_file():
            return
        time.sleep(2.0)
    pytest.fail("one-image staged run did not publish within ten minutes")


def test_live_one_image_staged_dispatch_completes_with_recovery_roles() -> None:
    """One image traverses dynamic controllers and exact staged publication."""
    root, partition, forbidden = require_live_environment()
    with prepared_case(root, forbidden) as (
        case_root,
        pipeline_path,
        output_dir,
    ):
        initial_ids: tuple[str, ...] = ()
        scheduler_generation: UUID | None = None
        try:
            input_dir = case_root / "input"
            images = tuple(input_dir.glob("*.tiff"))
            assert len(images) == 1
            image_path = images[0]
            pipeline_path.write_text(
                ImagePipeline(
                    ops=[FakeGpuDetector(threshold=0.3)],
                    meas=[MeasureSize()],
                ).to_json(),
                encoding="utf-8",
            )
            dataset = Dataset(
                "ds",
                [image_path],
                input_dir,
                output_dir,
            )
            output_manager = OutputManager.from_config(
                output_dir,
                ".tiff",
                save_overlays=False,
            )
            output_manager.create_structure([dataset])
            strategy = StagedSlurmStrategy(
                _staged_config(
                    output_dir,
                    input_dir,
                    pipeline_path,
                    partition,
                    image_path.name,
                ),
                output_manager,
            )

            strategy.execute([dataset], output_dir)
            initial_ids = tuple(
                str(job_id)
                for job_id in getattr(strategy, "submitted_job_ids", ())
            )
            assert len(initial_ids) == 1
            assert all(job_id.isdigit() for job_id in initial_ids)
            metadata = json.loads(
                job_metadata_path(output_dir).read_text(encoding="utf-8")
            )
            scheduler_generation = UUID(str(metadata["slurm_generation"]))

            _wait_for_staged_completion(output_dir)

            ledger = read_job_ledger(output_dir)
            submitted_roles = {
                str(row.get("role"))
                for row in ledger
                if row.get("status") in {"submitted", "recovered"}
            }
            assert {
                "controller-initial",
                "controller",
                "stage1",
                "stage2",
                "stage3",
                "finalizer",
            } <= submitted_roles
            assert any(
                str(row.get("token", "")).startswith("controller-after-")
                and row.get("role") == "controller"
                and row.get("status") in {"submitted", "recovered"}
                for row in ledger
            )
            completion = json.loads(
                staged_completion_path(output_dir).read_text(encoding="utf-8")
            )
            assert completion["epoch"] == scheduler_generation.hex
            assert not tuple(
                (output_dir / "results" / "ds" / "objmap").glob("*.npy")
            )
            print(
                "LIVE_STAGED_COMPLETION "
                f"generation={scheduler_generation.hex} "
                f"output={output_dir} "
                f"initial={','.join(initial_ids)} "
                f"roles={','.join(sorted(submitted_roles))}"
            )
        finally:
            evidence_name = cleanup_case(
                case_root,
                output_dir,
                scheduler_generation,
                iter(initial_ids),
                forbidden=forbidden,
            )
            validate_retained_case_evidence(
                case_root,
                evidence_name,
                scheduler_generation=scheduler_generation,
            )

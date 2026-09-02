"""Lifecycle revocation fences every authoritative provenance checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_process_single import process_single_image_core
from phenotypic._cli._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
)
from phenotypic._cli._cli_stage2_token import (
    stage2_raw_path,
    stage2_token_exists,
)
from phenotypic._cli._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import zarr_store_path


def _root_bytes(store: Path) -> bytes:
    return (store / "zarr.json").read_bytes()


def _revoke_on_call(
    store: Path,
    revoke_at: int,
) -> tuple[dict[str, Any], Any]:
    observed: dict[str, Any] = {"calls": 0, "before_rejection": None}

    def _check() -> None:
        observed["calls"] += 1
        if observed["calls"] == revoke_at:
            observed["before_rejection"] = _root_bytes(store)
            raise SlurmGenerationInactiveError("test lifecycle revoked")

    return observed, _check


def test_stage1_revocation_before_operation_checkpoint_leaves_root_unchanged(
    staged_run_with_provenance: Any,
) -> None:
    run = staged_run_with_provenance
    store = run.store()
    observed, check = _revoke_on_call(store, 2)

    with pytest.raises(
        SlurmGenerationInactiveError, match="test lifecycle revoked"
    ):
        stage1_preprocess_core(
            run.plan,
            run.image_path,
            "ds",
            "img",
            run.output_dir,
            run.output_manager,
            image_type="Image",
            active_check=check,
            work_id=run.work_id,
            pipeline_path=run.pipeline_path,
        )

    assert observed["calls"] == 2
    assert observed["before_rejection"] is not None
    assert _root_bytes(store) == observed["before_rejection"]


def test_stage2_revocation_after_raw_write_never_publishes_token(
    staged_run: Any,
) -> None:
    run = staged_run
    run.run_stage1()
    run.plan.gpu_detector._ensure_model_loaded()
    store = run.store()
    observed, check = _revoke_on_call(store, 2)

    with pytest.raises(
        SlurmGenerationInactiveError, match="test lifecycle revoked"
    ):
        stage2_detect_core(
            run.plan.gpu_detector,
            run.output_dir,
            "ds",
            "img",
            "Image",
            active_check=check,
        )

    assert observed["calls"] == 2
    assert stage2_raw_path(run.output_dir, "ds", "img").is_file()
    assert not stage2_token_exists(run.output_dir, "ds", "img")


@pytest.mark.parametrize("revoke_at", [1, 2, 3])
def test_stage3_revocation_never_mutates_root_after_rejection(
    staged_run_with_provenance: Any,
    revoke_at: int,
) -> None:
    run = staged_run_with_provenance
    run.run_stage1()
    run.run_stage2()
    store = run.store()
    observed, check = _revoke_on_call(store, revoke_at)

    with pytest.raises(
        SlurmGenerationInactiveError, match="test lifecycle revoked"
    ):
        stage3_merge_measure_core(
            run.plan,
            run.output_dir,
            "ds",
            "img",
            run.output_manager,
            image_type="Image",
            active_check=check,
            work_id=run.work_id,
        )

    assert observed["calls"] == revoke_at
    assert observed["before_rejection"] is not None
    assert _root_bytes(store) == observed["before_rejection"]


def test_ordinary_worker_revocation_before_operation_checkpoint_keeps_prefix(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((72, 60, 3), dtype=np.uint8)
    pixels[20:50, 20:40, :] = 200
    image_path = tmp_path / "plate.tiff"
    PILImage.fromarray(pixels).save(image_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(
            ops={
                "prepare": BlurGauss(sigma=1.0),
                "detect": OtsuDetector(),
            }
        ).to_json()
        or "",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    manager = OutputManager.from_config(
        output_dir, ".tiff", save_overlays=False
    )
    manager.create_structure(
        [Dataset("ds", [image_path], tmp_path, output_dir)]
    )
    store = zarr_store_path(output_dir, "ds", image_path.stem)
    observed, check = _revoke_on_call(store, 2)

    with pytest.raises(
        SlurmGenerationInactiveError, match="test lifecycle revoked"
    ):
        process_single_image_core(
            pipeline_path=pipeline_path,
            image_path=image_path,
            output_dir=output_dir,
            dataset_name="ds",
            image_type="Image",
            read_kwargs={},
            output_manager=manager,
            active_check=check,
        )

    assert observed["calls"] == 2
    assert observed["before_rejection"] is not None
    assert _root_bytes(store) == observed["before_rejection"]

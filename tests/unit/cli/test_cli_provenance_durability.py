"""Durable full-forward journals and decoded-original checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import phenotypic
import pytest
import zarr
from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic._cli._cli_failure_tracker import PerImageScientificError
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_process_single import process_single_image_core
from phenotypic._cli._cli_types import Dataset
from phenotypic.correction import CropImage
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss, MedianFilter
from phenotypic.sdk_ import zarr_store_path
from tests._ngff_conformance import assert_store_conforms


class _HardStop(BaseException):
    """Simulate termination that ordinary exception handling cannot catch."""


def _root(store: Path) -> dict[str, Any]:
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))


def _journal(store: Path) -> dict[str, Any]:
    return _root(store)["attributes"]["phenotypic"]["provenance"]


def _worker_case(
    tmp_path: Path, operations: dict[str, Any]
) -> tuple[np.ndarray, Path, Path, Path, OutputManager]:
    pixels = np.zeros((72, 60, 3), dtype=np.uint8)
    pixels[20:50, 20:40, :] = 200
    image_path = tmp_path / "plate.tiff"
    PILImage.fromarray(pixels).save(image_path)
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_operations = {**operations, "detect": OtsuDetector()}
    pipeline_path.write_text(
        ImagePipeline(ops=pipeline_operations).to_json() or "", encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    manager = OutputManager.from_config(
        output_dir, ".tiff", save_overlays=False
    )
    manager.create_structure(
        [Dataset("ds", [image_path], tmp_path, output_dir)]
    )
    store = zarr_store_path(output_dir, "ds", image_path.stem)
    return pixels, image_path, pipeline_path, store, manager


def _run_worker(
    image_path: Path,
    pipeline_path: Path,
    manager: OutputManager,
    *,
    drop_originals: bool = False,
    work_id: str = "durability-work-id",
) -> None:
    process_single_image_core(
        pipeline_path=pipeline_path,
        image_path=image_path,
        output_dir=manager.base_dir,
        dataset_name="ds",
        image_type="Image",
        read_kwargs={},
        output_manager=manager,
        drop_originals=drop_originals,
        work_id=work_id,
    )


def test_full_forward_default_retains_preoperation_pixels_and_completes(
    tmp_path: Path,
) -> None:
    pixels, image_path, pipeline_path, store, manager = _worker_case(
        tmp_path,
        {"crop": CropImage(left=2, right=3, top=4, bottom=5)},
    )

    _run_worker(image_path, pipeline_path, manager)

    journal = _journal(store)
    application = journal["applications"][-1]
    assert journal["status"] == "complete"
    assert application["status"] == "complete"
    assert application["kind"] == "full"
    assert application["input_filename"] == image_path.name
    assert journal["original_filename"] == image_path.name
    assert [entry["pipeline_step_path"] for entry in application["operations"]] == [
        ["crop"], ["detect"]
    ]
    assert application["pipeline"]["source_path"] == pipeline_path.name
    original = np.asarray(
        zarr.open_array(store=str(store / "original" / "0"), mode="r")
    )
    processed = np.asarray(
        zarr.open_array(store=str(store / "rgb" / "0"), mode="r")
    )
    np.testing.assert_array_equal(np.moveaxis(original, 0, -1), pixels)
    assert processed.shape[1:] == (63, 55)
    assert _root(store / "OME")["attributes"]["ome"]["series"][-1] == "original"
    assert 'Name="original"' in (store / "OME" / "METADATA.ome.xml").read_text()
    assert_store_conforms(store)


def test_drop_originals_uses_journal_only_checkpoint_then_final_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, image_path, pipeline_path, store, manager = _worker_case(
        tmp_path, {"median": MedianFilter()}
    )

    def _stop(self: MedianFilter, image: Any) -> Any:
        del self, image
        raise _HardStop()

    real_operate = MedianFilter._operate
    monkeypatch.setattr(MedianFilter, "_operate", _stop)
    with pytest.raises(_HardStop):
        _run_worker(image_path, pipeline_path, manager, drop_originals=True)

    assert _root(store) == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "phenotypic": {
                "provenance": _journal(store),
                "work_id": "durability-work-id",
            }
        },
    }
    assert _journal(store)["status"] == "in_progress"
    assert _journal(store)["applications"][-1]["operations"] == []
    assert not (store / "original").exists()
    assert not (store / "OME").exists()

    monkeypatch.setattr(MedianFilter, "_operate", real_operate)
    _run_worker(image_path, pipeline_path, manager, drop_originals=True)
    assert _journal(store)["status"] == "complete"
    assert len(_journal(store)["applications"]) == 1
    assert len(_journal(store)["applications"][-1]["operations"]) == 2
    assert not (store / "original").exists()
    assert_store_conforms(store)


def test_normal_failure_marks_prefix_failed_and_retry_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pixels, image_path, pipeline_path, store, manager = _worker_case(
        tmp_path,
        {"first": BlurGauss(sigma=1.0), "second": MedianFilter()},
    )

    def _fail(self: MedianFilter, image: Any) -> Any:
        del self, image
        raise RuntimeError("simulated operation failure")

    real_operate = MedianFilter._operate
    monkeypatch.setattr(MedianFilter, "_operate", _fail)
    with pytest.raises(PerImageScientificError):
        _run_worker(image_path, pipeline_path, manager)

    failed = _journal(store)
    failed_application = failed["applications"][-1]
    failed_version = failed_application["phenotypic_version"]
    assert failed["status"] == "failed"
    assert failed_application["status"] == "failed"
    assert [entry["operation_name"] for entry in failed_application["operations"]] == [
        "BlurGauss"
    ]
    retained = np.asarray(
        zarr.open_array(store=str(store / "original" / "0"), mode="r")
    )
    np.testing.assert_array_equal(np.moveaxis(retained, 0, -1), pixels)

    monkeypatch.setattr(MedianFilter, "_operate", real_operate)
    monkeypatch.setattr(phenotypic, "__version__", "retry-build-sentinel")
    _run_worker(image_path, pipeline_path, manager)

    complete = _journal(store)
    complete_application = complete["applications"][-1]
    assert complete["status"] == "complete"
    assert len(complete["applications"]) == 1
    assert complete_application["phenotypic_version"] == failed_version
    assert [entry["operation_name"] for entry in complete_application["operations"]] == [
        "BlurGauss",
        "MedianFilter",
        "OtsuDetector",
    ]
    assert [entry["sequence"] for entry in complete_application["operations"]] == [1, 2, 3]


def test_retry_refuses_checkpoint_from_a_different_work_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, image_path, pipeline_path, store, manager = _worker_case(
        tmp_path, {"first": BlurGauss(sigma=1.0)}
    )

    def _stop(self: BlurGauss, image: Any) -> Any:
        del self, image
        raise _HardStop()

    real_operate = BlurGauss._operate
    monkeypatch.setattr(BlurGauss, "_operate", _stop)
    with pytest.raises(_HardStop):
        _run_worker(image_path, pipeline_path, manager, work_id="work-a")
    before = (store / "zarr.json").read_bytes()

    monkeypatch.setattr(BlurGauss, "_operate", real_operate)
    with pytest.raises(PerImageScientificError, match="work identity"):
        _run_worker(image_path, pipeline_path, manager, work_id="work-b")

    assert (store / "zarr.json").read_bytes() == before


def test_hard_interruption_leaves_valid_original_checkpoint_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pixels, image_path, pipeline_path, store, manager = _worker_case(
        tmp_path, {"first": BlurGauss(sigma=1.0)}
    )

    def _stop(self: BlurGauss, image: Any) -> Any:
        del self, image
        raise _HardStop()

    monkeypatch.setattr(BlurGauss, "_operate", _stop)
    with pytest.raises(_HardStop):
        _run_worker(image_path, pipeline_path, manager)

    assert _journal(store)["status"] == "in_progress"
    assert _journal(store)["applications"][-1]["operations"] == []
    retained = np.asarray(
        zarr.open_array(store=str(store / "original" / "0"), mode="r")
    )
    np.testing.assert_array_equal(np.moveaxis(retained, 0, -1), pixels)
    assert_store_conforms(store)

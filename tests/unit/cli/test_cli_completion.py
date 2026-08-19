"""Tests for marker-last per-image success evidence."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from phenotypic._cli._cli_completion import (
    current_aggregate_is_current,
    current_run_is_complete,
    publish_aggregate_snapshot,
    publish_image_success,
    publish_run_completion_evidence,
    valid_aggregate_snapshot,
    valid_image_success,
    valid_run_completion,
)
from phenotypic._cli._cli_state_management import save_processing_state
from phenotypic._cli._cli_types import DatasetState, ProcessingState
from phenotypic.sdk_ import (
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
)


def test_success_marker_binds_work_id_and_artifact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "plate" / "image.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"scientific result")

    marker = publish_image_success(
        tmp_path,
        work_id="work-a",
        dataset="plate",
        relative_image_path="plate/image.tif",
        image_stem="image",
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="epoch",
        artifacts={"measurements": artifact},
    )

    assert marker.is_file()
    assert valid_image_success(
        tmp_path, dataset="plate", image_stem="image", work_id="work-a"
    )
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="image", work_id="work-b"
    )

    artifact.write_bytes(b"changed")
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="image", work_id="work-a"
    )


def test_aggregate_and_run_markers_reject_mixed_core_bytes(tmp_path: Path) -> None:
    measurement = tmp_path / "results" / "plate" / "measurements" / "image.parquet"
    measurement.parent.mkdir(parents=True)
    measurement.write_bytes(b"image measurement")
    publish_image_success(
        tmp_path,
        work_id="work-a",
        dataset="plate",
        relative_image_path="plate/image.tif",
        image_stem="image",
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="epoch",
        artifacts={"measurements": measurement},
    )
    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=tmp_path / "pipeline.json",
            input_path=tmp_path / "input",
            output_dir=tmp_path,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={"plate": DatasetState(initial_images={"image.tif"})},
            config={
                "success_markers_required": True,
                "work_ids": {"plate": {"image.tif": "work-a"}},
                "processing_generation": "generation",
                "pipeline_sha256": "pipeline",
            },
        ),
        tmp_path,
    )
    core_paths = (
        master_measurements_csv_path(tmp_path),
        master_measurements_parquet_path(tmp_path),
        measurements_csv_path(tmp_path),
        measurements_parquet_path(tmp_path),
    )
    for index, path in enumerate(core_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"core-{index}".encode())

    publish_aggregate_snapshot(tmp_path)
    assert valid_aggregate_snapshot(tmp_path) is not None
    assert current_run_is_complete(tmp_path) is True
    run_marker = publish_run_completion_evidence(
        tmp_path, execution_epoch="local"
    )
    first_bytes = run_marker.read_bytes()
    assert valid_run_completion(tmp_path) is not None
    assert publish_run_completion_evidence(
        tmp_path, execution_epoch="later-noop"
    ).read_bytes() == first_bytes

    measurements_parquet_path(tmp_path).write_bytes(b"mixed")
    assert valid_aggregate_snapshot(tmp_path) is None
    assert current_run_is_complete(tmp_path) is False
    assert valid_run_completion(tmp_path) is None


def test_partial_aggregate_becomes_stale_when_new_success_appears(
    tmp_path: Path,
) -> None:
    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=tmp_path / "pipeline.json",
            input_path=tmp_path / "input",
            output_dir=tmp_path,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={
                "plate": DatasetState(initial_images={"a.tif", "b.tif"})
            },
            config={
                "success_markers_required": True,
                "work_ids": {"plate": {"a.tif": "work-a", "b.tif": "work-b"}},
                "processing_generation": "generation",
                "pipeline_sha256": "pipeline",
            },
        ),
        tmp_path,
    )
    measurements_dir = tmp_path / "results" / "plate" / "measurements"
    measurements_dir.mkdir(parents=True)
    a_path = measurements_dir / "a.parquet"
    a_path.write_bytes(b"a")
    publish_image_success(
        tmp_path,
        work_id="work-a",
        dataset="plate",
        relative_image_path="plate/a.tif",
        image_stem="a",
        mode="full",
        attempt_id="a",
        lifecycle_epoch="epoch",
        artifacts={"measurements": a_path},
    )
    for index, path in enumerate(
        (
            master_measurements_csv_path(tmp_path),
            master_measurements_parquet_path(tmp_path),
            measurements_csv_path(tmp_path),
            measurements_parquet_path(tmp_path),
        )
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"partial-{index}".encode())
    publish_aggregate_snapshot(tmp_path)
    assert current_aggregate_is_current(tmp_path) is True

    b_path = measurements_dir / "b.parquet"
    b_path.write_bytes(b"b")
    publish_image_success(
        tmp_path,
        work_id="work-b",
        dataset="plate",
        relative_image_path="plate/b.tif",
        image_stem="b",
        mode="full",
        attempt_id="b",
        lifecycle_epoch="epoch",
        artifacts={"measurements": b_path},
    )

    assert valid_aggregate_snapshot(tmp_path) is not None
    assert current_aggregate_is_current(tmp_path) is False
    assert current_run_is_complete(tmp_path) is False

"""Artifact-classification tests for staged GPU resume planning."""

from pathlib import Path

import h5py
import numpy as np

from phenotypic._cli._cli_sidecar import write_sidecar
from phenotypic._cli._cli_staged_resume import (
    build_staged_resume_plan,
    migrate_legacy_stage3_markers,
    stage3_completion_exists,
    write_stage3_completion_marker,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.sdk_ import dataset_hdf_dir, dataset_measurements_dir


def _dataset(tmp_path: Path, names: list[str]) -> Dataset:
    images = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"image")
        images.append(path)
    return Dataset("plate", images, tmp_path, tmp_path / "out")


def _valid_hdf(output_dir: Path, stem: str) -> Path:
    path = dataset_hdf_dir(output_dir, "plate") / f"{stem}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 2
        layers = handle.create_group("layers")
        for name in ("gray", "detect_mat", "objmap"):
            layers.create_dataset(name, data=np.zeros((2, 2)))
    return path


def test_resume_plan_classifies_each_durable_stage(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(
        tmp_path, ["stage1.tif", "stage2.tif", "stage3.tif", "done.tif"]
    )
    _valid_hdf(output_dir, "stage2")
    _valid_hdf(output_dir, "stage3")
    write_sidecar(output_dir, "plate", "stage3", np.zeros((2, 2)))
    _valid_hdf(output_dir, "done")
    write_stage3_completion_marker(
        output_dir, "plate", "done.tif", "done"
    )

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"
    assert plan.counts == {
        "stage1": 1,
        "stage2": 1,
        "stage3": 1,
        "complete": 1,
    }
    assert [image.name for image in plan.datasets[0].images] == [
        "stage1.tif",
        "stage2.tif",
        "stage3.tif",
    ]


def test_invalid_hdf_requires_stage1(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["broken.tif"])
    hdf = dataset_hdf_dir(output_dir, "plate") / "broken.h5"
    hdf.parent.mkdir(parents=True, exist_ok=True)
    hdf.write_bytes(b"not hdf5")

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"


def test_hdf_without_phenotypic_layers_requires_stage1(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["foreign.tif"])
    hdf = dataset_hdf_dir(output_dir, "plate") / "foreign.h5"
    hdf.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf, "w"):
        pass

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"


def test_terminal_marker_does_not_mask_invalid_hdf(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["broken.tif"])
    write_stage3_completion_marker(
        output_dir, "plate", "broken.tif", "broken"
    )

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"


def test_legacy_parquet_is_migrated_to_terminal_marker(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["legacy.tif"])
    _valid_hdf(output_dir, "legacy")
    parquet = dataset_measurements_dir(output_dir, "plate") / "legacy.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"legacy parquet")

    legacy_plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=False,
    )
    assert legacy_plan.initial_stage == "complete"
    assert migrate_legacy_stage3_markers(output_dir, legacy_plan) == 1
    assert stage3_completion_exists(output_dir, "plate", "legacy")

    current_plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )
    assert current_plan.initial_stage == "complete"


def test_unmarked_current_parquet_is_not_terminal(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["partial.tif"])
    _valid_hdf(output_dir, "partial")
    parquet = dataset_measurements_dir(output_dir, "plate") / "partial.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"partial publication")

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage2"

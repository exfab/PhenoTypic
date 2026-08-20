"""Artifact-classification tests for staged GPU resume planning."""

import json
from pathlib import Path

import numpy as np

from phenotypic import Image
from phenotypic._cli._cli_stage2_token import (
    delete_stage2_raw,
    stage2_token_path,
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_resume import (
    build_staged_resume_plan,
    clear_downstream_artifacts_for_stage1,
    migrate_legacy_stage3_markers,
    stage3_completion_exists,
    write_stage3_completion_marker,
)
from phenotypic._cli._cli_types import Dataset
from phenotypic.sdk_ import dataset_measurements_dir, zarr_store_path


def _write_stage2_signal(output_dir: Path, stem: str) -> None:
    """The sidecar's replacement is the raw array AND the token, together."""
    write_stage2_raw(
        output_dir, "plate", stem, np.zeros((4, 4), dtype=np.uint16)
    )
    write_stage2_token(output_dir, "plate", stem, objmap_shape=(4, 4))


def _dataset(tmp_path: Path, names: list[str]) -> Dataset:
    images = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"image")
        images.append(path)
    return Dataset("plate", images, tmp_path, tmp_path / "out")


def _valid_store(output_dir: Path, stem: str, *, work_id: str | None = None) -> Path:
    path = zarr_store_path(output_dir, "plate", stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image(np.zeros((4, 4, 3), dtype=np.uint8)).save2zarr(path, work_id=work_id)
    return path


def test_resume_plan_classifies_each_durable_stage(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(
        tmp_path, ["stage1.tif", "stage2.tif", "stage3.tif", "done.tif"]
    )
    _valid_store(output_dir, "stage2")
    _valid_store(output_dir, "stage3")
    _write_stage2_signal(output_dir, "stage3")
    _valid_store(output_dir, "done")
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


def test_invalid_store_requires_stage1(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["broken.tif"])
    store = _valid_store(output_dir, "broken")
    # Corrupt the root: an unparseable zarr.json reads as absent, not partial.
    (store / "zarr.json").write_text("{not json", encoding="utf-8")

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"


def test_changed_work_id_restarts_from_stage1(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["changed.tif"])
    _valid_store(output_dir, "changed", work_id="old-work")
    _write_stage2_signal(output_dir, "changed")
    write_stage3_completion_marker(
        output_dir, "plate", "changed.tif", "changed"
    )

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
        work_ids={"plate": {"changed.tif": "new-work"}},
    )

    assert plan.initial_stage == "stage1"


def test_store_without_phenotypic_block_requires_stage1(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["foreign.tif"])
    # A directory that looks like a store to a path glob but carries no
    # ``attributes.phenotypic`` -- another tool's zarr, or a future schema.
    store = zarr_store_path(output_dir, "plate", "foreign")
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group"}), encoding="utf-8"
    )

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage1"


def test_terminal_marker_does_not_mask_a_missing_store(tmp_path: Path) -> None:
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
    _valid_store(output_dir, "legacy")
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
    _valid_store(output_dir, "partial")
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


def test_token_without_its_raw_array_routes_back_to_stage2(
    tmp_path: Path,
) -> None:
    """The token is a flag; the raw ``.npy`` is Stage 3's actual input.

    Without an explicit branch such an image classifies ``"stage3"`` forever:
    the worker reports a missing prereq rather than a scientific failure, but
    nothing ever routes it back to Stage 2, so it cannot recover.
    """
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["partial.tif"])
    _valid_store(output_dir, "partial")
    _write_stage2_signal(output_dir, "partial")
    delete_stage2_raw(output_dir, "plate", "partial")
    assert stage2_token_path(output_dir, "plate", "partial").is_file()

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage2"


def test_token_without_raw_and_with_a_parquet_is_still_stage2(
    tmp_path: Path,
) -> None:
    """The raw check must NOT be folded into ``stage2_done``.

    ``not stage2_done`` is a conjunct of the legacy-``"complete"`` branch, so
    ANDing the raw array into the token probe would flip exactly this image --
    token present, raw missing, parquet on disk -- all the way to
    ``"complete"``, permanently skipping the Stage 3 that never ran.
    """
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["halfway.tif"])
    _valid_store(output_dir, "halfway")
    _write_stage2_signal(output_dir, "halfway")
    delete_stage2_raw(output_dir, "plate", "halfway")
    parquet = dataset_measurements_dir(output_dir, "plate") / "halfway.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"partial publication")

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=False,
    )

    assert plan.initial_stage == "stage2"


def test_raw_array_without_a_token_is_stage2(tmp_path: Path) -> None:
    """Raw is written before the token, so this is the crash-mid-Stage-2 state.

    Stage 2 simply recomputes and overwrites; it must never read as Stage 3.
    """
    output_dir = tmp_path / "out"
    dataset = _dataset(tmp_path, ["crashed.tif"])
    _valid_store(output_dir, "crashed")
    write_stage2_raw(
        output_dir, "plate", "crashed", np.zeros((4, 4), dtype=np.uint16)
    )

    plan = build_staged_resume_plan(
        datasets=[dataset],
        output_dir=output_dir,
        input_root=tmp_path,
        process_only_layer=None,
        markers_required=True,
    )

    assert plan.initial_stage == "stage2"


def test_clear_downstream_artifacts_removes_token_raw_and_marker(
    tmp_path: Path,
) -> None:
    """All three, and the store is deliberately left alone (D13).

    Removing the store here would open a window in which the image is absent
    and destroy the only fallback if the following Stage 1 fails; Stage 1's
    promote replaces it atomically anyway.
    """
    from phenotypic._cli._cli_stage2_token import stage2_raw_path

    output_dir = tmp_path / "out"
    store = _valid_store(output_dir, "redo")
    _write_stage2_signal(output_dir, "redo")
    write_stage3_completion_marker(output_dir, "plate", "redo.tif", "redo")

    clear_downstream_artifacts_for_stage1(output_dir, "plate", "redo")

    assert not stage2_token_path(output_dir, "plate", "redo").exists()
    assert not stage2_raw_path(output_dir, "plate", "redo").exists()
    assert not stage3_completion_exists(output_dir, "plate", "redo")
    assert store.is_dir(), "the store must survive until Stage 1 replaces it"


def test_reconcile_consumes_the_token_and_the_raw_array(tmp_path: Path) -> None:
    from phenotypic._cli._cli_stage2_token import stage2_raw_path
    from phenotypic._cli._cli_staged_resume import reconcile_stage3_publications

    output_dir = tmp_path / "out"
    parquet = dataset_measurements_dir(output_dir, "plate") / "done.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"complete")
    _write_stage2_signal(output_dir, "done")
    write_stage3_completion_marker(output_dir, "plate", "done.tif", "done")

    moved = reconcile_stage3_publications(
        output_dir, {"plate": ["done.tif"]}, namespace="test"
    )

    assert moved == 0
    assert not stage2_token_path(output_dir, "plate", "done").exists()
    assert not stage2_raw_path(output_dir, "plate", "done").exists()
    assert parquet.is_file()

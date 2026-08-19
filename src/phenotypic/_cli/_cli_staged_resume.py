"""Artifact-defined resume planning for staged GPU pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

import h5py  # type: ignore[import-untyped]

from phenotypic.sdk_ import (
    atomic_write_json,
    dataset_hdf_dir,
    dataset_measurements_dir,
    progress_dir,
)

from ._cli_process_only import process_only_output_path
from ._cli_sidecar import delete_sidecar, sidecar_exists
from ._cli_types import Dataset

ResumeStage = Literal["stage1", "stage2", "stage3", "complete"]
_STAGE_ORDER: dict[ResumeStage, int] = {
    "stage1": 0,
    "stage2": 1,
    "stage3": 2,
    "complete": 3,
}


@dataclass(frozen=True)
class StagedResumeItem:
    """One image and the earliest staged operation it still requires."""

    dataset: str
    image: Path
    stage: ResumeStage


@dataclass(frozen=True)
class StagedResumePlan:
    """Artifact-derived worklist and global initial stage for a resume run."""

    datasets: list[Dataset]
    items: tuple[StagedResumeItem, ...]
    initial_stage: ResumeStage

    @property
    def counts(self) -> dict[ResumeStage, int]:
        """Return image counts grouped by required stage."""
        counts: dict[ResumeStage, int] = {
            "stage1": 0,
            "stage2": 0,
            "stage3": 0,
            "complete": 0,
        }
        for item in self.items:
            counts[item.stage] += 1
        return counts


def pipeline_content_digest(pipeline_path: Path) -> str:
    """Return a SHA-256 digest of the serialized pipeline definition."""
    return hashlib.sha256(Path(pipeline_path).read_bytes()).hexdigest()


def valid_staged_hdf(path: Path) -> bool:
    """Return whether *path* contains the image layers Stage 2 requires."""
    try:
        if not path.is_file() or not h5py.is_hdf5(path):
            return False
        with h5py.File(path, "r") as handle:
            schema_version = int(handle.attrs.get("schema_version", 1))
            layers = (
                handle["layers"]
                if schema_version >= 2 and "layers" in handle
                else handle
            )
            detect_name = (
                "detect_mat" if "detect_mat" in layers else "enh_gray"
            )
            names = ("gray", detect_name, "objmap")
            if any(name not in layers for name in names):
                return False
            datasets = [layers[name] for name in names]
            if any(not isinstance(item, h5py.Dataset) for item in datasets):
                return False
            shapes = [item.shape for item in datasets]
            return all(
                len(shape) >= 2 and shape[0] > 0 and shape[1] > 0
                for shape in shapes
            ) and all(shape[:2] == shapes[0][:2] for shape in shapes[1:])
    except (OSError, TypeError, ValueError):
        return False


def staged_hdf_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged HDF is bound to ``work_id``."""
    if not valid_staged_hdf(path):
        return False
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("phenotypic_work_id")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value == work_id
    except (OSError, UnicodeDecodeError):
        return False


def stage3_completion_marker_path(
    output_dir: Path, dataset: str, image_stem: str
) -> Path:
    """Return the durable per-image Stage 3 completion marker path."""
    return (
        progress_dir(output_dir)
        / "stage3_complete"
        / dataset
        / f"{image_stem}.json"
    )


def stage3_completion_exists(
    output_dir: Path, dataset: str, image_stem: str
) -> bool:
    """Return whether Stage 3 fully published all required image artifacts."""
    return stage3_completion_marker_path(
        output_dir, dataset, image_stem
    ).is_file()


def write_stage3_completion_marker(
    output_dir: Path,
    dataset: str,
    image_name: str,
    image_stem: str,
    *,
    legacy_migration: bool = False,
) -> Path:
    """Atomically record complete Stage 3 publication for one image."""
    path = stage3_completion_marker_path(output_dir, dataset, image_stem)
    atomic_write_json(
        path,
        {
            "version": 1,
            "dataset": dataset,
            "image_name": image_name,
            "stem": image_stem,
            "legacy_migration": legacy_migration,
            "completed_at": datetime.now().isoformat(timespec="milliseconds"),
        },
    )
    return path


def remove_stage3_completion_marker(
    output_dir: Path, dataset: str, image_stem: str
) -> None:
    """Remove a terminal marker before regenerating upstream artifacts."""
    stage3_completion_marker_path(
        output_dir, dataset, image_stem
    ).unlink(missing_ok=True)


def classify_staged_image(
    *,
    output_dir: Path,
    dataset: str,
    image: Path,
    input_root: Path,
    process_only_layer: str | None,
    markers_required: bool,
    expected_work_id: str | None = None,
) -> ResumeStage:
    """Return the earliest stage required by one image's durable artifacts."""
    stem = image.stem
    if expected_work_id is not None:
        from ._cli_completion import valid_image_success

        if valid_image_success(
            output_dir,
            dataset=dataset,
            image_stem=stem,
            work_id=expected_work_id,
        ):
            return "complete"

    if process_only_layer == "objmap":
        terminal = process_only_output_path(
            output_dir, image, input_root, "objmap"
        )
        if terminal.is_file() and expected_work_id is None:
            return "complete"

    hdf = dataset_hdf_dir(output_dir, dataset) / f"{stem}.h5"
    if expected_work_id is not None:
        hdf_valid = staged_hdf_matches_work_id(hdf, expected_work_id)
    else:
        hdf_valid = valid_staged_hdf(hdf)
    if not hdf_valid:
        return "stage1"

    if (
        process_only_layer is None
        and expected_work_id is None
        and stage3_completion_exists(output_dir, dataset, stem)
    ):
        return "complete"
    if (
        process_only_layer is None
        and expected_work_id is not None
        and stage3_completion_exists(output_dir, dataset, stem)
        and (
            dataset_measurements_dir(output_dir, dataset) / f"{stem}.parquet"
        ).is_file()
    ):
        return "stage3"

    sidecar = sidecar_exists(output_dir, dataset, stem)
    parquet = (
        dataset_measurements_dir(output_dir, dataset) / f"{stem}.parquet"
    )
    if (
        process_only_layer is None
        and not markers_required
        and parquet.is_file()
        and not sidecar
    ):
        return "complete"

    return "stage3" if sidecar else "stage2"


def build_staged_resume_plan(
    *,
    datasets: Sequence[Dataset],
    output_dir: Path,
    input_root: Path,
    process_only_layer: str | None,
    markers_required: bool,
    work_ids: Mapping[str, Mapping[str, str]] | None = None,
) -> StagedResumePlan:
    """Build a filtered worklist and earliest global resume stage."""
    items: list[StagedResumeItem] = []
    pending_by_dataset: dict[str, list[Path]] = {}
    source_by_dataset = {dataset.name: dataset for dataset in datasets}
    for dataset in datasets:
        for image in dataset.images:
            expected_work_id = (
                work_ids.get(dataset.name, {}).get(image.name)
                if work_ids is not None
                else None
            )
            stage = classify_staged_image(
                output_dir=output_dir,
                dataset=dataset.name,
                image=image,
                input_root=input_root,
                process_only_layer=process_only_layer,
                markers_required=markers_required,
                expected_work_id=expected_work_id,
            )
            items.append(StagedResumeItem(dataset.name, image, stage))
            if stage != "complete":
                pending_by_dataset.setdefault(dataset.name, []).append(image)

    pending = [
        Dataset(
            name=name,
            images=images,
            input_dir=source_by_dataset[name].input_dir,
            output_dir=source_by_dataset[name].output_dir,
        )
        for name, images in pending_by_dataset.items()
    ]
    incomplete = [item.stage for item in items if item.stage != "complete"]
    initial: ResumeStage = (
        min(incomplete, key=_STAGE_ORDER.__getitem__)
        if incomplete
        else "complete"
    )
    return StagedResumePlan(pending, tuple(items), initial)


def migrate_legacy_stage3_markers(
    output_dir: Path, plan: StagedResumePlan
) -> int:
    """Create markers for legacy parquet-only completions discovered safely."""
    migrated = 0
    for item in plan.items:
        if item.stage != "complete":
            continue
        parquet = (
            dataset_measurements_dir(output_dir, item.dataset)
            / f"{item.image.stem}.parquet"
        )
        if not parquet.is_file() or stage3_completion_exists(
            output_dir, item.dataset, item.image.stem
        ):
            continue
        write_stage3_completion_marker(
            output_dir,
            item.dataset,
            item.image.name,
            item.image.stem,
            legacy_migration=True,
        )
        migrated += 1
    return migrated


def clear_downstream_artifacts_for_stage1(
    output_dir: Path, dataset: str, image_stem: str
) -> None:
    """Discard artifacts that cannot survive regeneration of the staged HDF."""
    delete_sidecar(output_dir, dataset, image_stem)
    remove_stage3_completion_marker(output_dir, dataset, image_stem)


def reconcile_stage3_publications(
    output_dir: Path,
    inventory: Mapping[str, Sequence[str]],
    *,
    namespace: str,
) -> int:
    """Clean completed sidecars and quarantine unmarked parquets.

    A parquet is only eligible for aggregation after its terminal Stage 3
    marker exists. Unmarked parquets are preserved outside the aggregation
    tree so a later resume can safely republish them.
    """
    moved = 0
    quarantine = progress_dir(output_dir) / "unpublished_stage3" / namespace
    from ._cli_completion import valid_image_success
    from ._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    work_ids = state.config.get("work_ids", {}) if state is not None else {}
    for dataset, image_names in inventory.items():
        for image_name in image_names:
            stem = Path(image_name).stem
            dataset_work_ids = (
                work_ids.get(dataset, {}) if isinstance(work_ids, dict) else {}
            )
            work_id = (
                dataset_work_ids.get(image_name)
                if isinstance(dataset_work_ids, dict)
                else None
            )
            generally_complete = bool(
                isinstance(work_id, str)
                and valid_image_success(
                    output_dir,
                    dataset=dataset,
                    image_stem=stem,
                    work_id=work_id,
                )
            )
            if generally_complete or stage3_completion_exists(
                output_dir, dataset, stem
            ):
                delete_sidecar(output_dir, dataset, stem)
                continue
            parquet = (
                dataset_measurements_dir(output_dir, dataset)
                / f"{stem}.parquet"
            )
            if not parquet.is_file():
                continue
            destination = quarantine / dataset / parquet.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            parquet.replace(destination)
            moved += 1
    return moved


__all__ = [
    "ResumeStage",
    "StagedResumeItem",
    "StagedResumePlan",
    "build_staged_resume_plan",
    "classify_staged_image",
    "clear_downstream_artifacts_for_stage1",
    "migrate_legacy_stage3_markers",
    "pipeline_content_digest",
    "reconcile_stage3_publications",
    "remove_stage3_completion_marker",
    "stage3_completion_exists",
    "stage3_completion_marker_path",
    "staged_hdf_matches_work_id",
    "valid_staged_hdf",
    "write_stage3_completion_marker",
]

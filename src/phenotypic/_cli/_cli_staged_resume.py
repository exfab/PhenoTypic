"""Artifact-defined resume planning for staged GPU pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

from phenotypic.sdk_ import (
    CommitGuard,
    atomic_write_json,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    dataset_measurements_dir,
    progress_dir,
    publication_commit,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import (  # noqa: F401 -- public re-export
    PhenotypicAttr,
    read_phenotypic_attributes,
    valid_staged_store,
)

from ._cli_process_only import process_only_output_path
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    stage2_raw_path,
    stage2_token_exists,
)
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


def valid_stage1_store(path: Path) -> bool:
    """Return whether a structurally valid store finished Stage 1.

    A decoded checkpoint is intentionally a valid OME-Zarr store so hard
    interruption never loses the source image. Its provenance lifecycle is
    still ``in_progress``, however, and therefore it must not be consumed by
    Stage 2 or mistaken for a resumable Stage-1 result. Stores created before
    provenance was introduced remain eligible through the missing-journal
    compatibility branch.
    """
    if not valid_staged_store(path):
        return False
    try:
        journal = read_phenotypic_attributes(path).get(
            PhenotypicAttr.PROVENANCE
        )
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False
    if journal is None:
        return True
    if not isinstance(journal, Mapping):
        return False
    return journal.get("status") in {"staged", "complete"}


def _staged_store_has_work_id(path: Path, work_id: str) -> bool:
    """Return whether a structurally valid store carries ``work_id``."""
    if not valid_staged_store(path):
        return False
    try:
        block = read_phenotypic_attributes(path)
        return block.get(PhenotypicAttr.WORK_ID) == work_id
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return False


def staged_store_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged store is bound to ``work_id``.

    Replaces ``staged_hdf_matches_work_id``. The work id lives in
    ``attributes.phenotypic.work_id``, written at store-build time -- never
    patched in afterwards, because the root ``zarr.json`` is written last.

    Args:
        path: Candidate ``*.ome.zarr`` directory.
        work_id: The work id this run expects the store to carry.

    Returns:
        ``True`` only for a valid staged store bound to *work_id*.
    """
    return valid_stage1_store(path) and _staged_store_has_work_id(
        path, work_id
    )


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
    commit_guard: CommitGuard | None = None,
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
        commit_guard=commit_guard,
    )
    return path


def remove_stage3_completion_marker(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Remove a terminal marker before regenerating upstream artifacts."""
    with publication_commit(commit_guard):
        stage3_completion_marker_path(output_dir, dataset, image_stem).unlink(
            missing_ok=True
        )


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
            output_dir, image, input_root, "objmap", fmt="tiff"
        )
        if terminal.is_file() and expected_work_id is None:
            return "complete"

    store = zarr_store_path(output_dir, dataset, stem)
    stage2_done = stage2_token_exists(output_dir, dataset, stem)
    if expected_work_id is not None:
        store_valid = staged_store_matches_work_id(store, expected_work_id)
        stage2_store_valid = _staged_store_has_work_id(store, expected_work_id)
    else:
        store_valid = valid_stage1_store(store)
        stage2_store_valid = valid_staged_store(store)
    # Stage 3 deliberately marks the root journal failed/in_progress before
    # publication. A retained Stage-2 token proves this same store previously
    # completed Stage 1, so it remains eligible for Stage 3 retry (or Stage 2
    # regeneration when the token's raw sidecar is missing).
    if not store_valid and not (stage2_done and stage2_store_valid):
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
            zarr_store_path(output_dir, dataset, stem)
            / MEASUREMENT_TABLE_RELATIVE_PATH
        ).is_file()
    ):
        return "stage3"

    measurement_table = store / MEASUREMENT_TABLE_RELATIVE_PATH
    if (
        process_only_layer is None
        and not markers_required
        and measurement_table.is_file()
        and not stage2_done
    ):
        return "complete"

    # An explicit branch, NOT `stage2_done and raw.is_file()` (ledger FLOW-40).
    # The token is only a flag; Stage 3's real INPUT is the raw .npy. Without
    # this, a token-present/raw-missing image classifies "stage3" forever: the
    # worker reports a missing prereq rather than a scientific failure -- an
    # improvement -- but nothing ever routes it back to Stage 2, so it cannot
    # recover.
    #
    # It must NOT be folded into `stage2_done`, because `not stage2_done` is a
    # conjunct of the "complete" branch above: ANDing the raw in would flip a
    # token-present/raw-missing image that has a parquet all the way to
    # "complete".
    if (
        stage2_done
        and not stage2_raw_path(output_dir, dataset, stem).is_file()
    ):
        return "stage2"

    return "stage3" if stage2_done else "stage2"


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
    """Create markers for completed embedded or explicit legacy tables."""
    migrated = 0
    for item in plan.items:
        if item.stage != "complete":
            continue
        measurement_table = (
            zarr_store_path(output_dir, item.dataset, item.image.stem)
            / MEASUREMENT_TABLE_RELATIVE_PATH
        )
        legacy_table = (
            dataset_measurements_dir(output_dir, item.dataset)
            / f"{item.image.stem}.parquet"
        )
        if (
            not measurement_table.is_file() and not legacy_table.is_file()
        ) or stage3_completion_exists(
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
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Discard artifacts that cannot survive regeneration of the staged store.

    Deletes nothing else. The store itself is left alone: Stage 1's promote
    replaces it atomically, so removing it here would only open a window in
    which the image is absent and destroy the fallback if Stage 1 then fails
    (OPEN-QUESTIONS **D13**).

    Token first, then the raw array -- deleting the raw and leaving the token
    makes the next Stage 3 replay into a ``FileNotFoundError``, while the
    reverse merely orphans a ``.npy`` that Stage 2 overwrites.
    """
    delete_stage2_token(
        output_dir, dataset, image_stem, commit_guard=commit_guard
    )
    delete_stage2_raw(
        output_dir, dataset, image_stem, commit_guard=commit_guard
    )
    remove_stage3_completion_marker(
        output_dir,
        dataset,
        image_stem,
        commit_guard=commit_guard,
    )


def reconcile_stage3_publications(
    output_dir: Path,
    inventory: Mapping[str, Sequence[str]],
    *,
    namespace: str,
) -> int:
    """Consume completed Stage-2 signals and quarantine unmarked parquets.

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
                # Token first, then the raw array -- same ordering rule as
                # every other consumption site.
                delete_stage2_token(output_dir, dataset, stem)
                delete_stage2_raw(output_dir, dataset, stem)
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
    "staged_store_matches_work_id",
    "valid_stage1_store",
    "valid_staged_store",
    "write_stage3_completion_marker",
]

"""Generation-bound authority barriers for ``--mode migrate``."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import polars as pl
import pytest

from phenotypic._cli._cli_migrate_image import (
    MigrationImagePartialResult,
    MigrationImageResult,
    MigrationImageStageError,
    MigrationImageStageFailure,
    ReclaimResult,
    SourceArtifactState,
    _migration_work_id,
)
from phenotypic._cli._cli_migrate import (
    MetadataPassResult,
    _execute_migration_tasks,
    _report_from_image_results,
    close_migration_generation,
    finalize_migration_attempt,
    invalidate_migration_terminal_authority,
    migration_terminal_status_path,
    publish_migration_terminal_status,
    run_migrate,
    echo_migration_summary,
)
from phenotypic._cli._cli_migrate_manifest import (
    MigrationImageTask,
    MigrationReclaimSeal,
    migration_image_seal_path,
    migration_reclaim_status_path,
    migration_task_status_path,
    publish_migration_reclaim_status,
    publish_migration_task_status,
    seal_migration_image_stage,
    seal_migration_reclaim_stage,
    write_migration_manifest,
)
from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    generation_is_active,
    initialize_slurm_lifecycle,
    ledger_job_for_token,
    load_slurm_lifecycle,
)
from phenotypic._cli._cli_completion import (
    publish_aggregate_snapshot,
    publish_image_success,
    publish_run_completion_evidence,
)
from phenotypic._cli._cli_state_management import (
    load_processing_state,
    save_processing_state,
)
from phenotypic._cli._cli_types import DatasetState, ProcessingState
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    aggregate_publication_marker_path,
    deliverables_dir,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    phenotypic_cache_dir,
    run_completion_marker_path,
)
from phenotypic.sdk_._hdf_to_zarr import emit_canonical_metadata_view
from phenotypic.sdk_._hdf_to_zarr import MigrationReport


_GENERATION = "generation-1"
_METADATA_DIGEST = "sha256:" + "a" * 64


def _task(run: Path, index: int) -> MigrationImageTask:
    """Return one canonical store-backed migration task."""
    stem = f"image-{index}"
    return MigrationImageTask(
        index=index,
        dataset="ds",
        stem=stem,
        hdf_path=run / "results" / "ds" / "hdf" / f"{stem}.h5",
        store_path=run / "results" / "ds" / "zarr" / f"{stem}.ome.zarr",
        measurement_path=(
            run / "results" / "ds" / "measurements" / f"{stem}.parquet"
        ),
        overlay_path=run / "deliverables" / "overlays" / "ds" / f"{stem}.png",
        marker_path=(
            run
            / ".phenotypic"
            / "progress"
            / "image_complete"
            / "ds"
            / f"{stem}.json"
        ),
    )


@pytest.mark.parametrize(
    ("stage", "table_installed", "expected_failure", "expected_summary"),
    (
        ("table", False, "table_failures", "Pass 3 FAILED"),
        ("overlay", True, "overlay_failures", "Pass 4 FAILED"),
    ),
)
def test_partial_stage_failures_keep_legacy_counts_and_cli_categories(
    tmp_path: Path,
    stage: str,
    table_installed: bool,
    expected_failure: str,
    expected_summary: str,
) -> None:
    """Later-stage failures do not erase completed legacy pass evidence."""
    from click.testing import CliRunner

    task = _task(tmp_path / "run", 0)
    partial = MigrationImagePartialResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=_migration_work_id(task.dataset, task.stem),
        converted=True,
        table_installed=table_installed,
        overlay_rendered=False,
    )
    target = task.measurement_path if stage == "table" else task.overlay_path
    assert target is not None
    failure = MigrationImageStageFailure(
        stage=stage,
        target=target,
        reason=f"{stage} failed",
        partial=partial,
    )

    report = _report_from_image_results((task,), (), (failure,))

    assert report.converted == 1
    assert report.tables_migrated == int(table_installed)
    assert getattr(report, expected_failure) == ((target, f"{stage} failed"),)
    assert report.failed == ()
    runner = CliRunner()
    with runner.isolation() as streams:
        echo_migration_summary(tmp_path / "run", report, dry_run=False)
        summary = "".join(stream.getvalue().decode() for stream in streams[:2])
    assert expected_summary in summary
    assert "Pass 2 FAILED" not in summary


def _manifest(run: Path, count: int = 2) -> tuple[Path, tuple[MigrationImageTask, ...]]:
    """Publish one immutable manifest and return its canonical header path."""
    tasks = tuple(_task(run, index) for index in range(count))
    write_migration_manifest(
        run,
        generation=_GENERATION,
        scientific_output=deliverables_dir(run),
        tasks=tasks,
    )
    return phenotypic_cache_dir(run) / "migration_manifest.json", tasks


def _record_path(task: MigrationImageTask) -> Path:
    """Return the RECORD the migrator publishes for one task.

    **Not `task.marker_path`, and that field is not wrong.** It means what it
    has always meant -- the legacy `image_complete/` marker the task reads as
    *input* -- and the tasks below still build it that way on purpose, because
    a legacy tree is what migrate consumes.

    What moved is the *output*: after P3's clean break `publish_image_success`
    writes `images/<ds>/<stem>.json`, and every digest in the seal chain
    (`MigrationImageResult.marker_digest` and the six controller
    re-derivations in `_cli_migrate_manifest`) is over that file. A fixture
    digesting the marker would agree with nothing production computes.

    Derived from `store_path` the same way `_image_result` already derives
    `output_dir`, so there is one spelling of that derivation to be wrong.
    """
    from phenotypic.sdk_ import image_record_path

    return image_record_path(
        task.store_path.parents[3], task.dataset, task.stem
    )


def _image_result(
    task: MigrationImageTask,
    *,
    lifecycle_epoch: str = "migration",
) -> MigrationImageResult:
    """Install real migrated artifacts and their canonical success marker."""
    output_dir = task.store_path.parents[3]
    task.store_path.mkdir(parents=True, exist_ok=True)
    (task.store_path / "zarr.json").write_bytes(b'{"zarr_format":3}')
    embedded = task.store_path / MEASUREMENT_TABLE_RELATIVE_PATH
    embedded.parent.mkdir(parents=True, exist_ok=True)
    embedded.write_bytes(b"embedded measurements")
    task.overlay_path.parent.mkdir(parents=True, exist_ok=True)
    task.overlay_path.write_bytes(b"canonical overlay")
    publish_image_success(
        output_dir,
        work_id=_migration_work_id(task.dataset, task.stem),
        dataset=task.dataset,
        relative_image_path=f"{task.dataset}/{task.stem}",
        image_stem=task.stem,
        mode="full",
        attempt_id="migration",
        lifecycle_epoch=lifecycle_epoch,
        artifacts={
            "store": task.store_path,
            "overlay": task.overlay_path,
            "measurements": embedded,
        },
    )
    return MigrationImageResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=_migration_work_id(task.dataset, task.stem),
        converted=True,
        table_installed=True,
        overlay_rendered=True,
        marker_digest=hashlib.sha256(
            _record_path(task).read_bytes()
        ).hexdigest(),
        skipped=False,
    )


class _Guard:
    """Record every authority commit that enters the generation fence."""

    def __init__(self) -> None:
        self.entries = 0

    @contextmanager
    def __call__(self) -> Iterator[None]:
        self.entries += 1
        yield


class _RejectingGuard(_Guard):
    """Raise at the exact authority commit point."""

    @contextmanager
    def __call__(self) -> Iterator[None]:
        self.entries += 1
        raise RuntimeError("generation guard rejected publication")
        yield


class _RejectOnEntry(_Guard):
    """Reject one exact terminal publication boundary."""

    def __init__(self, reject_at: int) -> None:
        super().__init__()
        self.reject_at = reject_at

    @contextmanager
    def __call__(self) -> Iterator[None]:
        self.entries += 1
        if self.entries == self.reject_at:
            raise RuntimeError(
                f"generation guard rejected publication {self.reject_at}"
            )
        yield


def _publish_image_statuses(
    run: Path,
    manifest_path: Path,
    tasks: tuple[MigrationImageTask, ...],
    *,
    guard: _Guard | None = None,
) -> tuple[MigrationImageResult, ...]:
    """Publish complete task status authority for every fixture index."""
    results = tuple(_image_result(task) for task in tasks)
    for result in results:
        publish_migration_task_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            metadata_terminal_digest=_METADATA_DIGEST,
            result=result,
            commit_guard=guard,
        )
    return results


def test_image_statuses_and_seal_bind_exact_terminal_authority(tmp_path: Path) -> None:
    """A clean seal covers every manifest identity and the current marker bytes."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run)
    guard = _Guard()
    _publish_image_statuses(run, manifest_path, tasks, guard=guard)

    seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
        commit_guard=guard,
    )

    assert seal.clean is True
    assert seal.failures == ()
    assert seal.generation == _GENERATION
    assert seal.manifest_digest
    assert seal.ordered_status_digest
    assert seal.metadata_terminal_digest == _METADATA_DIGEST
    assert guard.entries == len(tasks) + 1


@pytest.mark.parametrize(
    ("case", "failure_fragment"),
    (
        ("missing", "missing status index 1"),
        ("duplicate", "duplicate status index 0"),
        ("extra", "extra status index 7"),
        ("generation", "generation"),
        ("work_id", "work ID"),
        ("metadata", "metadata digest"),
        ("missing_marker", "marker"),
        ("marker_digest", "marker digest"),
    ),
)
def test_image_seal_refuses_incomplete_or_mismatched_authority(
    tmp_path: Path, case: str, failure_fragment: str
) -> None:
    """No malformed, stale, or incomplete index set can produce a clean seal."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run)
    _publish_image_statuses(run, manifest_path, tasks)
    control_root = phenotypic_cache_dir(run)
    first = migration_task_status_path(control_root, _GENERATION, 0)
    second = migration_task_status_path(control_root, _GENERATION, 1)

    if case == "missing":
        second.unlink()
    elif case == "duplicate":
        duplicate = first.with_name("duplicate.json")
        duplicate.write_bytes(first.read_bytes())
    elif case == "extra":
        payload = json.loads(first.read_text(encoding="utf-8"))
        payload["index"] = 7
        first.with_name("extra.json").write_text(json.dumps(payload), encoding="utf-8")
    elif case == "missing_marker":
        _record_path(tasks[0]).unlink()
    elif case == "marker_digest":
        _record_path(tasks[0]).write_bytes(b"new marker bytes")
    else:
        payload = json.loads(first.read_text(encoding="utf-8"))
        replacements = {
            "generation": "stale-generation",
            "work_id": "wrong-work-id",
            "metadata": "sha256:" + "b" * 64,
        }
        field = {
            "generation": "generation",
            "work_id": "work_id",
            "metadata": "metadata_terminal_digest",
        }[case]
        payload[field] = replacements[case]
        first.write_text(json.dumps(payload), encoding="utf-8")

    seal = seal_migration_image_stage(
        control_root,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )

    assert seal.clean is False
    assert any(failure_fragment in failure for failure in seal.failures)


def test_status_publication_refuses_result_marker_payload_mismatch(
    tmp_path: Path,
) -> None:
    """A result cannot publish after its marker bytes change."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    result = _image_result(tasks[0])
    _record_path(tasks[0]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="marker digest"):
        publish_migration_task_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            metadata_terminal_digest=_METADATA_DIGEST,
            result=result,
        )


def test_status_publication_requires_current_semantic_marker_authority(
    tmp_path: Path,
) -> None:
    """Unchanged marker bytes cannot authorize a mutated declared artifact."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    result = _image_result(tasks[0])
    (tasks[0].store_path / "zarr.json").write_bytes(b"mutated store root")

    with pytest.raises(ValueError, match="semantic marker authority"):
        publish_migration_task_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            metadata_terminal_digest=_METADATA_DIGEST,
            result=result,
        )


@pytest.mark.parametrize("mutation", ("store", "overlay"))
def test_image_seal_revalidates_current_semantic_artifact_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Artifact mutation/removal after status makes the image seal unclean."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    _publish_image_statuses(run, manifest_path, tasks)
    task = tasks[0]
    if mutation == "store":
        (task.store_path / "zarr.json").write_bytes(b"mutated store root")
    else:
        task.overlay_path.unlink()

    seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )

    assert seal.clean is False
    assert any("semantic marker authority" in failure for failure in seal.failures)


def _state(path: Path | None, data: bytes | None) -> SourceArtifactState:
    """Return one literal source state for later reclaim authority tests."""
    return SourceArtifactState(
        path=path,
        exists=data is not None,
        size=None if data is None else len(data),
        sha256=None if data is None else hashlib.sha256(data).hexdigest(),
    )


def _unused_reclaim_result(task: MigrationImageTask) -> ReclaimResult:
    """Keep the canonical Task 3 result type imported before reclaim RED tests."""
    absent = _state(task.hdf_path, None)
    return ReclaimResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=_migration_work_id(task.dataset, task.stem),
        marker_digest="",
        intended_deletions=(),
        hdf_prestate=absent,
        parquet_prestate=_state(task.measurement_path, None),
        observed_poststate=(absent, _state(task.measurement_path, None)),
        deleted_paths=(),
        retained_paths=(),
        reason=None,
    )


def _deleted_reclaim_result(
    task: MigrationImageTask, marker_digest: str
) -> ReclaimResult:
    """Delete exact source bytes and return matching before/after evidence."""
    assert task.hdf_path is not None
    assert task.measurement_path is not None
    hdf_bytes = f"hdf:{task.index}".encode()
    parquet_bytes = f"parquet:{task.index}".encode()
    task.hdf_path.parent.mkdir(parents=True, exist_ok=True)
    task.measurement_path.parent.mkdir(parents=True, exist_ok=True)
    task.hdf_path.write_bytes(hdf_bytes)
    task.measurement_path.write_bytes(parquet_bytes)
    hdf_prestate = _state(task.hdf_path, hdf_bytes)
    parquet_prestate = _state(task.measurement_path, parquet_bytes)
    task.hdf_path.unlink()
    task.measurement_path.unlink()
    return ReclaimResult(
        index=task.index,
        dataset=task.dataset,
        stem=task.stem,
        work_id=_migration_work_id(task.dataset, task.stem),
        marker_digest=marker_digest,
        intended_deletions=(task.hdf_path, task.measurement_path),
        hdf_prestate=hdf_prestate,
        parquet_prestate=parquet_prestate,
        observed_poststate=(
            _state(task.hdf_path, None),
            _state(task.measurement_path, None),
        ),
        deleted_paths=(task.hdf_path, task.measurement_path),
        retained_paths=(),
        reason=None,
    )


def _publish_reclaim_statuses(
    run: Path,
    manifest_path: Path,
    tasks: tuple[MigrationImageTask, ...],
    image_results: tuple[MigrationImageResult, ...],
    *,
    guard: _Guard | None = None,
) -> tuple[ReclaimResult, ...]:
    """Delete every fixture source and publish the exact reclaim results."""
    results = tuple(
        _deleted_reclaim_result(task, image_result.marker_digest)
        for task, image_result in zip(tasks, image_results, strict=True)
    )
    for result in results:
        publish_migration_reclaim_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            result=result,
            commit_guard=guard,
        )
    return results


def test_reclaim_statuses_and_seal_bind_exact_source_transition(
    tmp_path: Path,
) -> None:
    """A clean reclaim seal binds all ordered results and absent poststates."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run)
    image_results = _publish_image_statuses(run, manifest_path, tasks)
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    guard = _Guard()
    _publish_reclaim_statuses(
        run, manifest_path, tasks, image_results, guard=guard
    )

    seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
        commit_guard=guard,
    )

    assert seal is not None
    assert seal.clean is True
    assert seal.failures == ()
    assert seal.deletion_requested is True
    assert seal.manifest_digest == image_seal.manifest_digest
    assert seal.ordered_reclaim_status_digest
    assert guard.entries == len(tasks) + 1


def test_reclaim_authority_is_absent_when_deletion_is_disabled(
    tmp_path: Path,
) -> None:
    """Keeping sources requires no reclaim statuses and publishes no seal."""
    run = tmp_path / "run"
    manifest_path, _ = _manifest(run)

    seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=False,
    )

    assert seal is None
    generation_dir = (
        phenotypic_cache_dir(run) / "migration_generations" / _GENERATION
    )
    assert not list(generation_dir.glob("reclaim*"))


@pytest.mark.parametrize(
    ("case", "failure_fragment"),
    (
        ("missing", "missing reclaim status index 1"),
        ("duplicate", "duplicate reclaim status index 0"),
        ("extra", "extra reclaim status index 8"),
        ("generation", "generation"),
        ("work_id", "work ID"),
        ("marker", "marker digest"),
        ("prestate", "source prestate"),
        ("poststate", "poststate"),
    ),
)
def test_reclaim_seal_refuses_incomplete_or_mismatched_authority(
    tmp_path: Path, case: str, failure_fragment: str
) -> None:
    """Reclaim certification rejects every stale identity and state mismatch."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run)
    image_results = _publish_image_statuses(run, manifest_path, tasks)
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    _publish_reclaim_statuses(run, manifest_path, tasks, image_results)
    control_root = phenotypic_cache_dir(run)
    first = migration_reclaim_status_path(control_root, _GENERATION, 0)
    second = migration_reclaim_status_path(control_root, _GENERATION, 1)

    if case == "missing":
        second.unlink()
    elif case == "duplicate":
        first.with_name("duplicate.json").write_bytes(first.read_bytes())
    elif case == "extra":
        payload = json.loads(first.read_text(encoding="utf-8"))
        payload["index"] = 8
        first.with_name("extra.json").write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = json.loads(first.read_text(encoding="utf-8"))
        if case == "generation":
            payload["generation"] = "stale-generation"
        elif case == "work_id":
            payload["work_id"] = "wrong-work-id"
        elif case == "marker":
            payload["marker_payload_digest"] = "f" * 64
        elif case == "prestate":
            payload["hdf_prestate"]["path"] = str(run / "wrong.h5")
        else:
            payload["observed_poststate"][0] = {
                "path": str(tasks[0].hdf_path),
                "exists": True,
                "size": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            }
        first.write_text(json.dumps(payload), encoding="utf-8")

    seal = seal_migration_reclaim_stage(
        control_root,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
    )

    assert seal is not None
    assert seal.clean is False
    assert any(failure_fragment in failure for failure in seal.failures)


def test_reclaim_noop_after_unclean_image_seal_is_not_clean(
    tmp_path: Path,
) -> None:
    """An unclean image barrier retains sources and cannot pass reclaim."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    image_results = _publish_image_statuses(run, manifest_path, tasks)
    task = tasks[0]
    assert task.hdf_path is not None
    assert task.measurement_path is not None
    task.hdf_path.parent.mkdir(parents=True, exist_ok=True)
    task.measurement_path.parent.mkdir(parents=True, exist_ok=True)
    task.hdf_path.write_bytes(b"retained hdf")
    task.measurement_path.write_bytes(b"retained parquet")
    hdf_state = _state(task.hdf_path, b"retained hdf")
    parquet_state = _state(task.measurement_path, b"retained parquet")
    result = ReclaimResult(
        index=0,
        dataset=task.dataset,
        stem=task.stem,
        work_id=image_results[0].work_id,
        marker_digest=image_results[0].marker_digest,
        intended_deletions=(task.hdf_path, task.measurement_path),
        hdf_prestate=hdf_state,
        parquet_prestate=parquet_state,
        observed_poststate=(hdf_state, parquet_state),
        deleted_paths=(),
        retained_paths=(task.hdf_path, task.measurement_path),
        reason="image seal was not clean; sources retained",
    )
    publish_migration_reclaim_status(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        result=result,
    )
    clean_image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    unclean_image_seal = replace(
        clean_image_seal,
        clean=False,
        failures=("simulated image failure",),
    )

    seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=unclean_image_seal,
    )

    assert seal is not None
    assert seal.clean is False
    assert any("image seal" in failure for failure in seal.failures)
    assert any("retained" in failure for failure in seal.failures)


def test_reclaim_noop_records_missing_marker_without_deleting_sources(
    tmp_path: Path,
) -> None:
    """Every task gets a reclaim disposition even when image authority is absent."""
    from phenotypic._cli._cli_migrate import _retained_reclaim_result

    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    task = tasks[0]
    assert task.hdf_path is not None
    assert task.measurement_path is not None
    task.hdf_path.parent.mkdir(parents=True, exist_ok=True)
    task.measurement_path.parent.mkdir(parents=True, exist_ok=True)
    task.hdf_path.write_bytes(b"retained hdf")
    task.measurement_path.write_bytes(b"retained parquet")
    result = _retained_reclaim_result(run, task, None)

    status_path = publish_migration_reclaim_status(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        result=result,
    )
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    reclaim_seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
    )

    assert status_path.is_file()
    assert reclaim_seal is not None
    assert reclaim_seal.clean is False
    assert task.hdf_path.is_file()
    assert task.measurement_path.is_file()


def test_retained_reclaim_digests_the_record_when_both_files_exist(
    tmp_path: Path,
) -> None:
    """The retained path digests the RECORD, like its producer and validator.

    **This is the branch the two sibling tests cannot reach.** Both
    ``test_reclaim_noop_records_missing_marker_without_deleting_sources`` and
    its neighbour plant the sources but neither a legacy marker nor a record,
    so ``_retained_reclaim_result``'s digest and
    ``_validate_reclaim_result``'s digest are both ``""``,
    ``retained_after_unclean_image`` is True, and the comparison that would
    disagree is never entered. Presence and reachability differ by a branch.

    P3 repointed the validator onto the record and left this producer on
    ``task.marker_path``. On a migrating tree BOTH files exist -- migrate's
    ``_republish_image_marker`` still writes the legacy one -- with different
    bytes, so the seal appended *"reclaim result marker digest does not match
    current bytes"* and retained the sources for a reason that was an artifact
    of the split repoint rather than the condition that caused it.

    Planting both with deliberately different bytes is what makes this a
    regression test rather than a restatement: before the fix it fails, and it
    fails on the digest sentence specifically.
    """
    from phenotypic._cli._cli_migrate import _retained_reclaim_result
    from phenotypic._cli._cli_migrate_manifest import _validate_reclaim_result

    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    task = tasks[0]
    assert task.hdf_path is not None
    assert task.measurement_path is not None
    task.hdf_path.parent.mkdir(parents=True, exist_ok=True)
    task.measurement_path.parent.mkdir(parents=True, exist_ok=True)
    task.hdf_path.write_bytes(b"retained hdf")
    task.measurement_path.write_bytes(b"retained parquet")

    # BOTH, with different bytes -- the state a migrating tree is actually in.
    record = _record_path(task)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_bytes(b'{"record": "current authority"}')
    task.marker_path.parent.mkdir(parents=True, exist_ok=True)
    task.marker_path.write_bytes(b'{"legacy": "superseded marker"}')
    assert record.read_bytes() != task.marker_path.read_bytes()

    result = _retained_reclaim_result(run, task, None)
    _payload, failures = _validate_reclaim_result(run, task, result)

    # The precise claim: not "no failures" -- other clauses may legitimately
    # fire on this fixture -- but that the DIGEST clause does not, because the
    # producer and the validator now read the same file.
    assert "reclaim result marker digest does not match current bytes" not in (
        failures
    ), f"producer and validator digest different files; failures={failures}"

    assert task.hdf_path.is_file()
    assert task.measurement_path.is_file()


def test_reclaim_status_rejects_tampered_well_formed_prestate(
    tmp_path: Path,
) -> None:
    """A retained source's claimed prestate must equal its current exact bytes."""
    from phenotypic._cli._cli_migrate import _retained_reclaim_result

    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    image_result = _image_result(tasks[0])
    task = tasks[0]
    assert task.hdf_path is not None
    assert task.measurement_path is not None
    task.hdf_path.parent.mkdir(parents=True, exist_ok=True)
    task.measurement_path.parent.mkdir(parents=True, exist_ok=True)
    task.hdf_path.write_bytes(b"retained hdf")
    task.measurement_path.write_bytes(b"retained parquet")
    result = _retained_reclaim_result(run, task, image_result)
    tampered = replace(
        result,
        hdf_prestate=replace(
            result.hdf_prestate,
            size=result.hdf_prestate.size + 1,  # type: ignore[operator]
            sha256=hashlib.sha256(b"different bytes").hexdigest(),
        ),
    )

    with pytest.raises(ValueError, match="prestate.*current source"):
        publish_migration_reclaim_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            result=tampered,
        )


def test_reclaim_status_requires_manifest_ordered_deleted_paths(
    tmp_path: Path,
) -> None:
    """Deleted source classification is an exact ordered transition list."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    image_result = _image_result(tasks[0])
    result = _deleted_reclaim_result(tasks[0], image_result.marker_digest)
    reordered = replace(result, deleted_paths=tuple(reversed(result.deleted_paths)))

    with pytest.raises(ValueError, match="deleted paths"):
        publish_migration_reclaim_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            result=reordered,
        )


def test_reclaim_seal_revalidates_exact_deleted_transition_list(
    tmp_path: Path,
) -> None:
    """A well-formed edit cannot remove a source from sealed deletion evidence."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    image_result = _publish_image_statuses(run, manifest_path, tasks)[0]
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    reclaim_result = _deleted_reclaim_result(
        tasks[0], image_result.marker_digest
    )
    status_path = publish_migration_reclaim_status(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        result=reclaim_result,
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["deleted_paths"] = []
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
    )

    assert seal is not None
    assert seal.clean is False
    assert any("deleted paths" in failure for failure in seal.failures)


def test_reclaim_status_refuses_directory_replacing_claimed_absent_source(
    tmp_path: Path,
) -> None:
    """A directory or unreadable source path is not equivalent to absence."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    image_result = _image_result(tasks[0])
    result = _deleted_reclaim_result(tasks[0], image_result.marker_digest)
    task = tasks[0]
    assert task.hdf_path is not None
    task.hdf_path.mkdir()

    with pytest.raises((IsADirectoryError, ValueError)):
        publish_migration_reclaim_status(
            phenotypic_cache_dir(run),
            manifest_path=manifest_path,
            expected_scientific_output=deliverables_dir(run),
            generation=_GENERATION,
            result=result,
        )
    assert not migration_reclaim_status_path(
        phenotypic_cache_dir(run), _GENERATION, 0
    ).exists()


def test_reclaim_seal_rejects_image_seal_stale_before_deletion_evidence(
    tmp_path: Path,
) -> None:
    """Fresh reclaim evidence cannot revive an image seal whose marker changed."""
    run = tmp_path / "run"
    manifest_path, tasks = _manifest(run, count=1)
    _publish_image_statuses(run, manifest_path, tasks)
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    task = tasks[0]
    _record_path(task).write_bytes(b"new authoritative marker bytes")
    marker_digest = hashlib.sha256(
        _record_path(task).read_bytes()
    ).hexdigest()
    reclaim_result = _deleted_reclaim_result(task, marker_digest)
    publish_migration_reclaim_status(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        result=reclaim_result,
    )

    reclaim_seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
    )

    assert reclaim_seal is not None
    assert reclaim_seal.clean is False
    assert any("image seal" in failure for failure in reclaim_seal.failures)


def _install_completion_fixture(run: Path) -> None:
    """Install real marker/state/core artifacts for terminal publisher tests."""
    measurement = run / "results" / "ds" / "measurements" / "image.parquet"
    measurement.parent.mkdir(parents=True)
    measurement.write_bytes(b"measurement")
    publish_image_success(
        run,
        work_id="work-a",
        dataset="ds",
        relative_image_path="ds/image.tif",
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
            pipeline_path=run / "pipeline.json",
            input_path=run / "input",
            output_dir=run,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={"ds": DatasetState(initial_images={"image.tif"})},
            config={
                "success_markers_required": True,
                "work_ids": {"ds": {"image.tif": "work-a"}},
                "processing_generation": "generation",
                "pipeline_sha256": "pipeline",
            },
        ),
        run,
    )
    for index, path in enumerate(
        (
            master_measurements_csv_path(run),
            master_measurements_parquet_path(run),
            measurements_csv_path(run),
            measurements_parquet_path(run),
        )
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"core-{index}".encode())


def test_canonical_metadata_view_replaces_only_inside_commit_guard(
    tmp_path: Path,
) -> None:
    """A stale generation cannot publish even a fully staged metadata view."""
    run = tmp_path / "run"
    snapshot = run / "deliverables" / "metadata.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("Strain\nWT\n", encoding="utf-8")
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        emit_canonical_metadata_view(run, commit_guard=guard)

    assert guard.entries == 1
    assert not (snapshot.parent / "metadata.canonical.csv").exists()


def test_aggregate_snapshot_replaces_only_inside_commit_guard(
    tmp_path: Path,
) -> None:
    """Aggregate authority cannot escape a rejected generation fence."""
    run = tmp_path / "run"
    _install_completion_fixture(run)
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        publish_aggregate_snapshot(run, commit_guard=guard)

    assert guard.entries == 1
    assert not aggregate_publication_marker_path(run).exists()


def test_run_completion_replaces_only_inside_commit_guard(tmp_path: Path) -> None:
    """Run completion remains absent when its generation loses ownership."""
    run = tmp_path / "run"
    _install_completion_fixture(run)
    publish_aggregate_snapshot(run)
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        publish_run_completion_evidence(
            run,
            execution_epoch="generation",
            commit_guard=guard,
        )

    assert guard.entries == 1
    assert not run_completion_marker_path(run).exists()


def test_compatible_run_completion_still_validates_commit_guard(
    tmp_path: Path,
) -> None:
    """An idempotent completion return cannot bypass generation revocation."""
    run = tmp_path / "run"
    _install_completion_fixture(run)
    publish_aggregate_snapshot(run)
    completion = publish_run_completion_evidence(
        run,
        execution_epoch="generation",
    )
    before = completion.read_bytes()
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        publish_run_completion_evidence(
            run,
            execution_epoch="generation",
            commit_guard=guard,
        )

    assert guard.entries == 1
    assert completion.read_bytes() == before


def test_terminal_authority_invalidation_unlinks_both_markers_under_one_guard(
    tmp_path: Path,
) -> None:
    """No migration mutation can inherit aggregate or run-completion authority."""
    run = tmp_path / "run"
    aggregate = aggregate_publication_marker_path(run)
    completion = run_completion_marker_path(run)
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stale aggregate")
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_bytes(b"stale completion")
    guard = _Guard()

    invalidate_migration_terminal_authority(run, commit_guard=guard)

    assert guard.entries == 1
    assert not aggregate.exists()
    assert not completion.exists()


def test_typed_terminal_status_is_generation_fenced_and_durable(
    tmp_path: Path,
) -> None:
    """Attempt status records the typed failure before lifecycle closure."""
    run = tmp_path / "run"
    guard = _Guard()
    report = MigrationReport(
        publication_failures=((run, "aggregate failed"),)
    )

    path = publish_migration_terminal_status(
        run,
        generation=_GENERATION,
        succeeded=False,
        failure_category="aggregate",
        reason="aggregate failed",
        report=report,
        commit_guard=guard,
    )

    assert path == migration_terminal_status_path(
        phenotypic_cache_dir(run), _GENERATION
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generation"] == _GENERATION
    assert payload["status"] == "failed"
    assert payload["failure_category"] == "aggregate"
    assert payload["reason"] == "aggregate failed"
    assert payload["report"]["publication_failures"]
    assert guard.entries == 1


def test_typed_terminal_status_refuses_unknown_failure_category(
    tmp_path: Path,
) -> None:
    """Terminal failure categories are a closed value set, not free text."""
    run = tmp_path / "run"
    report = MigrationReport(failed=((run, "failed"),))

    with pytest.raises(ValueError, match="failure category"):
        publish_migration_terminal_status(
            run,
            generation=_GENERATION,
            succeeded=False,
            failure_category="typo",
            reason="failed",
            report=report,
        )


def _clean_authority_fixture(
    run: Path,
) -> tuple[
    Path,
    tuple[MigrationImageTask, ...],
    MetadataPassResult,
    Any,
]:
    """Return real manifest/image-seal authority plus typed metadata authority."""
    manifest_path, tasks = _manifest(run, count=1)
    _publish_image_statuses(run, manifest_path, tasks)
    image_seal = seal_migration_image_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    authority = SimpleNamespace(terminal_receipt_digest=_METADATA_DIGEST)
    metadata = MetadataPassResult(
        headers_migrated=0,
        failures=(),
        authority=authority,
    )
    return manifest_path, tasks, metadata, image_seal


def _patch_successful_terminal_publishers(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    """Replace only heavyweight final publishers with ordered successful seams."""
    from phenotypic._cli import _cli_migrate as subject

    monkeypatch.setattr(
        subject,
        "metadata_migration_authority",
        lambda *_: SimpleNamespace(terminal_receipt_digest=_METADATA_DIGEST),
    )
    monkeypatch.setattr(
        subject,
        "emit_canonical_metadata_view",
        lambda *_, **__: events.append("canonical") or Path("canonical.csv"),
    )
    monkeypatch.setattr(
        subject,
        "_publish_migration_aggregate",
        lambda *_, **__: events.append("aggregate"),
    )
    monkeypatch.setattr(
        subject,
        "publish_run_completion_evidence",
        lambda *_, **__: events.append("completion") or Path("completion.json"),
    )
    monkeypatch.setattr(
        subject,
        "valid_run_completion",
        lambda *_: events.append("completion_validate") or {"status": "complete"},
    )


def test_finalizer_publishes_science_then_status_then_closes_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical view, aggregate, and completion precede durable terminal closure."""
    run = tmp_path / "run"
    manifest_path, _, metadata, image_seal = _clean_authority_fixture(run)
    events: list[str] = []
    _patch_successful_terminal_publishers(monkeypatch, events)
    from phenotypic._cli import _cli_migrate as subject

    def terminal(*_: Any, **kwargs: Any) -> Path:
        assert kwargs["succeeded"] is True
        events.append("terminal")
        return run / "terminal.json"

    def close(*_: Any, **kwargs: Any) -> None:
        assert kwargs["succeeded"] is True
        events.append("close")

    monkeypatch.setattr(subject, "publish_migration_terminal_status", terminal)
    monkeypatch.setattr(subject, "close_migration_generation", close)

    report = finalize_migration_attempt(
        run,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_pass=metadata,
        image_seal=image_seal,
        reclaim_seal=None,
        deletion_requested=False,
        dry_run=False,
        report=MigrationReport(converted=1),
        image_failures=(),
        reclaim_failures=(),
        commit_guard=_Guard(),
    )

    assert report.ok
    assert events == [
        "canonical",
        "aggregate",
        "completion",
        "completion_validate",
        "terminal",
        "close",
    ]


def test_finalizer_rejects_marker_changed_after_clean_image_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seal cannot authorize science after its bound marker bytes change."""
    run = tmp_path / "run"
    manifest_path, tasks, metadata, image_seal = _clean_authority_fixture(run)
    _record_path(tasks[0]).write_bytes(b"marker changed after seal")
    events: list[str] = []
    _patch_successful_terminal_publishers(monkeypatch, events)
    from phenotypic._cli import _cli_migrate as subject

    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        subject,
        "publish_migration_terminal_status",
        lambda *_, **kwargs: observed.update(kwargs) or run / "terminal.json",
    )
    monkeypatch.setattr(subject, "close_migration_generation", lambda *_, **__: None)

    report = finalize_migration_attempt(
        run,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_pass=metadata,
        image_seal=image_seal,
        reclaim_seal=None,
        deletion_requested=False,
        dry_run=False,
        report=MigrationReport(),
        image_failures=(),
        reclaim_failures=(),
        commit_guard=_Guard(),
    )

    assert not report.ok
    assert observed["failure_category"] == "image_seal"
    assert "canonical" not in events


def test_finalizer_rejects_source_recreated_after_clean_reclaim_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean reclaim seal is stale when any claimed-absent source reappears."""
    run = tmp_path / "run"
    manifest_path, tasks, metadata, image_seal = _clean_authority_fixture(run)
    task = tasks[0]
    marker_digest = hashlib.sha256(
        _record_path(task).read_bytes()
    ).hexdigest()
    reclaim_result = _deleted_reclaim_result(task, marker_digest)
    publish_migration_reclaim_status(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        result=reclaim_result,
    )
    reclaim_seal = seal_migration_reclaim_stage(
        phenotypic_cache_dir(run),
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        deletion_requested=True,
        image_seal=image_seal,
    )
    assert reclaim_seal is not None and reclaim_seal.clean
    assert task.hdf_path is not None
    task.hdf_path.write_bytes(b"recreated after reclaim seal")
    events: list[str] = []
    _patch_successful_terminal_publishers(monkeypatch, events)
    from phenotypic._cli import _cli_migrate as subject

    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        subject,
        "publish_migration_terminal_status",
        lambda *_, **kwargs: observed.update(kwargs) or run / "terminal.json",
    )
    monkeypatch.setattr(subject, "close_migration_generation", lambda *_, **__: None)

    report = finalize_migration_attempt(
        run,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_pass=metadata,
        image_seal=image_seal,
        reclaim_seal=reclaim_seal,
        deletion_requested=True,
        dry_run=False,
        report=MigrationReport(),
        image_failures=(),
        reclaim_failures=(),
        commit_guard=_Guard(),
    )

    assert not report.ok
    assert observed["failure_category"] == "reclaim"
    assert "canonical" not in events


@pytest.mark.parametrize(
    "failure_case",
    (
        "metadata",
        "provenance",
        "image",
        "image_seal",
        "reclaim_noop",
        "reclaim",
        "aggregate",
        "completion",
    ),
)
def test_finalizer_publishes_typed_failure_before_exact_lifecycle_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
) -> None:
    """Every terminal failure is typed, durable, and closes with the same reason."""
    run = tmp_path / "run"
    manifest_path, _, metadata, image_seal = _clean_authority_fixture(run)
    events: list[str] = []
    _patch_successful_terminal_publishers(monkeypatch, events)
    from phenotypic._cli import _cli_migrate as subject

    report = MigrationReport()
    image_failures: tuple[tuple[Path, str], ...] = ()
    reclaim_failures: tuple[tuple[Path, str], ...] = ()
    reclaim_seal: MigrationReclaimSeal | None = None
    deletion_requested = failure_case in {"reclaim_noop", "reclaim"}
    if failure_case == "metadata":
        metadata = MetadataPassResult(
            headers_migrated=0,
            failures=((run, "metadata failed"),),
            authority=None,
        )
    elif failure_case == "provenance":
        report = MigrationReport(
            provenance_failures=((run / "plate.ome.zarr", "future schema"),)
        )
    elif failure_case == "image":
        image_failures = ((run / "image.h5", "image failed"),)
    elif failure_case == "image_seal":
        image_seal = replace(
            image_seal, clean=False, failures=("image seal failed",)
        )
    elif failure_case == "reclaim_noop":
        reclaim_seal = MigrationReclaimSeal(
            generation=_GENERATION,
            manifest_digest=image_seal.manifest_digest,
            ordered_reclaim_status_digest="a" * 64,
            deletion_requested=True,
            clean=False,
            failures=("reclaim status index 0 retained sources",),
            seal_path=(
                phenotypic_cache_dir(run)
                / "migration_generations"
                / _GENERATION
                / "reclaim_seal.json"
            ),
        )
    elif failure_case == "reclaim":
        reclaim_failures = ((run / "source.h5", "unlink failed"),)
    elif failure_case == "aggregate":
        monkeypatch.setattr(
            subject,
            "_publish_migration_aggregate",
            lambda *_, **__: (_ for _ in ()).throw(RuntimeError("aggregate failed")),
        )
    elif failure_case == "completion":
        monkeypatch.setattr(subject, "valid_run_completion", lambda *_: None)

    observed: dict[str, Any] = {}

    def terminal(*_: Any, **kwargs: Any) -> Path:
        events.append("terminal")
        observed.update(kwargs)
        return run / "terminal.json"

    def close(*_: Any, **kwargs: Any) -> None:
        events.append("close")
        assert kwargs["reason"] == observed["reason"]
        assert kwargs["succeeded"] is False

    monkeypatch.setattr(subject, "publish_migration_terminal_status", terminal)
    monkeypatch.setattr(subject, "close_migration_generation", close)

    final = finalize_migration_attempt(
        run,
        manifest_path=manifest_path,
        expected_scientific_output=deliverables_dir(run),
        generation=_GENERATION,
        metadata_pass=metadata,
        image_seal=image_seal,
        reclaim_seal=reclaim_seal,
        deletion_requested=deletion_requested,
        dry_run=False,
        report=report,
        image_failures=image_failures,
        reclaim_failures=reclaim_failures,
        commit_guard=_Guard(),
    )

    assert not final.ok
    assert observed["failure_category"] == failure_case
    assert events[-2:] == ["terminal", "close"]


def test_closed_generation_allows_new_attempt_without_reusing_old_token(
    tmp_path: Path,
) -> None:
    """A terminal migration releases lifecycle ownership for a fresh retry."""
    run = tmp_path / "run"
    initialize_slurm_lifecycle(run, generation="first", mode="migrate")
    append_lifecycle_entry(
        run,
        generation="first",
        token="image-0",
        role="migration-image",
        status="submitted",
        job_id="123",
    )
    terminal = publish_migration_terminal_status(
        run,
        generation="first",
        succeeded=True,
        failure_category=None,
        reason=None,
        report=MigrationReport(),
    )
    assert terminal.is_file()
    close_migration_generation(
        run, generation="first", succeeded=True, reason=None
    )
    assert not generation_is_active(run, "first")

    initialize_slurm_lifecycle(run, generation="second", mode="migrate")

    assert generation_is_active(run, "second")
    assert ledger_job_for_token(run, "second", "image-0") is None


def test_failed_generation_closes_with_exact_terminal_reason(tmp_path: Path) -> None:
    """Failure closure preserves the same reason carried by durable status."""
    run = tmp_path / "run"
    initialize_slurm_lifecycle(run, generation=_GENERATION, mode="migrate")
    publish_migration_terminal_status(
        run,
        generation=_GENERATION,
        succeeded=False,
        failure_category="image",
        reason="marker mismatch",
        report=MigrationReport(),
        commit_guard=lambda: (
            __import__(
                "phenotypic._cli._cli_slurm_lifecycle",
                fromlist=["generation_publication_guard"],
            ).generation_publication_guard(run, _GENERATION)
        ),
    )
    close_migration_generation(
        run,
        generation=_GENERATION,
        succeeded=False,
        reason="marker mismatch",
    )

    lifecycle = load_slurm_lifecycle(run)
    assert lifecycle is not None
    assert lifecycle["active"] is False
    assert lifecycle["terminal_status"] == "failed"
    assert lifecycle["terminal_error"] == "marker mismatch"


def test_aggregate_core_outputs_replace_only_inside_commit_guard(
    tmp_path: Path,
) -> None:
    """The aggregate builder cannot write its master after losing generation."""
    import polars as pl

    from phenotypic._cli._cli_output_manager import aggregate_measurements

    run = tmp_path / "run"
    source = run / "results" / "ds" / "measurements" / "image.parquet"
    source.parent.mkdir(parents=True)
    pl.DataFrame(
        {"Object_Label": [1], "Size_Area": [25.0]}
    ).write_parquet(source)
    publish_image_success(
        run,
        work_id="work-a",
        dataset="ds",
        relative_image_path="ds/image.tif",
        image_stem="image",
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="epoch",
        artifacts={"measurements": source},
    )
    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=run / "pipeline.json",
            input_path=run / "input",
            output_dir=run,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={"ds": DatasetState(initial_images={"image.tif"})},
            config={
                "success_markers_required": True,
                "work_ids": {"ds": {"image.tif": "work-a"}},
                "processing_generation": "generation",
                "pipeline_sha256": "pipeline",
            },
        ),
        run,
    )
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        aggregate_measurements(
            run,
            ["ds"],
            no_qc=True,
            commit_guard=guard,
        )

    assert guard.entries == 1
    assert not master_measurements_csv_path(run).exists()
    assert not master_measurements_parquet_path(run).exists()


def test_measurement_mirror_does_not_swallow_generation_rejection(
    tmp_path: Path,
) -> None:
    """Mirror best-effort handling cannot convert a stale fence into success."""
    from phenotypic._cli._cli_output_manager import _seed_measurements

    run = tmp_path / "run"
    guard = _RejectingGuard()

    with pytest.raises(RuntimeError, match="guard rejected"):
        _seed_measurements(
            run,
            pl.DataFrame({"Object_Label": [1], "Size_Area": [25.0]}),
            commit_guard=guard,
        )

    assert guard.entries == 1
    assert not measurements_csv_path(run).exists()
    assert not measurements_parquet_path(run).exists()


@pytest.mark.parametrize(
    ("reject_at", "expected_events"),
    (
        (1, []),
        (4, ["legacy_qc"]),
        (6, ["legacy_qc", "rembi", "split"]),
    ),
)
def test_terminal_auxiliary_writers_stop_at_early_middle_and_late_revocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject_at: int,
    expected_events: list[str],
) -> None:
    """Every terminal writer validates the generation before its side effect."""
    import phenotypic.sdk_ as sdk
    from phenotypic._cli import _cli_error_outputs
    from phenotypic._cli import _cli_output_manager as subject
    from phenotypic.sdk_ import _rembi_manifest

    run = tmp_path / f"run-{reject_at}"
    events: list[str] = []
    monkeypatch.setattr(
        sdk,
        "migrate_legacy_qc",
        lambda *_: events.append("legacy_qc"),
    )
    monkeypatch.setattr(
        _rembi_manifest,
        "write_rembi_manifest",
        lambda *_args, **_kwargs: events.append("rembi"),
    )
    monkeypatch.setattr(
        subject,
        "split_master_by_feature",
        lambda *_args, **_kwargs: events.append("split") or {},
    )
    monkeypatch.setattr(
        _cli_error_outputs,
        "reemit_error_deliverables",
        lambda *_args, **_kwargs: events.append("errors"),
    )
    guard = _RejectOnEntry(reject_at)

    with pytest.raises(RuntimeError, match=f"publication {reject_at}"):
        subject.finalize_post_master_outputs(
            run,
            pl.DataFrame({"Object_Label": [1], "Size_Area": [25.0]}),
            pipeline=None,
            no_qc=True,
            commit_guard=guard,
        )

    assert events == expected_events
    assert "errors" not in events


def test_analysis_registry_write_is_generation_guarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis registry mutation cannot precede its generation validation."""
    import phenotypic.sdk_ as sdk
    from phenotypic._cli import _cli_output_manager as subject
    from phenotypic.plotting import _pipeline as plotting_pipeline
    from phenotypic.sdk_ import _rembi_manifest

    run = tmp_path / "run"
    events: list[str] = []
    monkeypatch.setattr(sdk, "migrate_legacy_qc", lambda *_: None)
    monkeypatch.setattr(
        _rembi_manifest, "write_rembi_manifest", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        subject, "_persist_pipeline_to_output_dir", lambda *_: None
    )
    monkeypatch.setattr(
        subject,
        "_emit_analysis_outputs",
        lambda *_: SimpleNamespace(
            analysis_id="analysis",
            table=pl.DataFrame({"value": [1]}),
            producer="test",
            artifacts=(),
            manifest_entry={},
        ),
    )

    class Registry:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def register(self, *_args: object, **_kwargs: object) -> None:
            events.append("registry")

    class Coordinator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def emit_measurements(self, *_args: object, **_kwargs: object) -> None:
            pass

        def emit_analyses(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(plotting_pipeline, "AnalysisRegistry", Registry)
    monkeypatch.setattr(plotting_pipeline, "PlotCoordinator", Coordinator)
    pipeline = SimpleNamespace(get_post=lambda: (), get_qc=lambda: None)
    guard = _RejectOnEntry(8)

    with pytest.raises(RuntimeError, match="publication 8"):
        subject.finalize_post_master_outputs(
            run,
            pl.DataFrame({"Object_Label": [1], "Size_Area": [25.0]}),
            pipeline=pipeline,
            no_qc=True,
            commit_guard=guard,
        )

    assert "registry" not in events


def test_migration_state_uses_manifest_inventory_before_stores_exist(
    tmp_path: Path,
) -> None:
    """Manifest identities establish marker work IDs before image conversion."""
    from phenotypic._cli._cli_migrate import _ensure_migration_processing_state

    run = tmp_path / "run"
    task = _task(run, 0)
    guard = _Guard()

    _ensure_migration_processing_state(
        run,
        tasks=(task,),
        commit_guard=guard,
    )

    state = load_processing_state(run)
    assert state is not None
    assert state.datasets[task.dataset].initial_images == {task.stem}
    assert state.datasets[task.dataset].completed == set()
    assert state.config["work_ids"] == {
        task.dataset: {
            task.stem: _migration_work_id(task.dataset, task.stem),
        }
    }
    assert guard.entries == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_hdf_to_zarr._republish_image_marker rewrites the legacy marker "
        "(:614,:647) and writes no record, so valid_image_success is false "
        "for every migrated image. P7 U-10: republish as a record with "
        "provenance='migrated'. Full rationale beside the shared marker in "
        "tests/unit/sdk_/test_migration_republishes_state.py."
    ),
)
def test_migration_preserves_ome_zarr_source_identity_from_full_run_state(
    tmp_path: Path,
) -> None:
    """A process-store source stays the same completed image during migration."""
    from phenotypic._cli._cli_migrate import _ensure_migration_processing_state
    from phenotypic._cli._cli_migrate_image import _configured_work_id

    run = tmp_path / "run"
    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=run / "pipeline.json",
            input_path=run / "input",
            output_dir=run,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={
                "ds": DatasetState(
                    initial_images={"plate.ome.zarr"},
                    completed={"plate.ome.zarr"},
                )
            },
            config={
                "success_markers_required": True,
                "work_ids": {
                    "ds": {"plate.ome.zarr": "process-store-work-id"}
                },
            },
        ),
        run,
    )
    base = _task(run, 0)
    task = replace(
        base,
        stem="plate",
        hdf_path=None,
        store_path=base.store_path.with_name("plate.ome.zarr"),
        measurement_path=None,
        overlay_path=base.overlay_path.with_name("plate.png"),
        marker_path=base.marker_path.with_name("plate.json"),
    )

    _ensure_migration_processing_state(run, tasks=(task,))

    state = load_processing_state(run)
    assert state is not None
    assert state.datasets["ds"].initial_images == {"plate.ome.zarr"}
    assert state.datasets["ds"].completed == {"plate.ome.zarr"}
    assert state.config["work_ids"] == {
        "ds": {"plate.ome.zarr": "process-store-work-id"}
    }
    assert _configured_work_id(run, "ds", "plate") == "process-store-work-id"


def test_local_migrate_uses_one_inventory_after_guarded_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local orchestration is metadata-first and consumes one task inventory."""
    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    tasks = (_task(run, 0), _task(run, 1))
    stale_aggregate = aggregate_publication_marker_path(run)
    stale_completion = run_completion_marker_path(run)
    stale_aggregate.parent.mkdir(parents=True)
    stale_completion.parent.mkdir(parents=True, exist_ok=True)
    stale_aggregate.write_bytes(b"stale")
    stale_completion.write_bytes(b"stale")
    events: list[str] = []
    monkeypatch.setattr(subject, "new_slurm_generation", lambda: _GENERATION)
    monkeypatch.setattr(
        subject,
        "discover_migration_tasks",
        lambda output: events.append("inventory") or tasks,
    )

    def manifest(*_: Any, **kwargs: Any) -> SimpleNamespace:
        assert not stale_aggregate.exists()
        assert not stale_completion.exists()
        assert tuple(kwargs["tasks"]) == tasks
        events.append("manifest")
        return SimpleNamespace(inventory_digest="manifest")

    monkeypatch.setattr(subject, "write_migration_manifest", manifest)

    authority = SimpleNamespace(terminal_receipt_digest=_METADATA_DIGEST)

    def metadata(*_: Any, **kwargs: Any) -> MetadataPassResult:
        assert kwargs["dry_run"] is False
        assert kwargs["commit_guard"] is not None
        events.append("metadata")
        return MetadataPassResult(0, (), authority)

    monkeypatch.setattr(subject, "run_metadata_pass", metadata)

    def state(*_: Any, **kwargs: Any) -> None:
        assert tuple(kwargs["tasks"]) == tasks
        assert kwargs["commit_guard"] is not None
        events.append("state")

    monkeypatch.setattr(subject, "_ensure_migration_processing_state", state)
    image_results = tuple(_image_result(task) for task in tasks)

    def execute(*_: Any, **kwargs: Any):
        assert kwargs["njobs"] == 2
        assert tuple(kwargs["tasks"]) == tasks
        events.append("images")
        return image_results, ()

    monkeypatch.setattr(subject, "_execute_migration_tasks", execute)
    monkeypatch.setattr(
        subject,
        "publish_migration_task_status",
        lambda *_, **__: events.append("image_status") or Path("status"),
    )
    clean_seal = SimpleNamespace(
        clean=True,
        failures=(),
        generation=_GENERATION,
        manifest_digest="manifest",
        metadata_terminal_digest=_METADATA_DIGEST,
    )
    monkeypatch.setattr(
        subject,
        "seal_migration_image_stage",
        lambda *_, **__: events.append("image_seal") or clean_seal,
    )

    def finalize(*_: Any, **kwargs: Any) -> MigrationReport:
        assert kwargs["image_seal"] is clean_seal
        assert kwargs["deletion_requested"] is False
        events.append("finalizer")
        return kwargs["report"]

    monkeypatch.setattr(subject, "finalize_migration_attempt", finalize)

    report = run_migrate(run, njobs=2)

    assert report.ok
    assert events == [
        "inventory",
        "manifest",
        "metadata",
        "state",
        "images",
        "image_status",
        "image_status",
        "image_seal",
        "finalizer",
    ]


def test_local_migrate_terminalizes_manifest_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once lifecycle ownership exists, setup failure gets status then closure."""
    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    monkeypatch.setattr(subject, "new_slurm_generation", lambda: _GENERATION)
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda *_: ())
    monkeypatch.setattr(
        subject,
        "write_migration_manifest",
        lambda *_, **__: (_ for _ in ()).throw(OSError("manifest disk failure")),
    )
    events: list[str] = []
    observed: dict[str, Any] = {}

    def terminal(*_: Any, **kwargs: Any) -> Path:
        events.append("terminal")
        observed.update(kwargs)
        return run / "terminal.json"

    def close(*_: Any, **kwargs: Any) -> None:
        events.append("close")
        assert kwargs["reason"] == observed["reason"]

    monkeypatch.setattr(subject, "publish_migration_terminal_status", terminal)
    monkeypatch.setattr(subject, "close_migration_generation", close)

    report = run_migrate(run)

    assert not report.ok
    assert observed["failure_category"] == "image_seal"
    assert "manifest disk failure" in observed["reason"]
    assert events == ["terminal", "close"]


def test_local_migrate_terminalizes_invalidation_failure_and_releases_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authority invalidation failure publishes status, closes, then permits retry."""
    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    monkeypatch.setattr(subject, "new_slurm_generation", lambda: _GENERATION)
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda *_: ())
    monkeypatch.setattr(
        subject,
        "invalidate_migration_terminal_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("completion unlink failed")
        ),
    )
    events: list[str] = []
    original_terminal = subject.publish_migration_terminal_status
    original_close = subject.close_migration_generation

    def terminal(*args: Any, **kwargs: Any) -> Path:
        events.append("terminal")
        return original_terminal(*args, **kwargs)

    def close(*args: Any, **kwargs: Any) -> None:
        events.append("close")
        original_close(*args, **kwargs)

    monkeypatch.setattr(subject, "publish_migration_terminal_status", terminal)
    monkeypatch.setattr(subject, "close_migration_generation", close)

    report = run_migrate(run)

    assert not report.ok
    assert events == ["terminal", "close"]
    terminal_status = json.loads(
        migration_terminal_status_path(
            phenotypic_cache_dir(run), _GENERATION
        ).read_text(encoding="utf-8")
    )
    assert terminal_status["failure_category"] == "aggregate"
    assert "completion unlink failed" in terminal_status["reason"]
    assert generation_is_active(run, _GENERATION) is False
    initialize_slurm_lifecycle(run, generation="retry-generation", mode="migrate")
    assert generation_is_active(run, "retry-generation") is True


def test_run_migrate_rerun_uses_new_generation_and_retries_incomplete_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete attempt closes; its rerun skips complete image work."""
    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    tasks = (_task(run, 0), _task(run, 1))
    generations = iter(("attempt-1", "attempt-2"))
    monkeypatch.setattr(subject, "new_slurm_generation", lambda: next(generations))
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda *_: tasks)
    authority = SimpleNamespace(
        terminal_receipt_digest=_METADATA_DIGEST,
        status_path=run / ".phenotypic" / "metadata-status.json",
    )
    monkeypatch.setattr(
        subject,
        "run_metadata_pass",
        lambda *_args, **_kwargs: MetadataPassResult(0, (), authority),
    )
    monkeypatch.setattr(subject, "metadata_migration_authority", lambda *_: authority)
    monkeypatch.setattr(
        subject, "_ensure_migration_processing_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        subject, "emit_canonical_metadata_view", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        subject, "_publish_migration_aggregate", lambda *_args, **_kwargs: None
    )
    completion = run_completion_marker_path(run)
    monkeypatch.setattr(
        subject,
        "publish_run_completion_evidence",
        lambda *_args, **_kwargs: completion,
    )
    monkeypatch.setattr(subject, "valid_run_completion", lambda *_: {"valid": True})
    calls: list[tuple[str, int]] = []
    mutations: list[tuple[str, int]] = []

    def migrate_task(
        _output_dir: Path,
        task: MigrationImageTask,
        **_kwargs: Any,
    ) -> MigrationImageResult:
        lifecycle = load_slurm_lifecycle(run)
        assert lifecycle is not None
        generation = str(lifecycle["generation"])
        calls.append((generation, task.index))
        if generation == "attempt-1" and task.index == 1:
            raise MigrationImageStageError(
                MigrationImageStageFailure(
                    stage="conversion",
                    target=task.hdf_path or task.store_path,
                    reason="conversion interrupted",
                    partial=MigrationImagePartialResult(
                        index=task.index,
                        dataset=task.dataset,
                        stem=task.stem,
                        work_id=_migration_work_id(task.dataset, task.stem),
                        converted=False,
                        table_installed=False,
                        overlay_rendered=False,
                    ),
                )
            )
        if _record_path(task).is_file():
            return MigrationImageResult(
                index=task.index,
                dataset=task.dataset,
                stem=task.stem,
                work_id=_migration_work_id(task.dataset, task.stem),
                converted=False,
                table_installed=False,
                overlay_rendered=False,
                marker_digest=hashlib.sha256(
                    _record_path(task).read_bytes()
                ).hexdigest(),
                skipped=True,
            )
        mutations.append((generation, task.index))
        return _image_result(task, lifecycle_epoch=generation)

    monkeypatch.setattr(subject, "migrate_image_task", migrate_task)

    first = run_migrate(run)
    second = run_migrate(run)

    assert not first.ok
    assert second.ok, (second, calls, mutations)
    assert calls == [
        ("attempt-1", 0),
        ("attempt-1", 1),
        ("attempt-2", 0),
        ("attempt-2", 1),
    ]
    assert mutations == [("attempt-1", 0), ("attempt-2", 1)]
    control_root = phenotypic_cache_dir(run)
    current_manifest = json.loads(
        (control_root / "migration_manifest.json").read_text(encoding="utf-8")
    )
    assert current_manifest["generation"] == "attempt-2"
    assert migration_terminal_status_path(
        control_root, "attempt-1"
    ).is_file()
    assert migration_terminal_status_path(
        control_root, "attempt-2"
    ).is_file()
    assert migration_task_status_path(control_root, "attempt-1", 0).is_file()
    assert not migration_task_status_path(control_root, "attempt-1", 1).exists()
    assert migration_task_status_path(control_root, "attempt-2", 0).is_file()
    assert migration_task_status_path(control_root, "attempt-2", 1).is_file()
    first_seal = json.loads(
        migration_image_seal_path(control_root, "attempt-1").read_text(
            encoding="utf-8"
        )
    )
    second_seal = json.loads(
        migration_image_seal_path(control_root, "attempt-2").read_text(
            encoding="utf-8"
        )
    )
    assert first_seal["clean"] is False
    assert second_seal["clean"] is True
    assert generation_is_active(run, "attempt-1") is False
    assert generation_is_active(run, "attempt-2") is False


def test_local_migrate_dry_run_preserves_terminal_authority_and_control_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run counts tasks without lifecycle, manifest, or scientific writes."""
    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    tasks = (_task(run, 0),)
    aggregate = aggregate_publication_marker_path(run)
    completion = run_completion_marker_path(run)
    aggregate.parent.mkdir(parents=True)
    completion.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_bytes(b"stale aggregate")
    completion.write_bytes(b"stale completion")
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda *_: tasks)
    monkeypatch.setattr(
        subject,
        "run_metadata_pass",
        lambda *_, **__: MetadataPassResult(2, (), None),
    )
    dry_result = MigrationImageResult(
        index=0,
        dataset="ds",
        stem="image-0",
        work_id=_migration_work_id("ds", "image-0"),
        converted=True,
        table_installed=True,
        overlay_rendered=True,
        marker_digest="",
        skipped=False,
    )
    monkeypatch.setattr(
        subject,
        "_execute_migration_tasks",
        lambda *_, **__: ((dry_result,), ()),
    )
    monkeypatch.setattr(
        subject,
        "initialize_slurm_lifecycle",
        lambda *_args, **_kwargs: pytest.fail("dry-run initialized lifecycle"),
    )
    monkeypatch.setattr(
        subject,
        "write_migration_manifest",
        lambda *_args, **_kwargs: pytest.fail("dry-run wrote manifest"),
    )
    monkeypatch.setattr(
        subject,
        "finalize_migration_attempt",
        lambda *_args, **_kwargs: pytest.fail("dry-run finalized"),
    )

    report = run_migrate(run, dry_run=True)

    assert report.converted == 1
    assert report.headers_migrated == 2
    assert aggregate.read_bytes() == b"stale aggregate"
    assert completion.read_bytes() == b"stale completion"
    assert not (
        phenotypic_cache_dir(run) / "migration_manifest.json"
    ).exists()
    assert not (
        phenotypic_cache_dir(run) / "migration_generations"
    ).exists()


def test_parallel_image_executor_uses_joblib_for_more_than_one_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local image stage dispatches the canonical inventory through joblib."""
    import joblib

    from phenotypic._cli import _cli_migrate as subject

    run = tmp_path / "run"
    tasks = (_task(run, 0), _task(run, 1))
    observed: dict[str, Any] = {}

    class FakeParallel:
        def __init__(self, *, n_jobs: int) -> None:
            observed["n_jobs"] = n_jobs

        def __call__(self, jobs: Any) -> list[Any]:
            values = list(jobs)
            observed["job_count"] = len(values)
            return [function(*args, **kwargs) for function, args, kwargs in values]

    monkeypatch.setattr(joblib, "Parallel", FakeParallel)
    monkeypatch.setattr(
        joblib,
        "delayed",
        lambda function: (
            lambda *args, **kwargs: (function, args, kwargs)
        ),
    )
    monkeypatch.setattr(
        subject,
        "migrate_image_task",
        lambda _output, task, **_: MigrationImageResult(
            index=task.index,
            dataset=task.dataset,
            stem=task.stem,
            work_id=_migration_work_id(task.dataset, task.stem),
            converted=True,
            table_installed=False,
            overlay_rendered=True,
            marker_digest="a" * 64,
            skipped=False,
        ),
    )

    results, failures = _execute_migration_tasks(
        run,
        tasks=tasks,
        metadata_csv=None,
        overlay_alpha=0.3,
        dry_run=False,
        njobs=3,
        commit_guard=None,
    )

    assert observed == {"n_jobs": 3, "job_count": 2}
    assert len(results) == 2
    assert failures == ()

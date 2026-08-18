"""Worker and singleton finalizer for SLURM recompile metadata migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from phenotypic.sdk_ import (
    MetadataMigrationResult,
    atomic_write_json,
    file_fingerprint,
    migrate_metadata_file,
    preflight_metadata_schema,
    progress_dir,
    recompile_dir,
)

from ._cli_recompile_metadata_migration import _metadata_bundle_layout
from ._cli_recompile_metadata_migration_slurm import (
    metadata_migration_finalizer_status_path,
    metadata_migration_task_status_path,
)
from ._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    assert_generation_active,
    generation_publication_guard,
)


@click.command("recompile-metadata-migration-worker")
@click.option(
    "--task-manifest",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--task-index", type=int)
@click.option("--finalize", is_flag=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--slurm-generation", required=True)
@click.option("--attempt-id", required=True)
def main(
    task_manifest: Path,
    task_index: int | None,
    finalize: bool,
    output_dir: Path,
    slurm_generation: str,
    attempt_id: str,
) -> None:
    """Migrate one planned target or validate the completed migration."""
    try:
        _assert_worker_generation(output_dir, slurm_generation, attempt_id)
        if finalize:
            if task_index is not None:
                raise ValueError("--task-index cannot be used with --finalize")
            finalize_metadata_migration(
                task_manifest,
                output_dir=output_dir,
                slurm_generation=slurm_generation,
                attempt_id=attempt_id,
            )
        else:
            if task_index is None:
                raise ValueError("--task-index is required for target workers")
            run_metadata_migration_target(
                task_manifest,
                task_index,
                output_dir=output_dir,
                slurm_generation=slurm_generation,
                attempt_id=attempt_id,
            )
    except SlurmGenerationInactiveError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        assert_generation_active(output_dir, slurm_generation)
        _record_worker_terminal_failure(
            task_manifest,
            task_index=task_index,
            finalize=finalize,
            error=exc,
            manifest_unreadable=not _manifest_is_readable(task_manifest),
            output_dir=output_dir,
            slurm_generation=slurm_generation,
        )
        from ._cli_slurm_lifecycle import deactivate_generation

        deactivate_generation(output_dir, slurm_generation)
        raise click.ClickException(str(exc)) from exc


def run_metadata_migration_target(
    task_manifest: Path,
    task_index: int,
    *,
    output_dir: Path,
    slurm_generation: str,
    attempt_id: str,
) -> None:
    """Migrate one target and atomically record its receipt reference."""
    output_dir = Path(output_dir).resolve()
    _assert_worker_generation(output_dir, slurm_generation, attempt_id)
    manifest = _load_manifest(task_manifest)
    _validate_manifest_identity(
        task_manifest,
        manifest,
        output_dir=output_dir,
        slurm_generation=slurm_generation,
        attempt_id=attempt_id,
    )
    target = _load_target(manifest, task_index)
    status_path = metadata_migration_task_status_path(
        task_manifest, task_index
    )
    try:
        _validate_manifest_target(task_manifest, manifest, target)
        with generation_publication_guard(output_dir, slurm_generation):
            result = migrate_metadata_file(
                Path(str(target["path"])),
                expected_source_fingerprint=str(target["source_fingerprint"]),
            )
        if result.status not in {"applied", "compatible"}:
            details = "; ".join(result.conflicts) or result.status
            raise RuntimeError(
                f"metadata migration {result.status}: {details}"
            )
        with generation_publication_guard(output_dir, slurm_generation):
            atomic_write_json(
                status_path,
                {
                    "status": "completed",
                    "task_index": task_index,
                    "path": str(target["path"]),
                    "source_fingerprint": str(target["source_fingerprint"]),
                    "resulting_fingerprint": result.resulting_fingerprint,
                    "result_status": result.status,
                    "receipt_path": (
                        str(result.receipt_path)
                        if result.receipt_path
                        else None
                    ),
                    "current_fingerprint": file_fingerprint(
                        Path(str(target["path"]))
                    ),
                },
                sort_keys=False,
            )
    except Exception as exc:
        with generation_publication_guard(output_dir, slurm_generation):
            atomic_write_json(
                status_path,
                {
                    "status": "failed",
                    "task_index": task_index,
                    "path": str(target.get("path", "")),
                    "source_fingerprint": str(
                        target.get("source_fingerprint", "")
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=False,
            )
        raise


def finalize_metadata_migration(
    task_manifest: Path,
    *,
    output_dir: Path,
    slurm_generation: str,
    attempt_id: str,
) -> None:
    """Validate all target outcomes and fresh canonical bundle preflight."""
    output_dir = Path(output_dir).resolve()
    _assert_worker_generation(output_dir, slurm_generation, attempt_id)
    manifest = _load_manifest(task_manifest)
    _validate_manifest_identity(
        task_manifest,
        manifest,
        output_dir=output_dir,
        slurm_generation=slurm_generation,
        attempt_id=attempt_id,
    )
    status_path = metadata_migration_finalizer_status_path(task_manifest)
    try:
        targets = manifest["targets"]
        if not isinstance(targets, list):
            raise ValueError("Migration manifest targets must be a list")

        receipts: list[str] = []
        failures: list[str] = []
        for task_index, target in enumerate(targets):
            target_status_path = metadata_migration_task_status_path(
                task_manifest, task_index
            )
            if not target_status_path.is_file():
                failures.append(
                    f"missing target status for index {task_index}: "
                    f"{target.get('path', '')}"
                )
                continue
            try:
                target_status = json.loads(
                    target_status_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError) as exc:
                failures.append(
                    f"unreadable target status {task_index}: {exc}"
                )
                continue
            if (
                target_status.get("status") != "completed"
                or target_status.get("path") != target.get("path")
                or target_status.get("source_fingerprint")
                != target.get("source_fingerprint")
            ):
                failures.append(
                    str(target_status.get("error"))
                    or f"invalid target status for index {task_index}"
                )
                continue
            receipt = target_status.get("receipt_path")
            if receipt:
                receipt_path = Path(str(receipt))
                with generation_publication_guard(
                    output_dir, slurm_generation
                ):
                    validation = migrate_metadata_file(
                        Path(str(target["path"])),
                        expected_source_fingerprint=str(
                            target["source_fingerprint"]
                        ),
                    )
                receipt_error = _validate_receipt_reference(
                    receipt_path, target, validation
                )
                if receipt_error is not None:
                    failures.append(receipt_error)
                    continue
                receipts.append(str(receipt_path))
            else:
                failures.append(
                    f"target lacks migration receipt: {target.get('path', '')}"
                )

        if failures:
            raise RuntimeError("; ".join(failures))

        fresh = preflight_metadata_schema(_metadata_bundle_layout(output_dir))
        if fresh.status != "compatible":
            details = "; ".join(fresh.conflicts) or fresh.status
            raise RuntimeError(
                "Fresh bundle preflight is not canonical after migration: "
                f"{details}"
            )

        with generation_publication_guard(output_dir, slurm_generation):
            from ._cli_completion import (
                refresh_success_markers_after_metadata_migration,
            )

            refresh_success_markers_after_metadata_migration(
                output_dir,
                receipt_paths=tuple(Path(receipt) for receipt in receipts),
            )
            atomic_write_json(
                status_path,
                {
                    "status": "completed",
                    "plan_fingerprint": manifest["plan_fingerprint"],
                    "target_count": len(targets),
                    "receipts": receipts,
                    "fresh_plan_fingerprint": fresh.plan_fingerprint,
                },
                sort_keys=False,
            )
        _finish_migration_generation(
            manifest,
            output_dir=output_dir,
            slurm_generation=slurm_generation,
            succeeded=True,
        )
    except Exception as exc:
        try:
            with generation_publication_guard(output_dir, slurm_generation):
                atomic_write_json(
                    status_path,
                    {
                        "status": "failed",
                        "plan_fingerprint": manifest.get("plan_fingerprint"),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=False,
                )
        finally:
            if not isinstance(exc, SlurmGenerationInactiveError):
                _finish_migration_generation(
                    manifest,
                    output_dir=output_dir,
                    slurm_generation=slurm_generation,
                    succeeded=False,
                )
        raise


def _assert_worker_generation(
    output_dir: Path, slurm_generation: str, attempt_id: str
) -> None:
    """Validate independently supplied migration worker ownership."""
    if not slurm_generation or not attempt_id:
        raise ValueError("SLURM generation and attempt id are required")
    if slurm_generation != attempt_id:
        raise ValueError("SLURM generation and attempt id must match")
    assert_generation_active(Path(output_dir), slurm_generation)


def _validate_manifest_identity(
    task_manifest: Path,
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    slurm_generation: str,
    attempt_id: str,
) -> None:
    """Bind manifest-owned fields to independently supplied script values."""
    if Path(str(manifest.get("output_dir", ""))).resolve() != output_dir:
        raise ValueError(
            "Migration manifest output directory does not match script"
        )
    if manifest.get("slurm_generation") != slurm_generation:
        raise ValueError("Migration manifest generation does not match script")
    if manifest.get("attempt_id") != attempt_id:
        raise ValueError("Migration manifest attempt id does not match script")
    try:
        task_manifest.resolve().relative_to(
            recompile_dir(progress_dir(output_dir)) / "attempts" / attempt_id
        )
    except ValueError as exc:
        raise ValueError(
            "Migration manifest is outside its attempt namespace"
        ) from exc


def _load_manifest(task_manifest: Path) -> dict[str, Any]:
    """Load and minimally validate one migration plan manifest."""
    manifest = json.loads(Path(task_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("targets"), list
    ):
        raise ValueError("Migration manifest does not contain a targets list")
    return manifest


def _manifest_is_readable(task_manifest: Path) -> bool:
    """Return whether a worker manifest can supply its task list."""
    try:
        _load_manifest(task_manifest)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _record_worker_terminal_failure(
    task_manifest: Path,
    *,
    task_index: int | None,
    finalize: bool,
    error: Exception,
    manifest_unreadable: bool,
    output_dir: Path,
    slurm_generation: str,
) -> None:
    """Write waiter-visible failure without trusting manifest semantics."""
    status_path = (
        metadata_migration_finalizer_status_path(task_manifest)
        if finalize
        else metadata_migration_task_status_path(
            task_manifest, 0 if task_index is None else task_index
        )
    )
    payload = {
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "manifest_unreadable": manifest_unreadable,
        "worker_terminal_failure": True,
    }
    if not status_path.is_file():
        with generation_publication_guard(output_dir, slurm_generation):
            atomic_write_json(status_path, payload, sort_keys=False)
    # Target workers can fail before the singleton finalizer is released.
    # Publish its terminal status as well so the submit-side waiter observes
    # the current attempt failure instead of polling forever.
    finalizer_status = metadata_migration_finalizer_status_path(task_manifest)
    if status_path != finalizer_status and not finalizer_status.is_file():
        with generation_publication_guard(output_dir, slurm_generation):
            atomic_write_json(finalizer_status, payload, sort_keys=False)


def _validate_receipt_reference(
    receipt_path: Path,
    target: dict[str, Any],
    validation: MetadataMigrationResult,
) -> str | None:
    """Return an error when a worker receipt does not prove its target."""
    if validation.status != "applied":
        details = "; ".join(validation.conflicts) or validation.status
        return f"durable migration receipt validation failed: {details}"
    if validation.receipt_path != receipt_path:
        return "durable migration receipt identity changed"
    current_fingerprint = file_fingerprint(Path(str(target["path"])))
    if not receipt_path.is_file():
        return f"missing migration receipt: {receipt_path}"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        records = receipt.get("targets")
        if receipt.get("state") != "applied" or not isinstance(records, list):
            raise ValueError("receipt is not applied")
        matching = [
            record
            for record in records
            if isinstance(record, dict)
            and Path(str(record.get("path", ""))).resolve()
            == Path(str(target["path"])).resolve()
            and record.get("source_fingerprint")
            == target.get("source_fingerprint")
            and record.get("state") in {"applied", "skipped"}
        ]
        if len(matching) != 1:
            raise ValueError("receipt does not contain the planned target")
        if matching[0].get("post_fingerprint") != current_fingerprint:
            raise ValueError("current target fingerprint differs from receipt")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f"invalid migration receipt {receipt_path}: {exc}"
    return None


def _finish_migration_generation(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    slurm_generation: str,
    succeeded: bool,
) -> None:
    """Fence terminal migration attempts and failed prerequisite chains."""
    should_finish = not bool(manifest.get("has_recompile_downstream"))
    if not succeeded:
        should_finish = True
    if should_finish:
        from ._cli_slurm_lifecycle import deactivate_generation

        deactivate_generation(output_dir, slurm_generation)


def _load_target(manifest: dict[str, Any], task_index: int) -> dict[str, Any]:
    """Load one target dictionary by zero-based index."""
    try:
        target = manifest["targets"][task_index]
    except IndexError as exc:
        raise ValueError(
            f"Migration target index out of range: {task_index}"
        ) from exc
    if not isinstance(target, dict):
        raise ValueError(f"Migration target {task_index} is not a dictionary")
    return target


def _validate_manifest_target(
    task_manifest: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
) -> None:
    """Reject a tampered plan target outside the authoritative bundle.

    Validation is repeated for every worker because the manifest is durable
    scheduler input. Earlier targets may already be canonical, but the bundle
    target path and kind set remains stable throughout the plan.
    """
    output_dir = Path(str(manifest["output_dir"])).resolve()
    plan_fingerprint = str(manifest["plan_fingerprint"])
    digest = plan_fingerprint.removeprefix("sha256:")[:16]
    plan_base = recompile_dir(progress_dir(output_dir))
    attempt_id = manifest.get("attempt_id")
    if attempt_id is not None:
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("Migration manifest attempt id is invalid")
        plan_base = plan_base / "attempts" / attempt_id
    expected_manifest = (
        plan_base / "metadata_migration" / digest / "migration_plan.json"
    )
    if Path(task_manifest).resolve() != expected_manifest.resolve():
        raise ValueError(
            "Migration manifest is outside its authoritative plan"
        )

    fresh = preflight_metadata_schema(_metadata_bundle_layout(output_dir))
    requested_path = str(Path(str(target["path"])).resolve())
    matches = [
        candidate
        for candidate in fresh.targets
        if str(Path(candidate.path).resolve()) == requested_path
        and candidate.kind == str(target["kind"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "Migration manifest target is outside the authoritative bundle: "
            f"{target.get('path', '')}"
        )
    candidate = matches[0]
    if candidate.status == "blocked":
        details = "; ".join(candidate.conflicts) or candidate.status
        raise RuntimeError(f"metadata migration blocked: {details}")
    if candidate.status == "migratable":
        planned_fields = {
            "status": candidate.status,
            "source_fingerprint": candidate.source_fingerprint,
            "proposed_header_map": [
                list(pair) for pair in candidate.proposed_header_map
            ],
            "needs_metadata_marker": candidate.needs_metadata_marker,
            "hdf_snapshot_fingerprint": candidate.hdf_snapshot_fingerprint,
            "conflicts": list(candidate.conflicts),
            "mixed_table": candidate.mixed_table,
        }
        manifest_fields = {key: target.get(key) for key in planned_fields}
        if manifest_fields != planned_fields:
            raise ValueError(
                "Migration manifest target planning fields no longer match "
                f"fresh preflight: {target.get('path', '')}"
            )


if __name__ == "__main__":
    main()

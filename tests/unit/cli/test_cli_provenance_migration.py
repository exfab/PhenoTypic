"""Explicit schema-v1 to schema-v2 store provenance migration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _operation(version: str) -> dict[str, Any]:
    return {
        "sequence": 1,
        "operation_name": "BlurGauss",
        "operation_class": "phenotypic.enhance._blur_gauss.BlurGauss",
        "phenotypic_version": version,
        "parameters": {"sigma": 1.5},
        "applied_at_utc": "2026-08-31T12:00:00.000Z",
        "duration_seconds": 0.25,
        "pipeline_step_path": ["blur"],
    }


def _write_store_root(
    store: Path,
    journal: dict[str, Any],
    *,
    root_version: str = "0.17.0",
) -> Path:
    store.mkdir(parents=True)
    root = store / "zarr.json"
    root.write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "phenotypic": {
                        "store_schema_version": 3,
                        "phenotypic_version": root_version,
                        "provenance": journal,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_upgrade_v1_store_root_preserves_history_and_recovers_version(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_migrate_provenance import upgrade_store_provenance

    operation = _operation("0.18.7")
    store = tmp_path / "plate.ome.zarr"
    _write_store_root(
        store,
        {
            "schema_version": 1,
            "status": "complete",
            "pipeline": {
                "source_path": r"C:\archive\pipelines\crop.json",
                "sha256": "a" * 64,
            },
            "retry_base_length": 0,
            "operations": [operation],
        },
    )

    result = upgrade_store_provenance(store)

    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    journal = payload["attributes"]["phenotypic"]["provenance"]
    assert result.upgraded is True
    assert result.schema_before == 1
    assert journal == {
        "schema_version": 2,
        "status": "complete",
        "original_filename": None,
        "applications": [
            {
                "sequence": 1,
                "kind": "legacy",
                "phenotypic_version": "0.18.7",
                "input_filename": None,
                "status": "complete",
                "pipeline": {
                    "source_path": "crop.json",
                    "sha256": "a" * 64,
                },
                "retry_base_length": 0,
                "operations": [operation],
            }
        ],
    }


def test_classifies_direct_full_and_process_targets_without_descending_stores(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from phenotypic._cli._cli_migrate_provenance import (
        classify_provenance_migration_target,
    )

    direct = tmp_path / "direct.ome.zarr"
    _write_store_root(direct, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    })
    (direct / "rgb" / "0" / "c").mkdir(parents=True)

    process_root = tmp_path / "process"
    process_store = process_root / "day1" / "plate.ome.zarr"
    _write_store_root(process_store, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    })
    (process_store / "rgb" / "0" / "c").mkdir(parents=True)

    full_root = tmp_path / "full"
    full_store = full_root / "results" / "day1" / "zarr" / "plate.ome.zarr"
    _write_store_root(full_store, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    })
    (full_store / "rgb" / "0" / "c").mkdir(parents=True)

    real_iterdir = Path.iterdir
    stores = {direct, process_store, full_store}

    def _guarded_iterdir(path: Path):
        if path in stores:
            raise AssertionError(f"inventory descended into store {path}")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _guarded_iterdir)

    direct_target = classify_provenance_migration_target(direct)
    process_target = classify_provenance_migration_target(process_root)
    full_target = classify_provenance_migration_target(full_root)

    assert direct_target.kind == "direct_store"
    assert direct_target.stores == (direct,)
    assert process_target.kind == "process_tree"
    assert process_target.stores == (process_store,)
    assert full_target.kind == "full_run"
    assert full_target.stores == (full_store,)


def test_run_migrate_direct_store_is_dry_run_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_migrate import run_migrate

    store = tmp_path / "direct.ome.zarr"
    root = _write_store_root(store, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    }, root_version="")
    before = root.read_bytes()

    dry_report = run_migrate(store, njobs=4, dry_run=True)

    assert dry_report.provenance_upgraded == 1
    assert root.read_bytes() == before

    report = run_migrate(store, njobs=4)
    after = root.read_bytes()
    second = run_migrate(store, njobs=4)

    assert report.provenance_upgraded == 1
    assert json.loads(after)["attributes"]["phenotypic"]["provenance"][
        "applications"
    ][0]["phenotypic_version"] is None
    assert second.provenance_upgraded == 0
    assert root.read_bytes() == after
    assert not (store / ".phenotypic").exists()


def test_full_run_upgrades_provenance_before_image_recertification(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from phenotypic._cli import _cli_migrate as subject
    from phenotypic._cli._cli_migrate import MetadataPassResult

    run = tmp_path / "run"
    store = run / "results" / "ds" / "zarr" / "plate.ome.zarr"
    _write_store_root(store, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    })
    authority = SimpleNamespace(
        terminal_receipt_digest="sha256:" + "a" * 64
    )
    monkeypatch.setattr(subject, "new_slurm_generation", lambda: "generation-1")
    monkeypatch.setattr(
        subject,
        "run_metadata_pass",
        lambda *args, **kwargs: MetadataPassResult(0, (), authority),
    )
    monkeypatch.setattr(
        subject, "_ensure_migration_processing_state", lambda *args, **kwargs: None
    )

    def _image_stage(*args: Any, **kwargs: Any):
        payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
        assert payload["attributes"]["phenotypic"]["provenance"][
            "schema_version"
        ] == 2
        return (), ()

    monkeypatch.setattr(subject, "_execute_migration_tasks", _image_stage)
    monkeypatch.setattr(
        subject,
        "seal_migration_image_stage",
        lambda *args, **kwargs: SimpleNamespace(
            clean=True, failures=(), generation="generation-1"
        ),
    )
    monkeypatch.setattr(
        subject,
        "finalize_migration_attempt",
        lambda *args, **kwargs: kwargs["report"],
    )

    report = subject.run_migrate(run)

    assert report.provenance_upgraded == 1
    assert report.provenance_failures == ()


def test_direct_store_refuses_active_external_migration_before_rewrite(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_migrate import MigrateModeError, run_migrate
    from phenotypic._cli._cli_migrate_provenance import (
        classify_provenance_migration_target,
        provenance_migration_lifecycle_root,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import phenotypic_cache_dir

    store = tmp_path / "direct.ome.zarr"
    root = _write_store_root(store, {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    })
    before = root.read_bytes()
    target = classify_provenance_migration_target(store)
    lifecycle_root = provenance_migration_lifecycle_root(target)
    initialize_slurm_lifecycle(
        lifecycle_root,
        generation="already-active",
        mode="migrate",
        owner_kind="local",
        control_root=phenotypic_cache_dir(lifecycle_root),
    )

    with pytest.raises(MigrateModeError, match="active"):
        run_migrate(store)

    assert root.read_bytes() == before
    assert not lifecycle_root.is_relative_to(store)
    assert ".phenotypic" in lifecycle_root.parts


@pytest.mark.parametrize(
    "journal",
    [
        {
            "schema_version": 3,
            "status": "complete",
            "original_filename": None,
            "applications": [],
        },
        {
            "schema_version": 1,
            "status": "complete",
            "pipeline": None,
            "retry_base_length": 0,
            "operations": [],
            "unexpected": True,
        },
    ],
)
def test_malformed_and_future_journals_are_reported_without_rewrite(
    tmp_path: Path, journal: dict[str, Any]
) -> None:
    from phenotypic._cli._cli_migrate import run_migrate

    store = tmp_path / "bad.ome.zarr"
    root = _write_store_root(store, journal)
    before = root.read_bytes()

    report = run_migrate(store)

    assert report.ok is False
    assert len(report.provenance_failures) == 1
    assert root.read_bytes() == before


def test_provenance_only_rejects_delete_sources_and_ambiguous_layout(
    tmp_path: Path,
) -> None:
    from phenotypic._cli._cli_migrate import MigrateModeError, run_migrate
    from phenotypic._cli._cli_migrate_provenance import (
        classify_provenance_migration_target,
    )

    process_root = tmp_path / "mixed"
    process_store = process_root / "process.ome.zarr"
    full_store = (
        process_root
        / "results"
        / "ds"
        / "zarr"
        / "full.ome.zarr"
    )
    legacy = {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    }
    _write_store_root(process_store, legacy)
    _write_store_root(full_store, legacy)

    with pytest.raises(ValueError, match="ambiguous"):
        classify_provenance_migration_target(process_root)
    with pytest.raises(MigrateModeError, match="delete-sources"):
        run_migrate(process_store, delete_sources=True)


def test_one_root_read_per_store_and_parallel_process_tree_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import joblib

    from phenotypic._cli._cli_migrate_provenance import (
        classify_provenance_migration_target,
        execute_provenance_migration,
    )

    process_root = tmp_path / "process"
    legacy = {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    }
    roots = []
    for name in ("a", "b"):
        roots.append(
            _write_store_root(
                process_root / f"{name}.ome.zarr", legacy
            )
        )
    reads = {root: 0 for root in roots}
    real_read_text = Path.read_text

    def _counted_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path in reads:
            reads[path] += 1
        return real_read_text(path, *args, **kwargs)

    observed: dict[str, int] = {}

    class FakeParallel:
        def __init__(self, *, n_jobs: int) -> None:
            observed["n_jobs"] = n_jobs

        def __call__(self, jobs: Any) -> list[Any]:
            values = list(jobs)
            observed["job_count"] = len(values)
            return [function(*args, **kwargs) for function, args, kwargs in values]

    monkeypatch.setattr(Path, "read_text", _counted_read)
    monkeypatch.setattr(joblib, "Parallel", FakeParallel)
    monkeypatch.setattr(
        joblib,
        "delayed",
        lambda function: (
            lambda *args, **kwargs: (function, args, kwargs)
        ),
    )

    target = classify_provenance_migration_target(process_root)
    results, failures = execute_provenance_migration(
        target, n_jobs=32, dry_run=True
    )

    assert observed == {"n_jobs": 32, "job_count": 2}
    assert len(results) == 2
    assert failures == ()
    assert reads == {root: 1 for root in roots}

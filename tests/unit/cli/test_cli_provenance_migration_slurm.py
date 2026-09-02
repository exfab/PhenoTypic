"""Behavior tests for dispatcher-fed provenance-only migration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import pytest


def _store(path: Path, *, schema_version: int = 1) -> Path:
    """Create one minimal PhenoTypic store with durable provenance."""
    path.mkdir(parents=True)
    provenance: dict[str, object]
    if schema_version == 1:
        provenance = {
            "schema_version": 1,
            "status": "complete",
            "pipeline": None,
            "retry_base_length": 0,
            "operations": [],
        }
    else:
        provenance = {
            "schema_version": schema_version,
            "status": "complete",
            "original_filename": None,
            "applications": [
                {
                    "sequence": 1,
                    "kind": "legacy",
                    "phenotypic_version": None,
                    "input_filename": None,
                    "status": "complete",
                    "pipeline": None,
                    "retry_base_length": 0,
                    "operations": [],
                }
            ],
        }
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5"},
                    "phenotypic": {
                        "store_schema_version": 3,
                        "provenance": provenance,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _target(root: Path, count: int):
    """Return one explicit process-tree target without scanning chunks."""
    from phenotypic._cli._cli_migrate_provenance import (
        ProvenanceMigrationTarget,
    )

    stores = tuple(
        _store(root / "dataset" / f"image_{index}.ome.zarr")
        for index in range(count)
    )
    return ProvenanceMigrationTarget("process_tree", root.resolve(), stores)


def _script_entries(path: Path) -> list[int]:
    """Read absolute manifest indices from one generated array script."""
    text = path.read_text(encoding="utf-8")
    block = text.split("TASK_INDICES=(\n", 1)[1].split("\n)", 1)[0]
    return [int(line.strip()) for line in block.splitlines()]


def test_provenance_plan_is_store_array_seal_finalizer_with_typed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance work is indexed and never grows metadata/image sidecars."""
    from phenotypic._cli import _cli_migrate_slurm as subject
    from phenotypic._cli._cli_migrate_provenance_manifest import (
        read_provenance_migration_task,
    )

    target = _target(tmp_path / "process", 5)
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 2)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 20)

    plan = subject.generate_provenance_migration_slurm_plan(
        target,
        slurm_args={"slurm_partition": "short"},
        dry_run=False,
        generation="generation-1",
    )

    assert plan.topology == "provenance_only"
    assert [path.stem for path in plan.flat_scripts] == [
        "provenance_chunk0",
        "provenance_chunk1",
        "provenance_chunk2",
        "provenance_seal",
    ]
    assert [_script_entries(path) for path in plan.flat_scripts] == [
        [0, 1],
        [2, 3],
        [4],
        [0],
    ]
    assert plan.finalizer_script.stem == "finalize"
    assert all("metadata" not in path.stem for path in plan.flat_scripts)
    assert all("image" not in path.stem for path in plan.flat_scripts)
    assert all("reclaim" not in path.stem for path in plan.flat_scripts)
    task = read_provenance_migration_task(
        plan.manifest_path,
        3,
        expected_target_root=target.root,
        expected_control_root=plan.control_root,
    )
    assert task.task_type == "provenance_store"
    assert task.index == 3
    assert task.store_path == target.stores[3].resolve()
    config = json.loads(
        (plan.control_root / "migration_config.json").read_text()
    )
    assert config["schema_version"] == 1
    assert config["topology"] == "provenance_only"
    assert config["target_kind"] == "process_tree"


def test_direct_store_plan_externalizes_lifecycle_and_control_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No scheduler or lifecycle artifact is ever written inside a store."""
    from phenotypic._cli import _cli_migrate_slurm as subject
    from phenotypic._cli._cli_migrate_provenance import (
        ProvenanceMigrationTarget,
        provenance_migration_lifecycle_root,
    )

    store = _store(tmp_path / "plate.ome.zarr")
    target = ProvenanceMigrationTarget("direct_store", store.resolve(), (store,))
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)

    plan = subject.generate_provenance_migration_slurm_plan(
        target, slurm_args={}, dry_run=False, generation="generation-1"
    )

    expected_lifecycle = provenance_migration_lifecycle_root(target).resolve()
    assert plan.lifecycle_root == expected_lifecycle
    assert plan.control_root.is_relative_to(expected_lifecycle)
    assert not plan.control_root.is_relative_to(store)
    assert not (store / ".phenotypic").exists()


def test_provenance_worker_chain_publishes_versioned_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store workers, seal, and finalizer preserve typed generation authority."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli import _cli_migrate_provenance_worker as worker
    from phenotypic._cli import _cli_migrate_slurm as slurm
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    target = _target(tmp_path / "process", 2)
    monkeypatch.setattr(slurm, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(slurm, "get_slurm_max_submit_jobs", lambda: 100)
    plan = slurm.generate_provenance_migration_slurm_plan(
        target, slurm_args={}, dry_run=False, generation="generation-1"
    )
    initialize_slurm_lifecycle(
        plan.lifecycle_root,
        generation=plan.generation,
        mode="migrate",
        owner_kind="slurm",
        control_root=plan.control_root,
    )
    runner = CliRunner()
    config = plan.control_root / "migration_config.json"
    for index in range(plan.task_count):
        result = runner.invoke(
            worker.provenance_migration_worker_cli,
            ["--config", str(config), "store", "--index", str(index)],
        )
        assert result.exit_code == 0, result.output
    seal = runner.invoke(
        worker.provenance_migration_worker_cli,
        ["--config", str(config), "seal"],
    )
    assert seal.exit_code == 0, seal.output
    finalizer = runner.invoke(
        worker.provenance_migration_worker_cli,
        ["--config", str(config), "finalize"],
    )
    assert finalizer.exit_code == 0, finalizer.output

    terminal = migrate._read_migration_terminal_status(
        migrate.migration_terminal_status_path(
            plan.control_root, plan.generation
        ),
        generation=plan.generation,
    )
    assert terminal is not None
    assert terminal["status"] == "succeeded"
    assert terminal["report"]["provenance_upgraded"] == 2
    lifecycle = load_slurm_lifecycle(plan.lifecycle_root)
    assert lifecycle is not None and lifecycle["active"] is False
    for index, store in enumerate(target.stores):
        status = json.loads(
            worker.provenance_worker_status_path(
                plan.control_root, plan.generation, index
            ).read_text()
        )
        assert status["schema_version"] == 1
        assert status["task_type"] == "provenance_store"
        assert status["state"] == "complete"
        assert status["store_path"] == str(store.resolve())
        root = json.loads((store / "zarr.json").read_text())
        assert root["attributes"]["phenotypic"]["provenance"]["schema_version"] == 2


def test_provenance_submitter_uses_one_shared_dispatch_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No provenance scheduler sidecar is submitted beside the store array."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    target = _target(tmp_path / "process", 2)
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)
    plan = subject.generate_provenance_migration_slurm_plan(
        target, slurm_args={}, dry_run=False, generation="generation-1"
    )
    calls: list[dict[str, object]] = []

    def _submit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(job_ids=["11", "12"], flat_scripts=[])

    monkeypatch.setattr(subject, "submit_slurm_script_chain", _submit)
    subject.submit_migration_slurm_plan(
        plan, slurm_args={}, console=SimpleNamespace(print=lambda *_: None)
    )

    assert len(calls) == 1
    assert calls[0]["flat_chunk_scripts"] == plan.flat_scripts
    assert calls[0]["finalizer_script"] == plan.finalizer_script
    assert calls[0]["output_dir"] == plan.lifecycle_root
    assert calls[0]["continuation_dependency_kinds"] == (
        "afterany",
        "afterany",
    )


def test_public_slurm_dispatch_routes_direct_store_to_provenance_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public migrate mode selects external fenced store-array work."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli import _cli_migrate_slurm as slurm
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    store = _store(tmp_path / "plate.ome.zarr")
    monkeypatch.setattr(migrate, "new_slurm_generation", lambda: "generation-1")
    monkeypatch.setattr(slurm, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(slurm, "get_slurm_max_submit_jobs", lambda: 100)
    monkeypatch.setattr(migrate, "get_slurm_array_limit", lambda: 100, raising=False)
    monkeypatch.setattr(
        migrate, "get_slurm_max_submit_jobs", lambda: 100, raising=False
    )
    submitted = []

    def _submit(plan, **_kwargs):
        submitted.append(plan)
        return SimpleNamespace(job_ids=["101"])

    monkeypatch.setattr(migrate, "submit_migration_slurm_plan", _submit)

    result = migrate.handle_migrate_mode(
        store, slurm_args={"slurm_partition": "short"}
    )

    assert result == 0
    assert len(submitted) == 1
    plan = submitted[0]
    assert plan.topology == "provenance_only"
    assert not plan.control_root.is_relative_to(store)
    lifecycle = load_slurm_lifecycle(plan.lifecycle_root)
    assert lifecycle is not None and lifecycle["active"] is True


def test_public_slurm_provenance_refuses_delete_sources_before_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected provenance-only option cannot initialize scheduler state."""
    import click

    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli._cli_migrate_provenance import (
        classify_provenance_migration_target,
        provenance_migration_lifecycle_root,
    )
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    store = _store(tmp_path / "plate.ome.zarr")
    lifecycle_root = provenance_migration_lifecycle_root(
        classify_provenance_migration_target(store)
    )
    monkeypatch.setattr(
        migrate,
        "generate_provenance_migration_slurm_plan",
        lambda *_a, **_k: pytest.fail("invalid option reached planning"),
        raising=False,
    )

    with pytest.raises(click.ClickException, match="delete-sources"):
        migrate.handle_migrate_mode(
            store, slurm_args={"slurm_partition": "short"}, delete_sources=True
        )

    assert load_slurm_lifecycle(lifecycle_root) is None


def test_full_run_image_worker_upgrades_provenance_before_image_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical full-run chain performs provenance before recertification."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate_image import MigrationImageResult
    from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
    from phenotypic._cli._cli_migrate_provenance import ProvenanceUpgradeResult

    output = tmp_path / "run"
    store = _store(output / "results" / "ds" / "zarr" / "img.ome.zarr")
    task = MigrationImageTask(
        index=0,
        dataset="ds",
        stem="img",
        hdf_path=None,
        store_path=store,
        measurement_path=None,
        overlay_path=output / "overlay.png",
        marker_path=output / "marker.json",
    )
    config = SimpleNamespace(
        lifecycle_root=output,
        generation="generation-1",
        manifest_path=tmp_path / "manifest.json",
        scientific_output=output / "deliverables",
        control_root=tmp_path / "control",
        output_dir=output,
        overlay_alpha=0.3,
        dry_run=False,
    )
    authority = SimpleNamespace(terminal_receipt_digest="sha256:" + "a" * 64)
    events: list[str] = []
    published: dict[str, object] = {}
    monkeypatch.setattr(worker, "assert_generation_active", lambda *_a: None)
    monkeypatch.setattr(worker, "_read_worker_status", lambda *_a: {})
    monkeypatch.setattr(
        worker, "_metadata_prerequisite", lambda *_a: (authority, None)
    )
    monkeypatch.setattr(worker, "read_migration_task", lambda *_a, **_k: task)

    def _upgrade(*_a, **_k):
        events.append("provenance")
        return ProvenanceUpgradeResult(store, 1, True)

    monkeypatch.setattr(worker, "upgrade_store_provenance", _upgrade)

    def _migrate(*_a, **_k):
        assert events == ["provenance"]
        events.append("image")
        return MigrationImageResult(
            index=0,
            dataset="ds",
            stem="img",
            work_id="work",
            converted=False,
            table_installed=False,
            overlay_rendered=False,
            marker_digest="b" * 64,
            skipped=True,
        )

    monkeypatch.setattr(worker, "migrate_image_task", _migrate)
    monkeypatch.setattr(
        worker, "publish_migration_task_status", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        worker,
        "_publish_worker_status",
        lambda *_a, **kwargs: published.update(kwargs) or tmp_path / "status.json",
    )

    assert worker._run_image_worker(config, 0) == 0
    assert events == ["provenance", "image"]
    extra = published["extra"]
    assert isinstance(extra, dict)
    assert extra["provenance"] == {
        "store_path": str(store),
        "schema_before": 1,
        "upgraded": True,
    }


def test_full_run_image_report_preserves_provenance_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distributed full-run provenance work reaches terminal report counters."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    store = tmp_path / "img.ome.zarr"
    task = MigrationImageTask(
        index=0,
        dataset="ds",
        stem="img",
        hdf_path=None,
        store_path=store,
        measurement_path=None,
        overlay_path=tmp_path / "overlay.png",
        marker_path=tmp_path / "marker.json",
    )
    config = SimpleNamespace(
        task_count=1,
        manifest_path=tmp_path / "manifest.json",
        scientific_output=tmp_path / "deliverables",
        control_root=tmp_path / "control",
    )
    status = {
        "status": "complete",
        "result": {
            "index": 0,
            "dataset": "ds",
            "stem": "img",
            "work_id": "work",
            "converted": False,
            "table_installed": False,
            "overlay_rendered": False,
            "marker_digest": "b" * 64,
            "skipped": True,
        },
        "provenance": {
            "store_path": str(store),
            "schema_before": 1,
            "upgraded": True,
        },
    }
    monkeypatch.setattr(worker, "read_migration_task", lambda *_a, **_k: task)
    monkeypatch.setattr(worker, "_read_worker_status", lambda *_a: status)
    monkeypatch.setattr(
        worker,
        "_report_from_image_results",
        lambda *_a, **_k: MigrationReport(skipped=1),
    )

    report, failures = worker._image_report(config)

    assert failures == ()
    assert report.provenance_upgraded == 1
    assert report.provenance_failures == ()


def test_provenance_worker_failure_seals_and_terminalizes_as_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed store closes afterany with exact failed-store authority."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli import _cli_migrate_provenance_worker as worker
    from phenotypic._cli import _cli_migrate_slurm as slurm
    from phenotypic._cli._cli_migrate_provenance import (
        ProvenanceMigrationTarget,
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    store = _store(tmp_path / "bad.ome.zarr", schema_version=3)
    target = ProvenanceMigrationTarget(
        "direct_store", store.resolve(), (store.resolve(),)
    )
    monkeypatch.setattr(slurm, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(slurm, "get_slurm_max_submit_jobs", lambda: 100)
    plan = slurm.generate_provenance_migration_slurm_plan(
        target, slurm_args={}, dry_run=False, generation="generation-1"
    )
    initialize_slurm_lifecycle(
        plan.lifecycle_root,
        generation=plan.generation,
        mode="migrate",
        owner_kind="slurm",
        control_root=plan.control_root,
    )
    runner = CliRunner()
    config = plan.control_root / "migration_config.json"

    store_result = runner.invoke(
        worker.provenance_migration_worker_cli,
        ["--config", str(config), "store", "--index", "0"],
    )
    seal_result = runner.invoke(
        worker.provenance_migration_worker_cli,
        ["--config", str(config), "seal"],
    )
    final_result = runner.invoke(
        worker.provenance_migration_worker_cli,
        ["--config", str(config), "finalize"],
    )

    assert store_result.exit_code == 1
    assert seal_result.exit_code == 1
    assert final_result.exit_code == 1
    terminal = migrate._read_migration_terminal_status(
        migrate.migration_terminal_status_path(
            plan.control_root, plan.generation
        ),
        generation=plan.generation,
    )
    assert terminal is not None
    assert terminal["status"] == "failed"
    assert terminal["failure_category"] == "provenance"
    failures = terminal["report"]["provenance_failures"]
    assert failures[0]["path"] == str(store.resolve())
    assert "unsupported provenance schema version" in failures[0]["reason"]
    lifecycle = load_slurm_lifecycle(plan.lifecycle_root)
    assert lifecycle is not None
    assert lifecycle["active"] is False
    assert lifecycle["terminal_status"] == "failed"

"""Pure preflight and explicit Results recipe migration tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pandas as pd
import pytest

from phenotypic._cli import _cli_output_manager
from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.analysis.qc import ExpectedVsDetectedCount
from phenotypic.gui.results_viewer import _compatibility
from phenotypic.gui.results_viewer._compatibility import (
    CompatibilityMigrationError,
    migrate_output_recipe,
    preflight_output_compatibility,
)
from phenotypic.sdk_ import (
    file_fingerprint,
    pipeline_json_path,
)
from phenotypic.sdk_._qc_recipe import QcRecipe
from phenotypic.schema import METADATA

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "output_compatibility"


def _historical_pipeline(tmp_path: Path) -> Path:
    """Write the exact historical GridOccupancy shape and current metadata."""
    metadata = tmp_path / "layout.csv"
    pd.DataFrame({
        str(METADATA.IMAGE_NAME): ["plate-a.png", "plate-a.png"],
        "Object_Label": [1, 2],
    }).to_csv(metadata, index=False)
    raw = (
        _FIXTURES / "grid_occupancy_metadata_source_v1.json"
    ).read_text(encoding="utf-8")
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    pipeline.write_text(
        raw.replace("__METADATA_PATH__", str(metadata)),
        encoding="utf-8",
    )
    return pipeline


def _historical_output_pipeline(tmp_path: Path) -> Path:
    """Write the historical fixture at the canonical CLI output location."""
    source = _historical_pipeline(tmp_path)
    target = pipeline_json_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return target


def _historical_legacy_output_pipeline(tmp_path: Path) -> Path:
    """Write the historical fixture at the V1 output pipeline location."""
    source = _historical_pipeline(tmp_path)
    target = tmp_path / "deliverables" / "pipeline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "master_measurements.parquet").write_bytes(b"fixture")
    source.replace(target)
    return target


def test_preflight_maps_exact_historical_grid_shape_without_writes(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    before = pipeline.read_bytes()

    report = preflight_output_compatibility(pipeline)

    assert report.status == "migratable"
    assert pipeline.read_bytes() == before
    assert report.migrated_pipeline_payload is not None
    entry = report.migrated_pipeline_payload["qc"][0]  # type: ignore[index]
    assert entry["historical_note"] == {"preserve": "verbatim"}  # type: ignore[index]
    params = entry["params"]  # type: ignore[index]
    assert params["metadata"].endswith("layout.csv")  # type: ignore[index]
    assert "metadata_source" not in params
    assert "cell_label" not in params
    assert params["groupby"] == [str(METADATA.IMAGE_NAME)]  # type: ignore[index]
    assert [issue.code for issue in report.issues] == [
        "qc.grid.metadata_source",
        "qc.grid.cell_label_null",
        "qc.grid.groupby_alias",
    ]


def test_preflight_blocks_ambiguous_metadata_without_guessing(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({str(METADATA.IMAGE_NAME): ["a"]}).to_csv(first, index=False)
    pd.DataFrame({str(METADATA.IMAGE_NAME): ["b"]}).to_csv(second, index=False)
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    payload = json.loads(_historical_pipeline(tmp_path).read_text(encoding="utf-8"))
    params = payload["qc"][0]["params"]
    params["metadata"] = str(first)
    params["metadata_source"] = str(second)
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert report.migrated_pipeline_payload is None
    assert [issue.code for issue in report.issues] == [
        "qc.grid.metadata.ambiguous"
    ]


def test_preflight_reports_equal_retired_metadata_source_once(
    tmp_path: Path,
) -> None:
    """A redundant retired field is removed with one deterministic issue."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    params = payload["qc"][0]["params"]
    params["metadata"] = params["metadata_source"]
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "migratable"
    assert sum(
        issue.code == "qc.grid.metadata_source" for issue in report.issues
    ) == 1


def test_preflight_blocks_missing_historical_metadata(
    tmp_path: Path,
) -> None:
    """A retired metadata path is never accepted without reading its target."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    payload["qc"][0]["params"]["metadata_source"] = str(
        tmp_path / "missing.csv"
    )
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert [issue.code for issue in report.issues][-1] == (
        "qc.grid.metadata.unavailable"
    )


def test_preflight_blocks_unmapped_historical_groupby(
    tmp_path: Path,
) -> None:
    """Only the observed and tested metadata column alias is migrated."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    payload["qc"][0]["params"]["groupby"] = ["Metadata_ImageFile"]
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert [issue.code for issue in report.issues][-1] == (
        "qc.grid_migration.invalid"
    )


def test_unknown_entry_is_blocked_and_raw_file_is_preserved(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    fixture = _FIXTURES / "unknown_qc_entry.json"
    pipeline.write_bytes(fixture.read_bytes())
    before = pipeline.read_bytes()
    recipe = QcRecipe._load_from_paths(pipeline, pipeline)
    warnings_before = list(recipe.load_warnings)

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert report.migrated_pipeline_payload is None
    assert pipeline.read_bytes() == before
    assert recipe.load_warnings == warnings_before
    assert len(report.issues) == 1


def test_unreadable_recipe_mutation_never_writes_minimal_pipeline(
    tmp_path: Path,
) -> None:
    """A corrupt existing pipeline is preserved byte-for-byte on mutation."""
    pipeline = pipeline_json_path(tmp_path)
    pipeline.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"qc": [broken'
    pipeline.write_bytes(original)
    recipe = QcRecipe.load(tmp_path)

    added = recipe.add(
        ExpectedVsDetectedCount,
        {
            "metadata": str(tmp_path / "missing.csv"),
            "groupby": [str(METADATA.IMAGE_NAME)],
        },
    )

    assert added is None
    assert pipeline.read_bytes() == original


def test_explicit_migration_is_backed_up_atomic_receipted_and_idempotent(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    original = pipeline.read_bytes()
    report = preflight_output_compatibility(pipeline)

    result = migrate_output_recipe(
        pipeline,
        expected_source_fingerprint=report.source_fingerprint,
    )

    assert result.applied is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "applied"
    assert receipt["old_fingerprint"] == report.source_fingerprint
    assert receipt["new_fingerprint"] == file_fingerprint(pipeline)
    assert receipt["backup_path"] == str(result.backup_path.resolve())
    assert preflight_output_compatibility(pipeline).status == "compatible"

    artifacts_before = sorted(result.backup_path.parent.iterdir())
    current = preflight_output_compatibility(pipeline)
    second = migrate_output_recipe(
        pipeline,
        expected_source_fingerprint=current.source_fingerprint,
    )
    assert second.applied is False
    assert sorted(result.backup_path.parent.iterdir()) == artifacts_before


def test_migration_refuses_changed_source_without_writing_backup(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    report = preflight_output_compatibility(pipeline)
    pipeline.write_text(
        pipeline.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CompatibilityMigrationError, match="changed"):
        migrate_output_recipe(
            pipeline,
            expected_source_fingerprint=report.source_fingerprint,
        )

    assert not (pipeline.parent / ".migration_backups").exists()


def test_migration_receipt_failure_rolls_back_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed applied receipt cannot leave a migrated source behind."""
    pipeline = _historical_pipeline(tmp_path)
    original = pipeline.read_bytes()
    report = preflight_output_compatibility(pipeline)
    real_atomic_write_json = _compatibility.atomic_write_json
    receipt_writes = 0

    def _fail_final_receipt(path, payload, **kwargs):
        nonlocal receipt_writes
        receipt_writes += 1
        if receipt_writes == 2:
            raise OSError("simulated receipt failure")
        return real_atomic_write_json(path, payload, **kwargs)

    monkeypatch.setattr(
        _compatibility,
        "atomic_write_json",
        _fail_final_receipt,
    )

    with pytest.raises(CompatibilityMigrationError, match="rolled back"):
        migrate_output_recipe(
            pipeline,
            expected_source_fingerprint=report.source_fingerprint,
        )

    assert pipeline.read_bytes() == original
    assert preflight_output_compatibility(pipeline).status == "migratable"
    receipts = list((tmp_path / ".migration_backups").glob("*.migration.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["state"] == "prepared"


def test_concurrent_migrations_use_one_locked_cas_publication(
    tmp_path: Path,
) -> None:
    """Two writers with one source fingerprint can publish only once."""
    pipeline = _historical_pipeline(tmp_path)
    report = preflight_output_compatibility(pipeline)

    def _migrate() -> str:
        try:
            result = migrate_output_recipe(
                pipeline,
                expected_source_fingerprint=report.source_fingerprint,
            )
        except CompatibilityMigrationError:
            return "refused"
        return "applied" if result.applied else "unchanged"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: _migrate(), range(2)))

    assert sorted(outcomes) == ["applied", "refused"]
    assert preflight_output_compatibility(pipeline).status == "compatible"
    receipts = list((tmp_path / ".migration_backups").glob("*.migration.json"))
    backups = list((tmp_path / ".migration_backups").glob("*.bak"))
    assert len(receipts) == 1
    assert len(backups) == 1


def test_cli_writer_generation_cannot_be_overwritten_by_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A migration waiting behind a CLI publish must fail its old CAS."""
    pipeline = _historical_output_pipeline(tmp_path)
    report = preflight_output_compatibility(pipeline)
    writer_holds_lock = Event()
    release_writer = Event()
    migration_attempted_lock = Event()
    real_atomic_write = _cli_output_manager.atomic_write_with_writer
    real_migration_lock = _compatibility.pipeline_publication_lock

    def _blocked_cli_write(path, writer):
        writer_holds_lock.set()
        assert release_writer.wait(timeout=5)
        return real_atomic_write(path, writer)

    @contextmanager
    def _observed_migration_lock(path):
        migration_attempted_lock.set()
        with real_migration_lock(path):
            yield

    monkeypatch.setattr(
        _cli_output_manager,
        "atomic_write_with_writer",
        _blocked_cli_write,
    )
    monkeypatch.setattr(
        _compatibility,
        "pipeline_publication_lock",
        _observed_migration_lock,
    )
    ordinary = ImagePipeline(name="ordinary-writer")

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer_future = executor.submit(
            _cli_output_manager._persist_pipeline_to_output_dir,
            tmp_path,
            ordinary,
        )
        assert writer_holds_lock.wait(timeout=5)
        migration_future = executor.submit(
            migrate_output_recipe,
            pipeline,
            expected_source_fingerprint=report.source_fingerprint,
        )
        assert migration_attempted_lock.wait(timeout=5)
        assert not migration_future.done()
        release_writer.set()

        assert writer_future.result(timeout=5) == pipeline
        with pytest.raises(CompatibilityMigrationError, match="changed"):
            migration_future.result(timeout=5)

    assert pipeline.read_text(encoding="utf-8") == ordinary.to_json()
    assert not (pipeline.parent / ".migration_backups").exists()


def test_legacy_migration_serializes_with_typed_cli_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A V1 migration re-resolves after a waiting V2 writer publishes."""
    legacy_pipeline = _historical_legacy_output_pipeline(tmp_path)
    report = preflight_output_compatibility(legacy_pipeline)
    typed_pipeline = pipeline_json_path(tmp_path)
    writer_holds_lock = Event()
    release_writer = Event()
    migration_attempted_lock = Event()
    real_atomic_write = _cli_output_manager.atomic_write_with_writer
    real_migration_lock = _compatibility.pipeline_publication_lock

    def _blocked_cli_write(path, writer):
        writer_holds_lock.set()
        assert release_writer.wait(timeout=5)
        return real_atomic_write(path, writer)

    @contextmanager
    def _observed_migration_lock(path):
        migration_attempted_lock.set()
        with real_migration_lock(path):
            yield

    monkeypatch.setattr(
        _cli_output_manager,
        "atomic_write_with_writer",
        _blocked_cli_write,
    )
    monkeypatch.setattr(
        _compatibility,
        "pipeline_publication_lock",
        _observed_migration_lock,
    )
    ordinary = ImagePipeline(name="typed-writer")

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer_future = executor.submit(
            _cli_output_manager._persist_pipeline_to_output_dir,
            tmp_path,
            ordinary,
        )
        assert writer_holds_lock.wait(timeout=5)
        migration_future = executor.submit(
            migrate_output_recipe,
            tmp_path,
            expected_source_fingerprint=report.source_fingerprint,
        )
        assert migration_attempted_lock.wait(timeout=5)
        assert not migration_future.done()
        release_writer.set()

        assert writer_future.result(timeout=5) == typed_pipeline
        with pytest.raises(CompatibilityMigrationError, match="changed"):
            migration_future.result(timeout=5)

    assert typed_pipeline.read_text(encoding="utf-8") == ordinary.to_json()
    assert preflight_output_compatibility(legacy_pipeline).status == "migratable"
    assert not (legacy_pipeline.parent / ".migration_backups").exists()


def test_receipt_rollback_cannot_overwrite_waiting_cli_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback completes under the lock before a waiting CLI generation."""
    pipeline = _historical_output_pipeline(tmp_path)
    report = preflight_output_compatibility(pipeline)
    final_receipt_started = Event()
    release_receipt_failure = Event()
    cli_attempted_lock = Event()
    real_atomic_write_json = _compatibility.atomic_write_json
    real_cli_lock = _cli_output_manager.pipeline_publication_lock
    receipt_writes = 0

    def _fail_final_receipt(path, payload, **kwargs):
        nonlocal receipt_writes
        receipt_writes += 1
        if receipt_writes == 2:
            final_receipt_started.set()
            assert release_receipt_failure.wait(timeout=5)
            raise OSError("simulated receipt failure")
        return real_atomic_write_json(path, payload, **kwargs)

    @contextmanager
    def _observed_cli_lock(path):
        cli_attempted_lock.set()
        with real_cli_lock(path):
            yield

    monkeypatch.setattr(
        _compatibility,
        "atomic_write_json",
        _fail_final_receipt,
    )
    monkeypatch.setattr(
        _cli_output_manager,
        "pipeline_publication_lock",
        _observed_cli_lock,
    )
    ordinary = ImagePipeline(name="post-rollback-writer")

    with ThreadPoolExecutor(max_workers=2) as executor:
        migration_future = executor.submit(
            migrate_output_recipe,
            pipeline,
            expected_source_fingerprint=report.source_fingerprint,
        )
        assert final_receipt_started.wait(timeout=5)
        writer_future = executor.submit(
            _cli_output_manager._persist_pipeline_to_output_dir,
            tmp_path,
            ordinary,
        )
        assert cli_attempted_lock.wait(timeout=5)
        assert not writer_future.done()
        release_receipt_failure.set()

        with pytest.raises(CompatibilityMigrationError, match="rolled back"):
            migration_future.result(timeout=5)
        assert writer_future.result(timeout=5) == pipeline

    assert pipeline.read_text(encoding="utf-8") == ordinary.to_json()

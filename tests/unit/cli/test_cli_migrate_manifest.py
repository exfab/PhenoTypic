"""Behavior tests for the indexed ``--mode migrate`` work inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """Return an empty output-root path for one synthetic migration run."""
    return tmp_path / "run"


def _dataset(run: Path, name: str = "ds") -> Path:
    """Create and return one legacy results dataset."""
    path = run / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hdf(run: Path, stem: str, dataset: str = "ds") -> Path:
    """Add one legacy HDF identity without requiring valid image content."""
    path = _dataset(run, dataset) / "hdf" / f"{stem}.h5"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"legacy hdf")
    return path


def _store(run: Path, stem: str, dataset: str = "ds") -> Path:
    """Add one OME-Zarr directory identity."""
    path = _dataset(run, dataset) / "zarr" / f"{stem}.ome.zarr"
    path.mkdir(parents=True)
    return path


def _table(run: Path, stem: str, dataset: str = "ds") -> Path:
    """Add one external per-image measurement table identity."""
    path = _dataset(run, dataset) / "measurements" / f"{stem}.parquet"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"external table")
    return path


def test_inventory_unifies_partial_states_by_canonical_identity(run: Path) -> None:
    """HDF, store, and table evidence is one deterministic work inventory."""
    from phenotypic._cli._cli_migrate_manifest import discover_migration_tasks

    hdf = _hdf(run, "hdf_only")
    resumed_store = _store(run, "hdf_only")
    store = _store(run, "store_only")
    table_hdf = _hdf(run, "table_only_with_hdf")
    table = _table(run, "table_only_with_hdf")

    tasks = discover_migration_tasks(run)

    assert [(task.dataset, task.stem) for task in tasks] == [
        ("ds", "hdf_only"),
        ("ds", "store_only"),
        ("ds", "table_only_with_hdf"),
    ]
    assert tasks[0].hdf_path == hdf.resolve()
    assert tasks[0].store_path == resumed_store.resolve()
    assert tasks[0].measurement_path is None
    assert tasks[1].hdf_path is None
    assert tasks[1].store_path == store.resolve()
    assert tasks[1].store_path.name == "store_only.ome.zarr"
    assert tasks[2].hdf_path == table_hdf.resolve()
    assert tasks[2].measurement_path == table.resolve()
    assert tasks[2].overlay_path == (
        run / "deliverables" / "overlays" / "ds" / "table_only_with_hdf.png"
    ).resolve()
    assert tasks[2].marker_path == (
        run
        / ".phenotypic"
        / "progress"
        / "image_complete"
        / "ds"
        / "table_only_with_hdf.json"
    ).resolve()


def test_inventory_order_is_dataset_then_stem(run: Path) -> None:
    """Scheduler indexes cannot depend on filesystem iteration order."""
    from phenotypic._cli._cli_migrate_manifest import discover_migration_tasks

    _store(run, "z", "beta")
    _hdf(run, "a", "beta")
    _hdf(run, "b", "alpha")

    tasks = discover_migration_tasks(run)

    assert [(task.index, task.dataset, task.stem) for task in tasks] == [
        (0, "alpha", "b"),
        (1, "beta", "a"),
        (2, "beta", "z"),
    ]


def test_inventory_refuses_measurement_table_without_an_image(run: Path) -> None:
    """A table cannot authorize a migration target on its own."""
    from phenotypic._cli._cli_migrate_manifest import discover_migration_tasks

    _table(run, "orphan")

    with pytest.raises(ValueError, match="measurement-only"):
        discover_migration_tasks(run)


def test_inventory_refuses_symlinked_candidate_that_escapes_run(run: Path) -> None:
    """A symlink must never let a migration manifest name an outside file."""
    from phenotypic._cli._cli_migrate_manifest import discover_migration_tasks

    outside = run.parent / "outside.h5"
    outside.write_bytes(b"not part of the run")
    hdf_dir = _dataset(run) / "hdf"
    hdf_dir.mkdir()
    (hdf_dir / "escape.h5").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        discover_migration_tasks(run)


def test_inventory_refuses_duplicate_artifact_kind_for_identity(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two HDF candidates for one canonical identity are ambiguous."""
    from phenotypic._cli import _cli_migrate_manifest as subject

    hdf = _hdf(run, "duplicate")
    duplicate = hdf.with_name("duplicate-copy.h5")
    duplicate.write_bytes(b"another candidate")
    original = subject._iter_hdf_candidates
    monkeypatch.setattr(subject, "_iter_hdf_candidates", lambda *_: (hdf, duplicate))
    monkeypatch.setattr(subject, "_candidate_stem", lambda _: "duplicate")

    with pytest.raises(ValueError, match="ambiguous"):
        subject.discover_migration_tasks(run)

    monkeypatch.setattr(subject, "_iter_hdf_candidates", original)


def _manifest_path(run: Path) -> Path:
    """Return the fixed private manifest header location for assertions."""
    return run / ".phenotypic" / "migration_manifest.json"


def _fixture_tasks(run: Path, count: int):
    """Create hand-derived task data for framed-record tests."""
    from phenotypic._cli._cli_migrate_manifest import MigrationImageTask

    return tuple(
        MigrationImageTask(
            index=index,
            dataset="ds",
            stem=f"image_{index:04d}",
            hdf_path=None,
            store_path=(run / "results" / "ds" / "zarr" / f"image_{index:04d}.ome.zarr").resolve(),
            measurement_path=None,
            overlay_path=(run / "deliverables" / "overlays" / "ds" / f"image_{index:04d}.png").resolve(),
            marker_path=(run / ".phenotypic" / "progress" / "image_complete" / "ds" / f"image_{index:04d}.json").resolve(),
        )
        for index in range(count)
    )


def _write_fixture_manifest(run: Path, count: int = 3):
    """Write one manifest with a fixed generation for direct-reader tests."""
    from phenotypic._cli._cli_migrate_manifest import write_migration_manifest

    return write_migration_manifest(
        run,
        generation="generation-1",
        scientific_output=(run / "deliverables").resolve(),
        tasks=_fixture_tasks(run, count),
    )


def test_read_seeks_directly_to_requested_index(run: Path) -> None:
    """A worker can load a late array item without decoding earlier records."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    _write_fixture_manifest(run, count=100)

    task = read_migration_task(_manifest_path(run), 73)

    assert task.index == 73
    assert task.stem == "image_0073"


def test_manifest_digest_is_deterministic(run: Path) -> None:
    """Equivalent ordered inventories produce an identical content digest."""
    from phenotypic._cli._cli_migrate_manifest import write_migration_manifest

    first = _write_fixture_manifest(run)
    second = write_migration_manifest(
        run,
        generation="generation-2",
        scientific_output=(run / "deliverables").resolve(),
        tasks=_fixture_tasks(run, 3),
    )

    assert first.inventory_digest == second.inventory_digest


def test_reader_refuses_corrupt_offset_file(run: Path) -> None:
    """Offsets must be a complete aligned u64 index before seeking a record."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    manifest = _write_fixture_manifest(run)
    manifest.offsets_path.write_bytes(b"\0")

    with pytest.raises(ValueError, match="offset"):
        read_migration_task(_manifest_path(run), 0)


def test_reader_refuses_truncated_record_frame(run: Path) -> None:
    """The promised frame length must be physically present in the record file."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    manifest = _write_fixture_manifest(run)
    manifest.records_path.write_bytes(manifest.records_path.read_bytes()[:-10])

    with pytest.raises(ValueError, match="truncated"):
        read_migration_task(_manifest_path(run), 2)


def test_reader_refuses_payload_checksum_mismatch(run: Path) -> None:
    """Bit damage inside a JSON payload cannot silently select wrong work."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    manifest = _write_fixture_manifest(run)
    raw = bytearray(manifest.records_path.read_bytes())
    offset = int.from_bytes(manifest.offsets_path.read_bytes()[:8], "big")
    raw[offset + 8] ^= 1
    manifest.records_path.write_bytes(raw)

    with pytest.raises(ValueError, match="checksum"):
        read_migration_task(_manifest_path(run), 0)


def test_reader_refuses_record_with_wrong_generation(run: Path) -> None:
    """A frame copied from another migration generation is never reusable."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    _write_fixture_manifest(run)
    path = _manifest_path(run)
    header = json.loads(path.read_text(encoding="utf-8"))
    header["generation"] = "generation-2"
    path.write_text(json.dumps(header), encoding="utf-8")

    with pytest.raises(ValueError, match="generation"):
        read_migration_task(path, 0)


def test_reader_refuses_out_of_range_index(run: Path) -> None:
    """The array index is bounded by the manifest's recorded task count."""
    from phenotypic._cli._cli_migrate_manifest import read_migration_task

    _write_fixture_manifest(run, count=3)

    with pytest.raises(IndexError, match="out of range"):
        read_migration_task(_manifest_path(run), 3)

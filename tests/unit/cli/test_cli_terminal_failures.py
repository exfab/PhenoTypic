"""Tests for durable terminal-failure identity and journal semantics."""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import phenotypic._cli._cli_failure_tracker as failure_tracker
from phenotypic._cli._cli_failure_tracker import (
    TerminalFailureJournalError,
    append_terminal_failure,
    compute_work_id,
    migrate_legacy_terminal_failures,
    read_terminal_failures,
    terminal_failure_index,
    work_id_for_image,
)
from phenotypic._cli._cli_process_single import _worker_work_identity
from phenotypic._cli._cli_state_management import get_remaining_images_for_datasets
from phenotypic._cli._cli_types import Dataset, DatasetState, ProcessingState
from phenotypic._cli._cli_completion import publish_image_success
from phenotypic.sdk_ import failures_jsonl_path, terminal_failures_jsonl_path


def _append_in_child(output_dir: str, index: int) -> None:
    append_terminal_failure(
        Path(output_dir),
        work_id=f"work-{index}",
        dataset="plate",
        relative_image_path=f"plate/image-{index}.tif",
        failed_stage="full",
        exception=RuntimeError(f"failure {index}"),
        attempt_id=f"attempt-{index}",
        lifecycle_epoch="epoch",
    )


def _work_id(**overrides: str) -> str:
    fields = {
        "dataset": "plate",
        "relative_image_path": "plate/image.tif",
        "input_sha256": "input-a",
        "pipeline_fingerprint": "pipeline-a",
        "processing_config_digest": "config-a",
        "mode": "full",
    }
    fields.update(overrides)
    return compute_work_id(**fields)


def test_work_id_binds_every_scientific_identity_field() -> None:
    baseline = _work_id()
    assert _work_id(input_sha256="input-b") != baseline
    assert _work_id(pipeline_fingerprint="pipeline-b") != baseline
    assert _work_id(processing_config_digest="config-b") != baseline
    assert _work_id(mode="process") != baseline
    assert _work_id(dataset="other") != baseline
    assert _work_id(relative_image_path="other/image.tif") != baseline


def test_direct_store_relative_identity_is_its_name_not_dot(tmp_path: Path) -> None:
    store = tmp_path / "p01.ome.zarr"

    assert failure_tracker._normalized_input_relative_path(store, store) == Path(
        "p01.ome.zarr"
    )
    assert failure_tracker._normalized_input_relative_path(None, store) == Path(
        "p01.ome.zarr"
    )


def test_submission_and_worker_share_direct_store_identity(tmp_path: Path) -> None:
    store = tmp_path / "p01.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        input_path=store,
        pipeline_json=pipeline,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        process_only_layer=None,
        ext=".tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
        drop_originals=False,
        measure_only=False,
    )

    submission = work_id_for_image(config, "plate", store)
    worker = _worker_work_identity(
        pipeline=pipeline,
        image=store,
        input_root=store,
        dataset_name="plate",
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        layer=None,
        ext=".tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
        mode="full",
    )

    assert submission == worker
    assert submission[1] == "p01.ome.zarr"


def test_append_is_durable_and_index_uses_latest_duplicate(tmp_path: Path) -> None:
    for attempt in ("first", "second"):
        assert append_terminal_failure(
            tmp_path,
            work_id="same-work",
            dataset="plate",
            relative_image_path="plate/image.tif",
            failed_stage="stage2",
            exception=ValueError(attempt),
            attempt_id=attempt,
            lifecycle_epoch="epoch",
        )

    records = read_terminal_failures(tmp_path)
    assert [record.attempt_id for record in records] == ["first", "second"]
    assert terminal_failure_index(tmp_path)["same-work"].attempt_id == "second"


def test_failure_does_not_override_direct_store_canonical_success(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "results" / "plate" / "p01.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"measurement")
    publish_image_success(
        tmp_path,
        work_id="work-p01",
        dataset="plate",
        relative_image_path="p01.ome.zarr",
        image_stem="p01",
        mode="full",
        attempt_id="success",
        lifecycle_epoch="epoch",
        artifacts={"measurements": artifact},
    )

    assert not append_terminal_failure(
        tmp_path,
        work_id="work-p01",
        dataset="plate",
        relative_image_path="p01.ome.zarr",
        failed_stage="full",
        exception=RuntimeError("late worker"),
        attempt_id="late",
        lifecycle_epoch="epoch",
    )
    assert read_terminal_failures(tmp_path) == []


def test_append_preserves_and_skips_killed_partial_line(tmp_path: Path) -> None:
    journal = terminal_failures_jsonl_path(tmp_path)
    journal.parent.mkdir(parents=True)
    journal.write_bytes(b'{"work_id":"partial"')

    assert append_terminal_failure(
        tmp_path,
        work_id="complete-work",
        dataset="plate",
        relative_image_path="plate/image.tif",
        failed_stage="full",
        exception=RuntimeError("complete"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )

    assert [record.work_id for record in read_terminal_failures(tmp_path)] == [
        "complete-work"
    ]
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"work_id":"partial"'
    assert json.loads(lines[1])["work_id"] == "complete-work"


def test_concurrent_process_appends_do_not_lose_records(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_append_in_child, args=(str(tmp_path), index))
        for index in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert {record.work_id for record in read_terminal_failures(tmp_path)} == {
        f"work-{index}" for index in range(8)
    }


def test_legacy_migration_requires_exact_scientific_identity(tmp_path: Path) -> None:
    legacy = failures_jsonl_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    valid = {
        "work_id": "current-work",
        "dataset": "plate",
        "relative_image_path": "plate/image.tif",
        "failed_stage": "detection",
        "exception_type": "RuntimeError",
        "exception_message": "bug",
        "attempt_id": "attempt",
        "lifecycle_epoch": "epoch",
        "timestamp": "2026-08-17T00:00:00+00:00",
            "failure_classification": "per_image_scientific",
            "failure_boundary": "per_image_scientific",
        }
    ambiguous = {"dataset": "plate", "image": "other.tif", "error_type": "ValueError"}
    legacy.write_text(
        json.dumps(ambiguous) + "\n" + json.dumps(valid) + "\n",
        encoding="utf-8",
    )

    assert migrate_legacy_terminal_failures(
        tmp_path, valid_work_ids={"current-work"}
    ) == 1
    records = read_terminal_failures(tmp_path)
    assert [record.work_id for record in records] == ["current-work"]
    assert migrate_legacy_terminal_failures(
        tmp_path, valid_work_ids={"current-work"}
    ) == 0


def test_authoritative_journal_read_failure_aborts_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "phenotypic._cli._cli_failure_tracker.atomic_read",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("storage down")),
    )

    with pytest.raises(TerminalFailureJournalError, match="authoritative"):
        read_terminal_failures(tmp_path)


def test_lock_or_write_failure_leaves_no_terminal_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_append(*_args: object, **_kwargs: object) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(
        "phenotypic._cli._cli_failure_tracker.atomic_append", fail_append
    )
    assert not append_terminal_failure(
        tmp_path,
        work_id="work",
        dataset="plate",
        relative_image_path="plate/image.tif",
        failed_stage="full",
        exception=RuntimeError("scientific failure"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )
    assert read_terminal_failures(tmp_path) == []


def test_memory_error_is_never_terminal(tmp_path: Path) -> None:
    assert not append_terminal_failure(
        tmp_path,
        work_id="work",
        dataset="plate",
        relative_image_path="plate/image.tif",
        failed_stage="full",
        exception=MemoryError("oom"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )
    assert not terminal_failures_jsonl_path(tmp_path).exists()


def test_fsync_failure_rolls_back_uncommitted_terminal_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "phenotypic._cli._cli_file_locking.os.fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("fsync failed")),
    )
    assert not append_terminal_failure(
        tmp_path,
        work_id="work-fsync",
        dataset="plate",
        relative_image_path="plate/image.tif",
        failed_stage="measurement",
        exception=RuntimeError("scientific bug"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )
    assert read_terminal_failures(tmp_path) == []


def test_timeout_error_is_never_terminal(tmp_path: Path) -> None:
    assert not append_terminal_failure(
        tmp_path,
        work_id="work-timeout",
        dataset="plate",
        relative_image_path="plate/image.tif",
        failed_stage="detection",
        exception=TimeoutError("infrastructure timeout"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )
    assert not terminal_failures_jsonl_path(tmp_path).exists()


def test_matching_failure_is_skipped_and_explicit_retry_selects_it(
    tmp_path: Path,
) -> None:
    image = tmp_path / "plate" / "image.tif"
    image.parent.mkdir()
    image.write_bytes(b"image-a")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        input_path=tmp_path,
        pipeline_json=pipeline,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        process_only_layer=None,
        ext=".tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
        drop_originals=False,
        measure_only=False,
    )
    dataset = Dataset("plate", [image], image.parent, tmp_path / "out")
    state = ProcessingState(
        version="2.0.0",
        pipeline_path=pipeline,
        input_path=tmp_path,
        output_dir=tmp_path,
        timestamp=datetime.now(),
        execution_mode="local",
        last_updated=datetime.now(),
        datasets={"plate": DatasetState(initial_images={image.name})},
        config={},
    )
    work_id, relative_path = work_id_for_image(config, "plate", image)
    assert append_terminal_failure(
        tmp_path,
        work_id=work_id,
        dataset="plate",
        relative_image_path=relative_path,
        failed_stage="full",
        exception=RuntimeError("bad image"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )

    assert get_remaining_images_for_datasets(
        state, [dataset], config=config, output_dir=tmp_path
    ) == []
    assert get_remaining_images_for_datasets(
        state,
        [dataset],
        retry_failures=True,
        config=config,
        output_dir=tmp_path,
    )[0].images == [image]

    state.datasets["plate"].completed.add(image.name)
    assert get_remaining_images_for_datasets(
        state,
        [dataset],
        retry_failures=True,
        config=config,
        output_dir=tmp_path,
    ) == []


def test_changed_input_does_not_match_historical_failure(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    image.write_bytes(b"old")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        input_path=tmp_path,
        pipeline_json=pipeline,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        detect_mode="gray",
        process_only_layer=None,
        ext=".tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
        drop_originals=False,
        measure_only=False,
    )
    old_work_id, relative_path = work_id_for_image(config, "plate", image)
    append_terminal_failure(
        tmp_path,
        work_id=old_work_id,
        dataset="plate",
        relative_image_path=relative_path,
        failed_stage="full",
        exception=RuntimeError("old failure"),
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )
    image.write_bytes(b"changed")
    new_work_id, _ = work_id_for_image(config, "plate", image)
    assert new_work_id != old_work_id

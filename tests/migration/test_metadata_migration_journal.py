"""Framed, recoverable metadata-bundle migration journal coverage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from contextlib import contextmanager
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock

import pandas as pd
import pytest

from phenotypic.sdk_ import (
    BundleLayout,
    metadata_migration_authority,
    migrate_metadata_bundle,
    migrate_preflighted_metadata_bundle,
    preflight_metadata_schema,
)
from phenotypic.sdk_._metadata_migration import (
    BUNDLE_DURABLE_TARGET_ROLE,
    NON_IMAGE_KINDS,
    reconcile_metadata_migration_bundle,
)

LEGACY_STRAIN = "MetadataGenetic_Strain"
CANONICAL_STRAIN = "Metadata_Strain"


@pytest.fixture
def migratable_bundle(tmp_path: Path) -> BundleLayout:
    output = tmp_path / "output"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    for index in range(2):
        pd.DataFrame({LEGACY_STRAIN: [f"strain-{index}"]}).to_parquet(
            measurements / f"plate-{index}.parquet", index=False
        )
    pd.DataFrame({LEGACY_STRAIN: ["aggregate-a"]}).to_parquet(
        measurements / "_dataset_aggregated.parquet", index=False
    )
    second = output / "results" / "dataset-2" / "measurements"
    second.mkdir(parents=True)
    pd.DataFrame({LEGACY_STRAIN: ["aggregate-b"]}).to_parquet(
        second / "_dataset_aggregated.parquet", index=False
    )
    return BundleLayout(deliverables_base=deliverables, output_root=output)


@pytest.fixture
def compatible_bundle(tmp_path: Path) -> BundleLayout:
    output = tmp_path / "output"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    measurements = output / "results" / "dataset" / "measurements"
    measurements.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["WT"]}).to_parquet(
        measurements / "plate.parquet", index=False
    )
    pd.DataFrame({CANONICAL_STRAIN: ["aggregate-a"]}).to_parquet(
        measurements / "_dataset_aggregated.parquet", index=False
    )
    second = output / "results" / "dataset-2" / "measurements"
    second.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["aggregate-b"]}).to_parquet(
        second / "_dataset_aggregated.parquet", index=False
    )
    return BundleLayout(deliverables_base=deliverables, output_root=output)


def _decode_frames(path: Path) -> list[dict[str, object]]:
    data = path.read_bytes()
    frames: list[dict[str, object]] = []
    offset = 0
    while offset < len(data):
        (length,) = struct.unpack(">Q", data[offset : offset + 8])
        payload_start = offset + 8
        payload_end = payload_start + length
        payload = data[payload_start:payload_end]
        checksum = data[payload_end : payload_end + 32]
        assert checksum == hashlib.sha256(payload).digest()
        frames.append(json.loads(payload))
        offset = payload_end + 32
    return frames


def _encode_frames(frames: list[dict[str, object]]) -> bytes:
    encoded = bytearray()
    for frame in frames:
        payload = json.dumps(
            frame, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        encoded.extend(struct.pack(">Q", len(payload)))
        encoded.extend(payload)
        encoded.extend(hashlib.sha256(payload).digest())
    return bytes(encoded)


class _SimulatedProcessDeath(BaseException):
    pass


def _interrupt_after_first_target_replace(
    bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
):
    import phenotypic.sdk_._metadata_migration as migration

    report = preflight_metadata_schema(bundle, kinds=NON_IMAGE_KINDS)
    real_publish = migration._publish_temp

    def die_after_replace(temp: Path, target: Path) -> None:
        real_publish(temp, target)
        raise _SimulatedProcessDeath("after target replace")

    monkeypatch.setattr(migration, "_publish_temp", die_after_replace)
    with pytest.raises(_SimulatedProcessDeath, match="after target replace"):
        migrate_preflighted_metadata_bundle(
            bundle, report=report, kinds=NON_IMAGE_KINDS
        )
    monkeypatch.setattr(migration, "_publish_temp", real_publish)
    assert bundle.output_root is not None
    return report, migration._journal_paths(
        bundle.output_root, report.plan_fingerprint
    )


def _forbid_semantic_reparse(monkeypatch: pytest.MonkeyPatch) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    def fail_semantic_parse(*args: object, **kwargs: object):
        raise AssertionError("interrupted authority must precede semantic preflight")

    monkeypatch.setattr(migration, "_preflight_file", fail_semantic_parse)


def _write_historical_v3_journal(
    bundle: BundleLayout, *, terminal: bool
) -> tuple[Path, str]:
    """Write exact schema-3 authority without using the schema-4 constructor."""
    import phenotypic.sdk_._metadata_migration as migration

    assert bundle.output_root is not None
    paths = migration._discover_legacy_bundle_targets(
        bundle, kinds=NON_IMAGE_KINDS
    )
    targets = tuple(
        migration._preflight_file(
            path,
            mixed_table=migration._bundle_target_is_mixed_table(bundle, path),
        )
        for path in paths
    )
    report = migration._report_from_targets(
        str(bundle.deliverables_base), targets
    )
    payload = {
        "schema_version": 3,
        "kinds": sorted(NON_IMAGE_KINDS),
        "scope": "bundle",
        "bundle_root": str(bundle.output_root),
        "state": "applied" if terminal else "prepared",
        "source": report.source,
        "source_fingerprint": report.source_fingerprint,
        "plan_fingerprint": report.plan_fingerprint,
        "targets": [
            {
                **asdict(target),
                "state": (
                    "skipped"
                    if target.status == "compatible"
                    else "pending"
                ),
                "post_fingerprint": None,
                "rollback_fingerprint": None,
                "temp_path": None,
                "backup_path": None,
                "hdf_snapshot": None,
            }
            for target in report.targets
        ],
    }
    plan_path, log_path, receipt_path = migration._journal_paths(
        bundle.output_root, report.plan_fingerprint
    )
    plan = dict(payload)
    plan["state"] = "prepared"
    plan["journal_schema_version"] = 1
    migration._write_journal_plan(
        bundle.output_root,
        plan_path, log_path, plan, commit_guard=None
    )
    if terminal:
        migration._write_receipt(receipt_path, payload)
    digest = "sha256:" + hashlib.sha256(
        receipt_path.read_bytes() if terminal else plan_path.read_bytes()
    ).hexdigest()
    return receipt_path, digest


def test_journal_frames_are_ordered_and_terminal_receipt_is_compact(
    migratable_bundle: BundleLayout,
) -> None:
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )

    result = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.status == "applied"
    assert result.receipt_path is not None
    assert result.receipt_path.name == "receipt.json"
    plan_path = result.receipt_path.with_name("plan.json")
    log_path = result.receipt_path.with_name("transitions.log")
    assert plan_path.is_file()
    frames = _decode_frames(log_path)
    assert [frame["sequence"] for frame in frames] == list(range(len(frames)))
    assert [frame["next_state"] for frame in frames] == [
        state
        for _ in range(report.migratable_count)
        for state in ("prepared", "applied")
    ]
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "applied"
    assert all(target["temp_path"] is None for target in receipt["targets"])
    assert "transitions" not in receipt


def test_empty_candidate_directory_after_creation_crash_is_reusable(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after mkdir leaves no authority and retry reuses the directory."""
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    plan_path, _, _ = migration._journal_paths(
        migratable_bundle.output_root, report.plan_fingerprint
    )
    real_ensure = migration._ensure_directory_durable
    crashed = False

    def die_after_directory_creation(path: Path) -> None:
        nonlocal crashed
        real_ensure(path)
        if path == plan_path.parent and not crashed:
            crashed = True
            raise _SimulatedProcessDeath("after journal directory creation")

    monkeypatch.setattr(
        migration, "_ensure_directory_durable", die_after_directory_creation
    )
    with pytest.raises(_SimulatedProcessDeath, match="directory creation"):
        migrate_preflighted_metadata_bundle(
            migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
        )
    assert plan_path.parent.is_dir()
    assert tuple(plan_path.parent.iterdir()) == ()
    monkeypatch.setattr(migration, "_ensure_directory_durable", real_ensure)

    retried = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert retried.status == "applied"


def test_valid_plan_after_publication_crash_initializes_missing_empty_log(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable immutable plan can idempotently initialize its absent log."""
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    plan_path, log_path, receipt_path = migration._journal_paths(
        migratable_bundle.output_root, report.plan_fingerprint
    )
    real_atomic_write = migration.atomic_write_json
    crashed = False

    def die_after_plan_publication(
        path: Path, payload: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal crashed
        real_atomic_write(path, payload, *args, **kwargs)
        if path == plan_path and not crashed:
            crashed = True
            raise _SimulatedProcessDeath("after journal plan publication")

    monkeypatch.setattr(migration, "atomic_write_json", die_after_plan_publication)
    with pytest.raises(_SimulatedProcessDeath, match="plan publication"):
        migrate_preflighted_metadata_bundle(
            migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
        )
    assert plan_path.is_file()
    assert not log_path.exists()
    assert not receipt_path.exists()
    monkeypatch.setattr(migration, "atomic_write_json", real_atomic_write)

    retried = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert retried.status == "applied"
    assert log_path.is_file()


def test_no_plan_candidate_with_unexpected_contents_fails_closed(
    migratable_bundle: BundleLayout,
) -> None:
    """Only a strictly empty pre-plan directory is safe to reuse."""
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    plan_path, _, _ = migration._journal_paths(
        migratable_bundle.output_root, report.plan_fingerprint
    )
    plan_path.parent.mkdir(parents=True)
    (plan_path.parent / "unexpected.bin").write_bytes(b"unowned")

    result = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.status == "failed"
    assert "unexpected" in " ".join(result.conflicts).lower()


def test_terminal_receipt_without_transition_log_fails_closed(
    compatible_bundle: BundleLayout,
) -> None:
    """A receipt can never make a missing journal log look crash-reusable."""

    report = preflight_metadata_schema(
        compatible_bundle, kinds=NON_IMAGE_KINDS
    )
    first = migrate_preflighted_metadata_bundle(
        compatible_bundle, report=report, kinds=NON_IMAGE_KINDS
    )
    assert first.status == "compatible" and first.receipt_path is not None
    log_path = first.receipt_path.with_name("transitions.log")
    log_path.unlink()

    result = reconcile_metadata_migration_bundle(
        compatible_bundle, kinds=NON_IMAGE_KINDS
    )

    assert result is not None and result.status == "failed"
    assert "receipt" in " ".join(result.conflicts).lower()
    assert "log" in " ".join(result.conflicts).lower()


def test_each_complete_transition_is_fsynced_before_target_publication(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    events: list[str] = []
    real_append = migration._append_journal_transition
    real_publish = migration._publish_temp
    real_fsync = migration.os.fsync
    fsync_count = 0

    def record_fsync(fd: int) -> None:
        nonlocal fsync_count
        real_fsync(fd)
        fsync_count += 1

    def record_append(*args: object, **kwargs: object) -> None:
        before = fsync_count
        real_append(*args, **kwargs)
        assert fsync_count > before
        events.append("frame")

    def record_publish(temp: Path, target: Path) -> None:
        assert events[-1] == "frame"
        events.append("publish")
        real_publish(temp, target)

    monkeypatch.setattr(migration.os, "fsync", record_fsync)
    monkeypatch.setattr(migration, "_append_journal_transition", record_append)
    monkeypatch.setattr(migration, "_publish_temp", record_publish)
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )

    result = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.status == "applied"
    assert events == ["frame", "publish", "frame"] * report.migratable_count


def test_many_targets_replay_log_once_and_keep_one_append_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing retained writer state must reintroduce quadratic log reads."""
    import phenotypic.sdk_._metadata_migration as migration

    output = tmp_path / "output"
    deliverables = output / "deliverables"
    deliverables.mkdir(parents=True)
    target_count = 12
    for index in range(target_count):
        measurements = (
            output / "results" / f"dataset-{index}" / "measurements"
        )
        measurements.mkdir(parents=True)
        pd.DataFrame({LEGACY_STRAIN: [f"strain-{index}"]}).to_parquet(
            measurements / "_dataset_aggregated.parquet", index=False
        )
    bundle = BundleLayout(deliverables_base=deliverables, output_root=output)
    report = preflight_metadata_schema(bundle, kinds=NON_IMAGE_KINDS)
    _, log_path, _ = migration._journal_paths(
        output, report.plan_fingerprint
    )
    real_read_bytes = Path.read_bytes
    real_open = Path.open
    decoded_sizes: list[int] = []
    log_open_modes: list[str] = []

    def count_decoded_bytes(path: Path) -> bytes:
        data = real_read_bytes(path)
        if path == log_path:
            decoded_sizes.append(len(data))
        return data

    def count_log_opens(path: Path, *args: object, **kwargs: object):
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if path == log_path:
            log_open_modes.append(mode)
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", count_decoded_bytes)
        scoped.setattr(Path, "open", count_log_opens)
        result = migrate_preflighted_metadata_bundle(
            bundle, report=report, kinds=NON_IMAGE_KINDS
        )

    assert result.status == "applied", result.conflicts
    assert decoded_sizes == [0]
    assert log_open_modes.count("r+b") == 1
    assert len(_decode_frames(log_path)) == target_count * 2


def test_concurrent_journal_writer_is_refused(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing exclusive ownership must let both writers choose sequence 0."""
    import phenotypic.sdk_._metadata_migration as migration

    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    real_prepare = migration._prepare_receipt_target
    first_prepare_entered = Event()
    release_first_writer = Event()
    claim_lock = Lock()
    first_prepare_claimed = False

    def block_first_writer(
        target: dict[str, object], receipt_path: Path
    ) -> None:
        nonlocal first_prepare_claimed
        with claim_lock:
            is_first = not first_prepare_claimed
            first_prepare_claimed = True
        if is_first:
            first_prepare_entered.set()
            assert release_first_writer.wait(timeout=10)
        real_prepare(target, receipt_path)

    monkeypatch.setattr(
        migration, "_prepare_receipt_target", block_first_writer
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            migrate_preflighted_metadata_bundle,
            migratable_bundle,
            report=report,
            kinds=NON_IMAGE_KINDS,
        )
        assert first_prepare_entered.wait(timeout=10)
        try:
            competing = migrate_preflighted_metadata_bundle(
                migratable_bundle,
                report=report,
                kinds=NON_IMAGE_KINDS,
            )
        finally:
            release_first_writer.set()
        first = first_future.result(timeout=10)

    assert competing.status == "failed"
    assert "lock" in " ".join(competing.conflicts).lower()
    assert first.status == "applied", first.conflicts


def test_compatible_bundle_reuses_stable_noop_terminal_authority(
    compatible_bundle: BundleLayout,
) -> None:
    first_report = preflight_metadata_schema(
        compatible_bundle, kinds=NON_IMAGE_KINDS
    )
    first = migrate_preflighted_metadata_bundle(
        compatible_bundle, report=first_report, kinds=NON_IMAGE_KINDS
    )
    first_authority = metadata_migration_authority(compatible_bundle)
    first_receipt_bytes = first_authority.terminal_receipt_path.read_bytes()

    second_report = preflight_metadata_schema(
        compatible_bundle, kinds=NON_IMAGE_KINDS
    )
    second = migrate_preflighted_metadata_bundle(
        compatible_bundle, report=second_report, kinds=NON_IMAGE_KINDS
    )
    second_authority = metadata_migration_authority(compatible_bundle)

    assert first.status == second.status == "compatible", second.conflicts
    assert first_authority.compatible_noop is True
    assert second_authority == first_authority
    assert second_authority.terminal_receipt_path.name == "receipt.json"
    assert second_authority.terminal_receipt_path.read_bytes() == first_receipt_bytes


def test_new_compatible_report_reuses_completed_migration_authority(
    migratable_bundle: BundleLayout,
) -> None:
    initial_report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    first = migrate_preflighted_metadata_bundle(
        migratable_bundle,
        report=initial_report,
        kinds=NON_IMAGE_KINDS,
    )
    first_authority = metadata_migration_authority(migratable_bundle)
    compatible_report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    assert compatible_report.status == "compatible"

    second = migrate_preflighted_metadata_bundle(
        migratable_bundle,
        report=compatible_report,
        kinds=NON_IMAGE_KINDS,
    )
    second_authority = metadata_migration_authority(migratable_bundle)

    assert first.status == "applied"
    assert second.status == "compatible"
    assert second.receipt_path == first.receipt_path
    assert second_authority == first_authority
    assert migratable_bundle.output_root is not None
    assert len(
        list(
            (
                migratable_bundle.output_root
                / ".phenotypic"
                / "metadata_migration"
            ).glob("metadata-schema-*")
        )
    ) == 1


def test_process_interruption_resumes_before_fresh_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _ = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "applied"
    assert result.receipt_path is not None and result.receipt_path.is_file()


@pytest.mark.skipif(os.name != "posix", reason="symlink safety contract")
@pytest.mark.parametrize(
    "journal_child",
    ["plan.json", "transitions.log", "receipt.json", ".transitions.log.writer.lock"],
)
def test_interrupted_journal_rejects_child_symlinks_without_touching_victim(
    migratable_bundle: BundleLayout,
    monkeypatch: pytest.MonkeyPatch,
    journal_child: str,
) -> None:
    """Journal children cannot redirect authority I/O outside the bundle."""
    report, (plan_path, _, _) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    child = plan_path.parent / journal_child
    if child.exists():
        child.unlink()
    assert migratable_bundle.output_root is not None
    victim = migratable_bundle.output_root.parent / (
        f"external-{journal_child.replace('.', '-')}.bin"
    )
    victim.write_bytes(b"")
    child.symlink_to(victim)
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "symlink" in " ".join(result.conflicts).lower()
    assert victim.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="symlink safety contract")
@pytest.mark.parametrize(
    "mutable_child",
    ["transitions.log", ".transitions.log.writer.lock"],
)
def test_interrupted_journal_revalidates_mutable_child_at_guarded_seam(
    migratable_bundle: BundleLayout,
    monkeypatch: pytest.MonkeyPatch,
    mutable_child: str,
) -> None:
    """A guarded open or append refuses a journal child swapped to a link."""
    report, (plan_path, _, _) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    child = plan_path.parent / mutable_child
    assert child.is_file()
    retained = plan_path.parent / f"{mutable_child}.retained"
    assert migratable_bundle.output_root is not None
    victim = migratable_bundle.output_root.parent / (
        f"external-raced-{mutable_child.replace('.', '-')}.bin"
    )
    victim.write_bytes(b"")
    swapped = False

    @contextmanager
    def swap_child_on_guard_entry() -> Iterator[None]:
        nonlocal swapped
        if not swapped:
            swapped = True
            child.rename(retained)
            child.symlink_to(victim)
        yield

    _forbid_semantic_reparse(monkeypatch)
    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
        commit_guard=lambda: swap_child_on_guard_entry(),
    )

    assert swapped is True
    assert result.status == "failed"
    assert "symlink" in " ".join(result.conflicts).lower()
    assert victim.read_bytes() == b""


def test_torn_final_frame_replays_only_complete_transitions(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, (_, log_path, receipt_path) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    with log_path.open("ab") as handle:
        handle.write(struct.pack(">Q", 100) + b'{"incomplete":')
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "applied"
    assert receipt_path.is_file()


def test_complete_bad_checksum_fails_closed_before_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, (_, log_path, receipt_path) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    damaged = bytearray(log_path.read_bytes())
    damaged[-1] ^= 0xFF
    log_path.write_bytes(damaged)
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "checksum" in " ".join(result.conflicts).lower()
    assert not receipt_path.exists()


def test_non_monotonic_target_transition_fails_closed(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, (_, log_path, receipt_path) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    frames = _decode_frames(log_path)
    frames[0]["previous_state"] = "applied"
    log_path.write_bytes(_encode_frames(frames))
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "transition" in " ".join(result.conflicts).lower()
    assert not receipt_path.exists()


def test_changed_target_bytes_fail_closed_before_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _ = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    changed = Path(report.targets[0].path)
    pd.DataFrame({CANONICAL_STRAIN: ["externally changed"]}).to_parquet(
        changed, index=False
    )
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "changed" in " ".join(result.conflicts).lower()


def test_changed_authoritative_set_fails_closed_before_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _ = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    assert migratable_bundle.output_root is not None
    extra = (
        migratable_bundle.output_root
        / "results"
        / "extra-dataset"
        / "measurements"
        / "_dataset_aggregated.parquet"
    )
    extra.parent.mkdir(parents=True)
    pd.DataFrame({CANONICAL_STRAIN: ["extra"]}).to_parquet(extra, index=False)
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "target set" in " ".join(result.conflicts).lower()


def test_completed_authority_is_accepted_before_fresh_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    first = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )
    assert first.status == "applied"
    _forbid_semantic_reparse(monkeypatch)

    second = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert second.status == "compatible"
    assert second.receipt_path == first.receipt_path


def test_status_publish_revalidates_live_targets_inside_its_commit_guard(
    migratable_bundle: BundleLayout,
) -> None:
    """Removing the in-guard target check must certify raced target bytes."""
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    _, _, receipt_path = migration._journal_paths(
        migratable_bundle.output_root, report.plan_fingerprint
    )
    status_path = migration._metadata_status_path(
        migratable_bundle.output_root
    )
    raced = False

    @contextmanager
    def race_status_publication() -> Iterator[None]:
        nonlocal raced
        if receipt_path.is_file() and not status_path.exists() and not raced:
            raced = True
            pd.DataFrame({CANONICAL_STRAIN: ["raced"]}).to_parquet(
                Path(report.targets[0].path), index=False
            )
        yield

    with pytest.raises(ValueError, match="fingerprint"):
        migrate_preflighted_metadata_bundle(
            migratable_bundle,
            report=report,
            kinds=NON_IMAGE_KINDS,
            commit_guard=race_status_publication,
        )

    assert raced is True
    assert not status_path.exists()


def test_authority_loader_revalidates_live_target_bytes(
    migratable_bundle: BundleLayout,
) -> None:
    """Removing semantic receipt validation must accept changed live bytes."""
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    result = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )
    assert result.status == "applied"
    pd.DataFrame({CANONICAL_STRAIN: ["changed"]}).to_parquet(
        Path(report.targets[0].path), index=False
    )

    with pytest.raises(ValueError, match="fingerprint"):
        metadata_migration_authority(migratable_bundle)


def test_competing_journals_fail_closed(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, (plan_path, _, _) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    competitor = plan_path.parent.with_name(plan_path.parent.name + "-competitor")
    shutil.copytree(plan_path.parent, competitor)
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "failed"
    assert "competing" in " ".join(result.conflicts).lower()


@pytest.mark.parametrize(
    "damage",
    ["malformed", "receipt", "plan", "source", "resulting"],
)
def test_reconcile_refuses_existing_conflicting_status_authority(
    migratable_bundle: BundleLayout,
    damage: str,
) -> None:
    """Removing status reconciliation must silently overwrite this conflict."""
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    first = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )
    assert first.status == "applied"
    status_path = migration._metadata_status_path(
        migratable_bundle.output_root
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if damage == "malformed":
        damaged_bytes = b"{not-json"
    else:
        if damage == "receipt":
            payload["terminal_receipt_path"] = str(
                status_path.parent / "missing-receipt.json"
            )
        else:
            payload[f"{damage}_fingerprint"] = "sha256:" + "0" * 64
        damaged_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    status_path.write_bytes(damaged_bytes)

    reconciled = reconcile_metadata_migration_bundle(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )

    assert reconciled is not None and reconciled.status == "failed"
    assert "competing" in " ".join(reconciled.conflicts).lower()
    assert status_path.read_bytes() == damaged_bytes


def test_interrupted_legacy_receipt_replays_before_fresh_semantic_preflight(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    legacy_path = migration._receipt_path(
        migratable_bundle.output_root,
        report.plan_fingerprint,
        bundle=True,
    )
    legacy = migration._new_receipt(
        report,
        bundle_root=migratable_bundle.output_root,
        kinds=NON_IMAGE_KINDS,
    )
    migration._write_receipt(legacy_path, legacy)
    _forbid_semantic_reparse(monkeypatch)

    result = migrate_metadata_bundle(
        migratable_bundle,
        expected_plan_fingerprint=report.plan_fingerprint,
        kinds=NON_IMAGE_KINDS,
    )

    assert result.status == "applied"
    assert result.receipt_path == legacy_path
    assert metadata_migration_authority(migratable_bundle).terminal_receipt_path == legacy_path


def test_terminal_schema3_authority_is_validated_then_superseded_by_v4(
    compatible_bundle: BundleLayout,
) -> None:
    """A historical terminal receipt is preserved and digest-bound by v4."""
    old_receipt, old_digest = _write_historical_v3_journal(
        compatible_bundle, terminal=True
    )

    result = reconcile_metadata_migration_bundle(
        compatible_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )

    assert result is not None and result.status in {"compatible", "applied"}
    authority = metadata_migration_authority(compatible_bundle)
    assert authority.terminal_receipt_path != old_receipt
    superseding = json.loads(
        authority.terminal_receipt_path.read_text(encoding="utf-8")
    )
    assert superseding["schema_version"] == 4
    assert superseding["target_role"] == "bundle_durable"
    assert superseding["supersedes_digest"] == old_digest
    assert old_receipt.is_file()
    rerun = reconcile_metadata_migration_bundle(
        compatible_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )
    assert rerun is not None and rerun.status == "compatible"

    old_receipt.write_bytes(b"{corrupt historical authority")
    with pytest.raises(ValueError, match="supersed|digest|historical"):
        metadata_migration_authority(compatible_bundle)


def test_v4_adoption_does_not_revalidate_obsolete_v3_live_targets(
    compatible_bundle: BundleLayout,
) -> None:
    """One-time v3 adoption survives later Task-1 source reclamation."""
    old_receipt, _ = _write_historical_v3_journal(
        compatible_bundle, terminal=True
    )
    adopted = reconcile_metadata_migration_bundle(
        compatible_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )
    assert adopted is not None and adopted.status in {"compatible", "applied"}
    assert compatible_bundle.output_root is not None
    per_image = (
        compatible_bundle.output_root
        / "results"
        / "dataset"
        / "measurements"
        / "plate.parquet"
    )
    per_image.unlink()

    authority = metadata_migration_authority(compatible_bundle)
    rerun = reconcile_metadata_migration_bundle(
        compatible_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )

    assert authority.terminal_receipt_path != old_receipt
    assert rerun is not None and rerun.status == "compatible"


@pytest.mark.parametrize("historical_state", ["missing", "tampered"])
def test_v4_adoption_still_requires_digest_bound_historical_receipt(
    compatible_bundle: BundleLayout, historical_state: str
) -> None:
    """Immutable v3 existence and digest remain part of adopted authority."""
    old_receipt, _ = _write_historical_v3_journal(
        compatible_bundle, terminal=True
    )
    adopted = reconcile_metadata_migration_bundle(
        compatible_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )
    assert adopted is not None and adopted.status in {"compatible", "applied"}
    if historical_state == "missing":
        old_receipt.unlink()
    else:
        old_receipt.write_bytes(b"{tampered historical authority")

    with pytest.raises(ValueError, match="supersed|digest|historical"):
        metadata_migration_authority(compatible_bundle)


def test_interrupted_schema3_authority_recovers_then_is_superseded_by_v4(
    migratable_bundle: BundleLayout,
) -> None:
    """Recovery completes under v3 discovery before durable v4 preflight."""
    old_receipt, _ = _write_historical_v3_journal(
        migratable_bundle, terminal=False
    )

    result = reconcile_metadata_migration_bundle(
        migratable_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )

    assert result is not None and result.status in {"compatible", "applied"}
    assert old_receipt.is_file()
    old_digest = "sha256:" + hashlib.sha256(old_receipt.read_bytes()).hexdigest()
    authority = metadata_migration_authority(migratable_bundle)
    superseding = json.loads(
        authority.terminal_receipt_path.read_text(encoding="utf-8")
    )
    assert superseding["schema_version"] == 4
    assert superseding["supersedes_digest"] == old_digest


def test_competing_schema3_authority_fails_before_v4_supersession(
    compatible_bundle: BundleLayout,
) -> None:
    old_receipt, _ = _write_historical_v3_journal(
        compatible_bundle, terminal=True
    )
    competitor = old_receipt.parent.with_name(old_receipt.parent.name + "-copy")
    shutil.copytree(old_receipt.parent, competitor)

    result = reconcile_metadata_migration_bundle(
        compatible_bundle, kinds=NON_IMAGE_KINDS
    )

    assert result is not None and result.status == "failed"
    assert "competing" in " ".join(result.conflicts).lower()
    assert not (compatible_bundle.output_root / ".phenotypic" / "metadata_migration" / "status.json").exists()


def test_resume_revalidates_authoritative_set_inside_first_commit_guard(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, (_, log_path, _) = _interrupt_after_first_target_replace(
        migratable_bundle, monkeypatch
    )
    before = log_path.read_bytes()
    entered = False

    @contextmanager
    def change_set_on_guard_entry() -> Iterator[None]:
        nonlocal entered
        if not entered:
            entered = True
            assert migratable_bundle.output_root is not None
            extra = (
                migratable_bundle.output_root
                / "results"
                / "raced-dataset"
                / "measurements"
                / "_dataset_aggregated.parquet"
            )
            extra.parent.mkdir(parents=True)
            pd.DataFrame({CANONICAL_STRAIN: ["raced"]}).to_parquet(
                extra, index=False
            )
        yield

    result = migrate_preflighted_metadata_bundle(
        migratable_bundle,
        report=report,
        kinds=NON_IMAGE_KINDS,
        commit_guard=change_set_on_guard_entry,
    )

    assert result.status == "failed"
    assert "target set" in " ".join(result.conflicts).lower()
    assert log_path.read_bytes() == before


def test_authoritative_discovery_is_constant_not_per_transition(
    migratable_bundle: BundleLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    report = preflight_metadata_schema(
        migratable_bundle,
        kinds=NON_IMAGE_KINDS,
        target_role=BUNDLE_DURABLE_TARGET_ROLE,
    )
    real_discover = migration._discover_bundle_targets
    discoveries = 0

    def count_discovery(*args: object, **kwargs: object):
        nonlocal discoveries
        discoveries += 1
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(migration, "_discover_bundle_targets", count_discovery)

    result = migrate_preflighted_metadata_bundle(
        migratable_bundle, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.status == "applied"
    # Includes the mandatory full target validation immediately before the
    # terminal status replace; the count remains independent of transitions.
    assert discoveries == 6


def test_legacy_receipt_replay_holds_commit_guard_for_every_replace(
    migratable_bundle: BundleLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phenotypic.sdk_._metadata_migration as migration

    assert migratable_bundle.output_root is not None
    report = preflight_metadata_schema(
        migratable_bundle, kinds=NON_IMAGE_KINDS
    )
    legacy_path = migration._receipt_path(
        migratable_bundle.output_root,
        report.plan_fingerprint,
        bundle=True,
    )
    legacy = migration._new_receipt(
        report,
        bundle_root=migratable_bundle.output_root,
        kinds=NON_IMAGE_KINDS,
    )
    migration._write_receipt(legacy_path, legacy)
    guard_depth = 0
    replacements = 0
    real_replace = migration.os.replace

    @contextmanager
    def commit_guard() -> Iterator[None]:
        nonlocal guard_depth
        guard_depth += 1
        try:
            yield
        finally:
            guard_depth -= 1

    def require_guarded_replace(source: object, target: object) -> None:
        nonlocal replacements
        assert guard_depth > 0
        replacements += 1
        real_replace(source, target)

    monkeypatch.setattr(migration.os, "replace", require_guarded_replace)

    result = reconcile_metadata_migration_bundle(
        migratable_bundle,
        kinds=NON_IMAGE_KINDS,
        commit_guard=commit_guard,
    )

    assert result is not None and result.status == "applied"
    assert replacements > 0

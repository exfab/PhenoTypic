"""Framed, recoverable metadata-bundle migration journal coverage."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

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
        "prepared",
        "applied",
        "prepared",
        "applied",
    ]
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "applied"
    assert all(target["temp_path"] is None for target in receipt["targets"])
    assert "transitions" not in receipt


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
    assert events == ["frame", "publish", "frame"] * 2


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
    extra = Path(report.targets[0].path).with_name("extra.parquet")
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
            extra = Path(report.targets[0].path).with_name("raced.parquet")
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
        migratable_bundle, kinds=NON_IMAGE_KINDS
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
    assert discoveries == 5


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

"""Focused contracts for cancellable, cached Results output discovery."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli._cli_update_state import append_event
from phenotypic.gui.results_viewer import (
    OutputConsistencyReport,
    OutputDiscoveryCancellation as PublicCancellation,
    OutputDiscoveryProgress as PublicProgress,
    OutputRoot as PublicOutputRoot,
)
from phenotypic.gui.results_viewer._discovery_contracts import (
    OutputDiscoveryCancellation,
    OutputDiscoveryCancelledError,
    OutputDiscoveryProgress,
)
from phenotypic.gui.results_viewer._output_consistency import (
    OutputCompletionEvidence,
    classify_output_consistency,
    inspect_output_consistency,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._processing_inventory import (
    processing_inventory_cache_path,
)
from phenotypic.schema import METADATA
from phenotypic.sdk_ import (
    BundleLayout,
    ProcessingStateKey,
    event_log_path,
    master_measurements_parquet_path,
    measurements_parquet_path,
    gui_launch_owner_path,
    processing_state_path,
    resolve_manifest_json_path,
    run_completion_marker_path,
)


def _seed_output(root: Path, *, overlay_count: int = 2) -> None:
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["plate"] * 2,
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "Size_Area": [10.0, 20.0],
        }
    )
    master = master_measurements_parquet_path(root)
    master.parent.mkdir(parents=True)
    frame.write_parquet(master)
    frame.write_parquet(measurements_parquet_path(root))
    overlays = root / "deliverables" / "overlays" / "plate"
    overlays.mkdir(parents=True)
    for index in range(overlay_count):
        (overlays / f"image-{index}.png").write_bytes(b"overlay")
    hdf = root / "results" / "plate" / "hdf"
    hdf.mkdir(parents=True)
    (hdf / "a.h5").write_bytes(b"hdf")


def _publish_coherent_manifest(root: Path) -> None:
    manifest = resolve_manifest_json_path(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": True,
                "completed": 2,
                "failed": 0,
                "total_images": 2,
            }
        ),
        encoding="utf-8",
    )


def test_pure_consistency_classification_covers_all_states() -> None:
    coherent = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            manifest_present=True,
            manifest_is_complete=True,
            manifest_completed=2,
            manifest_failed=0,
            manifest_total=2,
        )
    )
    active = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            owner_status="running",
        )
    )
    incomplete = classify_output_consistency(
        OutputCompletionEvidence(standalone_bundle=False)
    )
    contradictory = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            manifest_present=True,
            manifest_is_complete=False,
            manifest_completed=4,
            manifest_failed=1,
            manifest_total=2,
            staged_marker_present=True,
            staged_marker_valid=True,
        )
    )
    unreadable_present_evidence = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            manifest_present=True,
            manifest_is_complete=True,
            manifest_completed=2,
            manifest_failed=0,
            manifest_total=2,
            completion_marker_present=True,
            completion_marker_valid=False,
        )
    )
    active_contradiction = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            owner_status="running",
            manifest_present=True,
            manifest_is_complete=True,
            manifest_completed=2,
            manifest_failed=0,
            manifest_total=2,
        )
    )
    unreadable_owner = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            owner_present=True,
            owner_readable=False,
            manifest_present=True,
            manifest_is_complete=True,
            manifest_completed=2,
            manifest_failed=0,
            manifest_total=2,
        )
    )
    unknown_owner = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            owner_present=True,
            owner_status="future-state",
            manifest_present=True,
            manifest_is_complete=True,
            manifest_completed=2,
            manifest_failed=0,
            manifest_total=2,
        )
    )

    assert coherent.state == "coherent"
    assert coherent.cache_reusable is True
    assert active.state == "active"
    assert incomplete.state == "incomplete"
    assert contradictory.state == "contradictory"
    assert contradictory.is_read_only is True
    assert unreadable_present_evidence.state == "incomplete"
    assert unreadable_present_evidence.cache_reusable is False
    assert active_contradiction.state == "contradictory"
    assert active_contradiction.has_active_owner is True
    assert unreadable_owner.state == "incomplete"
    assert unreadable_owner.reasons == ("output owner record is unreadable",)
    assert unknown_owner.state == "incomplete"
    assert unknown_owner.reasons == (
        "output owner status is missing or unknown",
    )


def test_active_owner_tolerates_nonterminal_manifest_event_lag() -> None:
    report = classify_output_consistency(
        OutputCompletionEvidence(
            standalone_bundle=False,
            owner_present=True,
            owner_status="running",
            manifest_present=True,
            manifest_is_complete=False,
            manifest_completed=1,
            manifest_failed=0,
            manifest_total=2,
            processing_state_present=True,
            processing_event_log_present=True,
            processing_total=2,
            processing_completed=2,
            processing_failed=0,
            processing_unfinished=0,
        )
    )

    assert report.state == "active"
    assert report.reasons == ("a nonterminal GUI owner is active",)


def test_o2_discovery_contracts_are_publicly_importable() -> None:
    assert PublicCancellation is OutputDiscoveryCancellation
    assert PublicProgress is OutputDiscoveryProgress
    assert PublicOutputRoot is OutputRoot
    assert OutputConsistencyReport.__name__ == "OutputConsistencyReport"


def test_coherent_terminal_inventory_persists_and_reuses_externally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source)
    _publish_coherent_manifest(source)
    selected_before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    first = OutputRoot.discover(source, cache_root=cache_root)
    second = OutputRoot.discover(source, cache_root=cache_root)

    cache_path = processing_inventory_cache_path(
        source,
        cache_root=cache_root,
    )
    assert first.consistency.state == "coherent"
    assert first.snapshot.processing_inventory_cache_hit is False
    assert second.snapshot.processing_inventory_cache_hit is True
    assert cache_path.is_file()
    assert cache_path.is_relative_to(cache_root)
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == selected_before


def test_terminal_event_log_supersedes_stale_processing_snapshot(
    tmp_path: Path,
) -> None:
    """Fresh CLI output remains coherent when its initial snapshot is stale."""
    source = tmp_path / "output"
    _seed_output(source)
    _publish_coherent_manifest(source)
    state_path = processing_state_path(source)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                ProcessingStateKey.DATASETS: {
                    "plate": {
                        ProcessingStateKey.INITIAL_IMAGES: ["a", "b"],
                        ProcessingStateKey.COMPLETED: [],
                        ProcessingStateKey.FAILED: [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    log_path = event_log_path(source)
    for image_name in ("a", "b"):
        append_event(log_path, "plate", image_name, "started")
        append_event(log_path, "plate", image_name, "completed")
    selected_before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    report = inspect_output_consistency(BundleLayout.detect(source))

    assert report.state == "coherent"
    assert report.evidence.processing_completed == 2
    assert report.evidence.processing_unfinished == 0
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == selected_before


def test_unreadable_processing_event_log_stays_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    _seed_output(source)
    _publish_coherent_manifest(source)
    state_path = processing_state_path(source)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                ProcessingStateKey.DATASETS: {
                    "plate": {
                        ProcessingStateKey.INITIAL_IMAGES: ["a", "b"],
                        ProcessingStateKey.COMPLETED: [],
                        ProcessingStateKey.FAILED: [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    corrupt_log_path = event_log_path(source)
    corrupt_log_path.write_bytes(b"\xff")

    report = inspect_output_consistency(BundleLayout.detect(source))

    assert report.state == "incomplete"
    assert report.evidence.processing_event_log_present is True
    assert report.evidence.processing_event_log_readable is False
    assert "processing event log is unreadable" in report.reasons


def test_mutable_state_is_always_fresh_while_processing_cache_reuses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source)
    _publish_coherent_manifest(source)

    first = OutputRoot.discover(source, cache_root=cache_root)
    mirror = measurements_parquet_path(source)
    pl.read_parquet(mirror).with_columns(
        pl.lit("new").alias("Mutable")
    ).write_parquet(mirror)
    second = OutputRoot.discover(source, cache_root=cache_root)

    assert second.snapshot.processing_inventory_cache_hit is True
    assert second.source_fingerprint == first.source_fingerprint
    assert (
        second.consumed_state_fingerprint
        != first.consumed_state_fingerprint
    )
    assert "Mutable" in second.master_df.columns


def test_changed_processing_product_invalidates_terminal_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source)
    _publish_coherent_manifest(source)
    first = OutputRoot.discover(source, cache_root=cache_root)

    (source / "results" / "plate" / "hdf" / "a.h5").write_bytes(
        b"changed-hdf"
    )
    second = OutputRoot.discover(source, cache_root=cache_root)

    assert second.snapshot.processing_inventory_cache_hit is False
    assert second.source_fingerprint != first.source_fingerprint


def test_incomplete_and_contradictory_outputs_never_persist_inventory(
    tmp_path: Path,
) -> None:
    incomplete_source = tmp_path / "incomplete"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(incomplete_source)

    incomplete = OutputRoot.discover(
        incomplete_source,
        cache_root=cache_root,
    )
    assert incomplete.consistency.state == "incomplete"
    assert incomplete.processing_inventory.assurance == "read_only_bounded"
    assert not processing_inventory_cache_path(
        incomplete_source,
        cache_root=cache_root,
    ).exists()

    contradictory_source = tmp_path / "contradictory"
    _seed_output(contradictory_source)
    manifest = resolve_manifest_json_path(contradictory_source)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": False,
                "completed": 4,
                "failed": 1,
                "total_images": 2,
            }
        ),
        encoding="utf-8",
    )
    marker = run_completion_marker_path(contradictory_source)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "finalizer_succeeded": True,
            }
        ),
        encoding="utf-8",
    )

    contradictory = OutputRoot.discover(
        contradictory_source,
        cache_root=cache_root,
    )
    assert contradictory.consistency.state == "contradictory"
    assert contradictory.processing_inventory.assurance == "read_only_bounded"
    assert contradictory.mutation_snapshot_is_safe() is False
    assert contradictory.consistency.is_read_only is True
    assert contradictory.master_df.height == 2
    assert not processing_inventory_cache_path(
        contradictory_source,
        cache_root=cache_root,
    ).exists()

    owner = gui_launch_owner_path(contradictory_source)
    owner.write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    active_contradiction = OutputRoot.discover(
        contradictory_source,
        cache_root=cache_root,
    )
    assert active_contradiction.consistency.state == "contradictory"
    assert active_contradiction.snapshot.active_run is True


def test_read_only_inventory_never_walks_nested_processing_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only bind work scales with visible images, not artifact entries."""
    source = tmp_path / "large-incomplete"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source)
    image_count = 256
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["plate"] * image_count,
            str(METADATA.IMAGE_NAME): [
                f"image-{index}" for index in range(image_count)
            ],
            "Size_Area": [float(index) for index in range(image_count)],
        }
    )
    frame.write_parquet(master_measurements_parquet_path(source))
    frame.write_parquet(measurements_parquet_path(source))
    nested_results = source / "results"
    deep = nested_results / "plate" / "unrelated" / "deep"
    deep.mkdir(parents=True)
    for index in range(32):
        (deep / f"artifact-{index}.bin").write_bytes(b"unused")

    real_rglob = Path.rglob
    real_stat = Path.stat
    stat_calls = 0

    def _reject_results_walk(path: Path, pattern: str):
        if path == nested_results:
            raise AssertionError("read-only binding recursively walked results/")
        return real_rglob(path, pattern)

    def _count_stat(path: Path, *args, **kwargs):
        nonlocal stat_calls
        stat_calls += 1
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", _reject_results_walk)
    monkeypatch.setattr(Path, "stat", _count_stat)

    output = OutputRoot.discover(source, cache_root=cache_root)

    assert output.consistency.is_read_only
    assert output.processing_inventory.assurance == "read_only_bounded"
    assert len(output.processing_inventory.entries) <= 5
    assert stat_calls <= image_count * 4 + 100
    before_lookup = stat_calls
    assert output.bound_image_source_token("plate", "image-0")
    assert stat_calls == before_lookup

    unrelated_overlay = (
        source
        / "deliverables"
        / "overlays"
        / "plate"
        / "unrelated-new-overlay.png"
    )
    unrelated_overlay.parent.mkdir(parents=True, exist_ok=True)
    unrelated_overlay.write_bytes(b"unrelated")
    assert output.snapshot_is_current() is True


def test_discovery_reports_phases_and_can_cancel_during_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source, overlay_count=300)
    _publish_coherent_manifest(source)
    cancellation = OutputDiscoveryCancellation()
    updates: list[OutputDiscoveryProgress] = []

    def _capture(update: OutputDiscoveryProgress) -> None:
        updates.append(update)
        if update.phase == "inventory" and (update.completed or 0) >= 256:
            cancellation.cancel()

    with pytest.raises(OutputDiscoveryCancelledError):
        OutputRoot.discover(
            source,
            cache_root=cache_root,
            cancellation=cancellation,
            progress_callback=_capture,
        )

    assert updates[0].phase == "classifying"
    assert any(update.phase == "inventory" for update in updates)
    assert not processing_inventory_cache_path(
        source,
        cache_root=cache_root,
    ).exists()


def test_successful_discovery_emits_complete_phase(tmp_path: Path) -> None:
    source = tmp_path / "output"
    _seed_output(source)
    updates: list[OutputDiscoveryProgress] = []

    OutputRoot.discover(
        source,
        cache_root=tmp_path / "sandbox" / ".phenotypic-gui",
        progress_callback=updates.append,
    )

    assert updates[0].phase == "classifying"
    assert updates[-1].phase == "complete"
    assert {"inventory", "measurements", "indexing", "verifying"}.issubset(
        {update.phase for update in updates}
    )
    phase_rank = {
        "classifying": 0,
        "inventory": 1,
        "measurements": 2,
        "indexing": 3,
        "verifying": 4,
        "complete": 5,
    }
    assert [phase_rank[update.phase] for update in updates] == sorted(
        phase_rank[update.phase] for update in updates
    )
    for phase in phase_rank:
        completed = [
            update.completed
            for update in updates
            if update.phase == phase and update.completed is not None
        ]
        assert completed == sorted(completed)


def test_late_cancellation_does_not_publish_terminal_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output"
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    _seed_output(source)
    _publish_coherent_manifest(source)
    cancellation = OutputDiscoveryCancellation()

    def _cancel_after_inventory(update: OutputDiscoveryProgress) -> None:
        if update.detail == "Processing inventory captured.":
            cancellation.cancel()

    with pytest.raises(OutputDiscoveryCancelledError):
        OutputRoot.discover(
            source,
            cache_root=cache_root,
            cancellation=cancellation,
            progress_callback=_cancel_after_inventory,
        )

    assert not processing_inventory_cache_path(
        source,
        cache_root=cache_root,
    ).exists()

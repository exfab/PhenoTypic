"""Focused contracts for cancellable, cached Results output discovery."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic._gui.results_viewer import (
    OutputDiscoveryCancellation as PublicCancellation,
    OutputDiscoveryProgress as PublicProgress,
    OutputRoot as PublicOutputRoot,
)
from phenotypic._gui.results_viewer._discovery_contracts import (
    OutputDiscoveryCancellation,
    OutputDiscoveryCancelledError,
    OutputDiscoveryProgress,
)
from phenotypic._gui.results_viewer._output_root import OutputRoot
from phenotypic._gui.results_viewer._processing_inventory import (
    processing_inventory_cache_path,
)
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    master_measurements_parquet_path,
    measurements_parquet_path,
    gui_launch_owner_path,
    verification_cache_path,
    zarr_store_path,
)
from phenotypic._cli._cli_completion import publish_aggregate_snapshot
from tests._output_layout import build_complete_viewer_run


def _scientific_tree(root: Path) -> dict[Path, bytes]:
    """Every file discovery must leave byte-identical, and no others.

    **One file is excluded, and only one.** Discovery now
    resolves the run state, and a deep pass rewrites the tier-2 verification
    cache at `.phenotypic/verification_cache.json`. That is designed
    behaviour, not a leak: `persist_states` never creates `.phenotypic/`, so a
    tree this package has never written to is still left byte-for-byte alone,
    and a failed write is a return value rather than an exception, so a
    read-only output degrades to a deep pass instead of raising.

    What the viewer still must never touch is the scientific tree --
    `deliverables/`, `results/`, overlays. Narrowing the assertion to those is
    what keeps it meaningful; deleting it because one machine-state file moved
    would have thrown away the guarantee it exists for.
    """
    skip = verification_cache_path(root)
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != skip
    }


def _build_complete(
    root: Path, *, complete: bool = True, overlay_count: int = 0
) -> Path:
    """This file's `plate`/a,b tree, published so its verdict is real.

    `_seed_output` below writes the same shape by hand and deliberately
    publishes nothing, which is what the read-only tests want. This one runs
    the real publishers over it, because a manifest no longer makes a run
    complete -- §4.2 demotes it -- and the only thing that does is a run proof
    over the accepted inventory.
    """
    build_complete_viewer_run(
        root,
        frame=pl.DataFrame(
            {
                "Metadata_Dataset": ["plate"] * 2,
                str(IMAGE.IMAGE_NAME): ["a", "b"],
                "Size_Area": [10.0, 20.0],
            }
        ),
        stems=("a", "b"),
        complete=complete,
    )
    if overlay_count:
        overlays = root / "deliverables" / "overlays" / "plate"
        overlays.mkdir(parents=True, exist_ok=True)
        for index in range(overlay_count):
            (overlays / f"image-{index}.png").write_bytes(b"overlay")
    return root


def _seed_output(root: Path, *, overlay_count: int = 2) -> None:
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate"] * 2,
            str(IMAGE.IMAGE_NAME): ["a", "b"],
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
    # The per-image artifact discovery and the fingerprint contracts read.
    # A store DIRECTORY carrying a root ``zarr.json``: the root is the file
    # whose stat and bytes move on every promote, and the only part of a
    # store any staleness check may key on.
    store = zarr_store_path(root, "plate", "a")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")








def test_o2_discovery_contracts_are_publicly_importable() -> None:
    assert PublicCancellation is OutputDiscoveryCancellation
    assert PublicProgress is OutputDiscoveryProgress
    assert PublicOutputRoot is OutputRoot


def test_coherent_terminal_inventory_persists_and_reuses_externally(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    source = _build_complete(tmp_path / "output")
    selected_before = _scientific_tree(source)

    first = OutputRoot.discover(source, cache_root=cache_root)
    second = OutputRoot.discover(source, cache_root=cache_root)

    cache_path = processing_inventory_cache_path(
        source,
        cache_root=cache_root,
    )
    assert first.run_state is not None
    assert first.run_state.completion == "complete"
    assert first.snapshot.processing_inventory_cache_hit is False
    assert second.snapshot.processing_inventory_cache_hit is True
    assert cache_path.is_file()
    assert cache_path.is_relative_to(cache_root)
    assert _scientific_tree(source) == selected_before








def test_mutable_state_is_always_fresh_while_processing_cache_reuses(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    source = _build_complete(tmp_path / "output")

    first = OutputRoot.discover(source, cache_root=cache_root)
    mirror = measurements_parquet_path(source)
    pl.read_parquet(mirror).with_columns(
        pl.lit("new").alias("Mutable")
    ).write_parquet(mirror)
    # The aggregate proof fences `measurements_parquet` by size and sha256,
    # so rewriting it out of band makes the run non-`core_readable` and
    # discovery refuses it. A re-finalize republishes the proof over the new
    # bytes; without this the fixture describes a tree no writer produces.
    publish_aggregate_snapshot(
        source, source_work_ids=["work-a", "work-b"]
    )
    second = OutputRoot.discover(source, cache_root=cache_root)

    assert second.snapshot.processing_inventory_cache_hit is True
    assert second.source_fingerprint == first.source_fingerprint
    assert (
        second.snapshot.consumed_state_fingerprint
        != first.snapshot.consumed_state_fingerprint
    )
    assert "Mutable" in second.master_df.columns


def test_changed_processing_product_invalidates_terminal_cache(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    source = _build_complete(tmp_path / "output")
    first = OutputRoot.discover(source, cache_root=cache_root)

    (zarr_store_path(source, "plate", "a") / "zarr.json").write_text(
        '{"republished": 1}', encoding="utf-8"
    )
    second = OutputRoot.discover(source, cache_root=cache_root)

    assert second.snapshot.processing_inventory_cache_hit is False
    assert second.source_fingerprint != first.source_fingerprint


def test_an_unfinished_output_never_persists_an_inventory(
    tmp_path: Path,
) -> None:
    """Only a `complete` run may seed the persistent inventory cache.

    Formerly ``test_incomplete_and_contradictory_outputs_never_persist_inventory``.
    Its second half drove the classifier into ``contradictory`` by writing a
    manifest whose counts disagreed with the inventory beside a completion
    marker claiming success. Spec §4.3 deletes ``contradictory`` and §4.2
    demotes both manifest counts and that marker out of the evidence set, so
    that half no longer has a state to reach. What survives is the invariant
    it was really protecting: an output that is not `complete` is bound
    read-only and leaves no cache record behind.
    """
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    incomplete_source = _build_complete(
        tmp_path / "incomplete", complete=False
    )

    incomplete = OutputRoot.discover(
        incomplete_source,
        cache_root=cache_root,
    )
    assert incomplete.run_state is not None
    assert incomplete.run_state.completion == "incomplete"
    assert incomplete.run_is_complete is False
    assert incomplete.processing_inventory.assurance == "read_only_bounded"
    assert incomplete.mutation_snapshot_is_safe() is False
    assert not processing_inventory_cache_path(
        incomplete_source,
        cache_root=cache_root,
    ).exists()

    # An active owner is reported on the snapshot without changing the
    # binding's read-only status -- the two are separate axes, and that
    # separation is what lets the viewer display a running output at all.
    owner = gui_launch_owner_path(incomplete_source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    active = OutputRoot.discover(
        incomplete_source,
        cache_root=cache_root,
    )
    assert active.snapshot.active_run is True
    assert active.run_is_complete is False


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
            "Metadata_Dataset": ["plate"] * image_count,
            str(IMAGE.IMAGE_NAME): [
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

    assert not output.run_is_complete
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
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    source = _build_complete(tmp_path / "output", overlay_count=300)
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
    cache_root = tmp_path / "sandbox" / ".phenotypic-gui" / "viewer_cache"
    source = _build_complete(tmp_path / "output")
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

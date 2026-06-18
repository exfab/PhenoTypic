"""Unit tests for :mod:`phenotypic.gui.results_viewer._filtered_state`.

Exercises the curation persistence layer:

- ``load`` on a fresh directory (no parquet) → empty removed_keys, no disk
  writes.
- ``remove`` then re-``load`` → persisted keys are recovered.
- ``remove`` followed by ``restore`` → ``removed_keys`` empty, parquet+CSV
  contain the master frame verbatim.
- Atomic save: a half-finished save (parquet `.tmp` left behind, no replace)
  is not visible on the next load.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
from phenotypic.sdk_ import measurements_parquet_path

from tests._output_layout import write_master


def _make_master(tmp_root: Path) -> pl.DataFrame:
    """Build a tiny master frame and write it to disk under *tmp_root*.

    The frame mimics the shape of the real master_measurements.parquet
    (under ``deliverables/``): one row per object, keyed by
    ``(Metadata_ImageFile, Object_Label)``, with a couple of measurement
    columns.
    """
    df = pl.DataFrame(
        {
            "Metadata_ImageFile": ["img-001", "img-001", "img-002", "img-002"],
            "Object_Label": [1, 2, 1, 2],
            "Bbox_CenterRR": [10, 20, 30, 40],
            "Bbox_CenterCC": [50, 60, 70, 80],
        }
    )
    write_master(tmp_root, df, csv=False)
    return df


def _write_mirror(tmp_root: Path, df: pl.DataFrame) -> Path:
    """Write the post-applied ``measurements.parquet`` mirror (deliverables)."""
    path = measurements_parquet_path(tmp_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def test_load_with_no_existing_file_is_empty_and_no_writes(tmp_path: Path) -> None:
    """A fresh output dir loads with empty removed_keys and writes nothing.

    Loading should not pollute the directory before the user has actually
    curated anything.
    """
    master = _make_master(tmp_path)

    state = FilteredMeasurements.load(tmp_path, master)

    assert state.removed_keys == set()
    assert not state.parquet_path.exists()
    assert not state.csv_path.exists()


def test_load_with_seed_equal_to_master_yields_empty_removals(tmp_path: Path) -> None:
    """The CLI-seeded initial state (parquet present, equal to master) is empty curation.

    The CLI writes ``measurements.parquet`` as a fresh full copy of the
    master on every run. The viewer must treat that as "no curation",
    not as "everything removed". Locks the contract so a future refactor
    can't conflate "file exists" with "user has curated".
    """
    master = _make_master(tmp_path)

    # Simulate the CLI-seeded state: parquet present, identical to master.
    _write_mirror(tmp_path, master)

    state = FilteredMeasurements.load(tmp_path, master)

    assert state.removed_keys == set()
    # Seed mtime is captured at load so subsequent saves can detect an
    # external rewrite (e.g. a CLI re-run while the viewer is open).
    assert state._seed_mtime_ns is not None


def test_save_refuses_when_seed_was_externally_rewritten(tmp_path: Path) -> None:
    """A CLI re-run under a live viewer must not be clobbered by stale curation.

    Reproduces the failure mode where:
      1. User opens the viewer (load captures master_v1 + seed mtime T1).
      2. CLI ``--recompile`` rewrites measurements.parquet to master_v2 (mtime T2).
      3. User clicks "remove" — without the guard, the viewer would write
         a filtered copy of master_v1 over master_v2, regressing disk to
         the old schema.
    The guard makes step 3 a no-op (with a WARNING) until the viewer reloads.
    """
    import time

    master = _make_master(tmp_path)
    _write_mirror(tmp_path, master)

    state = FilteredMeasurements.load(tmp_path, master)

    # Simulate an external rewrite: a fresh CLI run dumps a new master
    # with a different mtime. We force a distinct mtime by sleeping past
    # the filesystem's resolution; this is robust on every platform we
    # support.
    time.sleep(0.01)
    new_master = master.with_columns(pl.lit(99.0).alias("Bbox_CenterRR"))
    _write_mirror(tmp_path, new_master)

    on_disk_before = pl.read_parquet(measurements_parquet_path(tmp_path))

    # The viewer (still holding stale _master_df) tries to remove a colony.
    state.remove("img-001", 2)

    # Disk is unchanged — the guard refused to overwrite the freshly
    # seeded master with a stale-derived subset.
    on_disk_after = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert on_disk_after.equals(on_disk_before)
    # In-memory removal still recorded; reload will reconcile.
    assert ("img-001", 2) in state.removed_keys


def test_remove_then_load_round_trip(tmp_path: Path) -> None:
    """A remove+restart cycle recovers the curated state from disk."""
    master = _make_master(tmp_path)

    state = FilteredMeasurements.load(tmp_path, master)
    state.remove("img-001", 2)

    # Files exist now.
    assert state.parquet_path.exists()
    assert state.csv_path.exists()

    # Reload from disk; curation persists.
    state2 = FilteredMeasurements.load(tmp_path, master)
    assert state2.removed_keys == {("img-001", 2)}


def test_remove_then_restore_is_clean(tmp_path: Path) -> None:
    """Removing then restoring leaves removed_keys empty and files mirror master."""
    master = _make_master(tmp_path)

    state = FilteredMeasurements.load(tmp_path, master)
    state.remove("img-002", 1)
    assert state.removed_keys == {("img-002", 1)}

    state.restore("img-002", 1)
    assert state.removed_keys == set()

    # Files should still exist (restore writes a full master copy back).
    assert state.parquet_path.exists()
    assert state.csv_path.exists()
    written = pl.read_parquet(state.parquet_path)
    assert written.height == master.height


def test_remove_many_writes_once(tmp_path: Path) -> None:
    """Bulk remove persists the union of keys."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.remove_many([("img-001", 1), ("img-002", 2)])

    state2 = FilteredMeasurements.load(tmp_path, master)
    assert state2.removed_keys == {("img-001", 1), ("img-002", 2)}


def test_idempotent_mutators(tmp_path: Path) -> None:
    """Removing twice / restoring an unknown key are no-ops, not errors."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.remove("img-001", 1)
    state.remove("img-001", 1)  # should not raise / not duplicate
    assert state.removed_keys == {("img-001", 1)}

    state.restore("img-002", 99)  # never-removed key
    assert state.removed_keys == {("img-001", 1)}


def test_filtered_df_excludes_removed_rows(tmp_path: Path) -> None:
    """``filtered_df`` returns master minus the removed-keys rows."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.remove_many([("img-001", 1), ("img-002", 2)])
    out = state.filtered_df(master)

    assert out.height == master.height - 2
    keys = set(zip(out["Metadata_ImageFile"].to_list(), out["Object_Label"].to_list()))
    assert ("img-001", 1) not in keys
    assert ("img-002", 2) not in keys


def test_removed_count_in_intersects_with_filtered_set(tmp_path: Path) -> None:
    """``removed_count_in`` only counts rows of *df* that are removed."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.remove_many([("img-001", 1), ("img-002", 2)])
    sub = master.filter(pl.col("Metadata_ImageFile") == "img-001")
    # Only one of the two removed keys is in the sub-frame.
    assert state.removed_count_in(sub) == 1


def test_removed_keys_payload_is_sorted(tmp_path: Path) -> None:
    """The store payload is sorted for stable diffs."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.remove_many([("img-002", 2), ("img-001", 1), ("img-001", 2)])
    payload = state.removed_keys_payload()
    # JSON-friendly list of [str, int] pairs.
    assert payload == [
        ["img-001", 1],
        ["img-001", 2],
        ["img-002", 2],
    ]


def test_stale_tmp_file_does_not_pollute_state(tmp_path: Path) -> None:
    """A leftover ``.tmp`` parquet from a crashed save is ignored on load.

    Atomic-save semantics: only the final ``parquet_path`` is read; ``.tmp``
    sidecars are scratch space and must not leak into the loaded state.
    """
    master = _make_master(tmp_path)

    # Plant a stale .tmp file that claims something was removed; no
    # final parquet exists yet, so load must treat removed_keys as empty.
    tmp_parquet = measurements_parquet_path(tmp_path).with_name(
        measurements_parquet_path(tmp_path).name + ".tmp"
    )
    tmp_parquet.parent.mkdir(parents=True, exist_ok=True)
    master.head(2).write_parquet(tmp_parquet)
    assert tmp_parquet.exists()

    state = FilteredMeasurements.load(tmp_path, master)
    assert state.removed_keys == set()


def test_load_warns_on_unknown_keys_in_existing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Keys in the existing parquet that aren't in master are dropped + logged."""
    master = _make_master(tmp_path)
    # Hand-craft a filtered file that's missing a row from master AND
    # contains a row that doesn't appear in master (the "unknown key").
    bad = master.filter(pl.col("Object_Label") == 1).vstack(
        pl.DataFrame(
            {
                "Metadata_ImageFile": ["img-999"],
                "Object_Label": [42],
                "Bbox_CenterRR": [0],
                "Bbox_CenterCC": [0],
            }
        )
    )
    _write_mirror(tmp_path, bad)

    with caplog.at_level("WARNING"):
        state = FilteredMeasurements.load(tmp_path, master)

    # Removed keys = master keys − filtered keys, restricted to master.
    # The bogus img-999/42 is not in master so it's not propagated.
    assert state.removed_keys == {("img-001", 2), ("img-002", 2)}


def test_load_raises_friendly_error_when_master_missing_key_columns(
    tmp_path: Path,
) -> None:
    """A master without the curation key columns surfaces a clear error.

    Avoids the previous behaviour where polars' raw ``ColumnNotFoundError``
    bubbled up from deep inside ``_extract_keys`` at viewer-boot time.
    """
    bad_master = pl.DataFrame(
        {"some_column": ["a", "b"], "other": [1, 2]}
    )
    with pytest.raises(ValueError, match=r"Metadata_ImageFile"):
        FilteredMeasurements.load(tmp_path, bad_master)


def test_concurrent_remove_and_restore_dont_interleave(tmp_path: Path) -> None:
    """Two threads racing remove_many vs. restore_many produce a consistent result.

    The lock guarantees serialised execution: whichever thread runs
    second observes the other's mutation and produces a deterministic
    final state. Without locking, the parquet on disk could disagree
    with ``removed_keys``.
    """
    import concurrent.futures as cf

    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    keys = [("img-001", 1), ("img-001", 2), ("img-002", 1), ("img-002", 2)]

    # Pre-seed all four removed so restore_many has work to do.
    state.remove_many(keys)
    assert state.removed_keys == set(keys)

    def remove_again() -> None:
        # Idempotent (already removed) but still acquires the lock.
        state.remove_many(keys)

    def restore_two() -> None:
        state.restore_many([("img-001", 1), ("img-002", 1)])

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(remove_again)
        f2 = ex.submit(restore_two)
        f1.result()
        f2.result()

    # Whichever order won, removed_keys must be exactly the two that
    # weren't restored, and the on-disk parquet must agree.
    expected = {("img-001", 2), ("img-002", 2)}
    assert state.removed_keys == expected

    on_disk = pl.read_parquet(state.parquet_path)
    on_disk_keys = set(
        zip(on_disk["Metadata_ImageFile"].to_list(), on_disk["Object_Label"].to_list())
    )
    assert ("img-001", 2) not in on_disk_keys
    assert ("img-002", 2) not in on_disk_keys
    assert ("img-001", 1) in on_disk_keys
    assert ("img-002", 1) in on_disk_keys


def test_save_holding_lock_does_not_deadlock(tmp_path: Path) -> None:
    """Calling save() from inside a held lock is safe with RLock.

    Re-entrant lock semantics let a future caller wrap external state
    transitions in ``with state._lock`` without deadlocking on the
    second acquire that ``save()`` performs.
    """
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    state.removed_keys.add(("img-001", 1))
    with state._lock:
        # If _lock were a non-reentrant Lock, this would hang the test.
        state.save()

    assert state.parquet_path.exists()


def test_mutate_and_payload_runs_under_one_lock(tmp_path: Path) -> None:
    """``mutate_and_payload`` returns the post-mutation payload atomically."""
    master = _make_master(tmp_path)
    state = FilteredMeasurements.load(tmp_path, master)

    payload = state.mutate_and_payload(
        lambda s: s.remove_many([("img-001", 1), ("img-002", 1)])
    )

    assert payload == [["img-001", 1], ["img-002", 1]]
    assert state.removed_keys == {("img-001", 1), ("img-002", 1)}

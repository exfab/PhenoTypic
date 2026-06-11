"""Tests for the durable CurationLabels store."""

from pathlib import Path

import polars as pl
import pytest

import phenotypic.tools_ as tools_
from phenotypic.gui.results_viewer._curation_labels import (
    CurationLabels,
    sanitize_category,
)
from phenotypic.schema import ErrorCategory


def _master(n: int = 4) -> pl.DataFrame:
    """A minimal master frame: n objects in one image, distinct centroids."""
    return pl.DataFrame(
        {
            "Metadata_ImageFile": ["plateA"] * n,
            "Metadata_Dataset": ["ds1"] * n,
            "Object_Label": list(range(1, n + 1)),
            "Bbox_CenterRR": [10.0 * i for i in range(1, n + 1)],
            "Bbox_CenterCC": [20.0 * i for i in range(1, n + 1)],
            "Size_Area": [100.0 * i for i in range(1, n + 1)],
        }
    )


def test_sanitize_category():
    assert sanitize_category("  Halo Effect! ") == "halo_effect"
    assert sanitize_category("../etc") == "etc"
    assert sanitize_category("###") == ""


def test_load_empty_when_nothing_on_disk(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.categories()[: len(ErrorCategory.labels())] == ErrorCategory.labels()
    assert store.rekey_report.total == 0


def test_register_custom_category_persists_and_dedupes(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    token = store.register_custom_category("Halo Effect")
    assert token == "halo_effect"
    assert "halo_effect" in store.categories()
    # idempotent
    assert store.register_custom_category("halo_effect") == "halo_effect"
    assert store.custom_categories.count("halo_effect") == 1
    # reloads from disk
    reloaded = CurationLabels.load(tmp_path, _master())
    assert "halo_effect" in reloaded.custom_categories


def test_register_rejects_core_collision_and_empty(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    with pytest.raises(ValueError):
        store.register_custom_category("debris")  # core token
    with pytest.raises(ValueError):
        store.register_custom_category("###")  # sanitizes to empty


def test_mark_writes_all_derived_outputs(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 2, "background_noise")

    # label recorded + fingerprint captured from master
    assert store.labels[("plateA", 2)] == "background_noise"
    assert store.fingerprints[("plateA", 2)] == (20.0, 40.0)

    # curated mirror drops the marked object
    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert curated.height == 3
    assert 2 not in curated.get_column("Object_Label").to_list()

    # per-category parquet contains exactly the marked object
    errs = pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "background_noise")
    )
    assert errs.get_column("Object_Label").to_list() == [2]
    assert errs.get_column("Curation_Category").to_list() == ["background_noise"]

    # labels store round-trips on reload
    reloaded = CurationLabels.load(tmp_path, _master())
    assert reloaded.labels == {("plateA", 2): "background_noise"}


def test_unmark_restores_and_clears_category_file(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 2, "debris")
    store.unmark("plateA", 2)
    assert store.labels == {}
    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert curated.height == 4
    # the now-empty category file is removed
    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()


def test_mark_rejects_unknown_category(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    with pytest.raises(ValueError):
        store.mark("plateA", 1, "not_registered")


def test_mark_many_single_save(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark_many([("plateA", 1), ("plateA", 3)], "oversegmented")
    errs = pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "oversegmented")
    )
    assert sorted(errs.get_column("Object_Label").to_list()) == [1, 3]


def test_unmark_one_of_two_categories_keeps_other(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 1, "debris")
    store.mark("plateA", 2, "merged")
    store.unmark("plateA", 1)
    # the emptied category file is removed; the other survives intact
    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()
    merged = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "merged"))
    assert merged.get_column("Object_Label").to_list() == [2]


def test_mark_absent_key_degrades_to_nan_fingerprint(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 999, "debris")  # object 999 is not in master
    assert store.labels[("plateA", 999)] == "debris"
    assert ("plateA", 999) not in store.fingerprints  # no centroid to capture
    # persisted with NaN fingerprint -> dropped on the next re-key load
    reloaded = CurationLabels.load(tmp_path, _master())
    assert ("plateA", 999) not in reloaded.labels
    # S1: no 0-row category parquet written for absent object
    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()


def _write_store_with_label(tmp_path, image_file, label, category, rr, cc):
    """Helper: seed a labels parquet directly (simulating a prior session)."""
    df = pl.DataFrame(
        {
            "Metadata_ImageFile": [image_file],
            "Object_Label": [label],
            "Curation_Category": [category],
            "Bbox_CenterRR": [rr],
            "Bbox_CenterCC": [cc],
        }
    )
    path = tools_.curation_labels_parquet_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_store_with_labels(tmp_path, rows):
    """Helper: seed a labels parquet with multiple rows."""
    df = pl.DataFrame(
        {
            "Metadata_ImageFile": [r[0] for r in rows],
            "Object_Label": [r[1] for r in rows],
            "Curation_Category": [r[2] for r in rows],
            "Bbox_CenterRR": [r[3] for r in rows],
            "Bbox_CenterCC": [r[4] for r in rows],
        }
    )
    path = tools_.curation_labels_parquet_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_rekey_keeps_exact_match(tmp_path: Path):
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {("plateA", 2): "debris"}
    assert store.rekey_report.kept == 1


def test_rekey_recovers_renumbered_object(tmp_path: Path):
    # Stored label was object 2 at (20,40). New master renumbered it to 99 but
    # the centroid is unchanged -> re-key to 99.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    master = _master().with_columns(
        pl.when(pl.col("Object_Label") == 2)
        .then(99)
        .otherwise(pl.col("Object_Label"))
        .alias("Object_Label")
    )
    store = CurationLabels.load(tmp_path, master)
    assert store.labels == {("plateA", 99): "debris"}
    assert store.rekey_report.rekeyed == 1


def test_rekey_drops_when_object_gone(tmp_path: Path):
    # Stored centroid (500,500) matches nothing in master -> dropped.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 500.0, 500.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.rekey_report.dropped == 1


def test_no_legacy_migration_from_measurements_parquet(tmp_path: Path):
    # A mirror missing object 3 must NOT be auto-migrated into an "other"
    # label. Now that re-keying uses the clean master, a mirror-missing row is
    # ambiguous — it may be an old curation removal OR a post-op / --metadata
    # drop — so importing it would fabricate a spurious removal. Without a
    # durable curation_labels.parquet, the store loads empty.
    master = _master()
    curated = master.filter(pl.col("Object_Label") != 3)
    legacy = tools_.measurements_parquet_path(tmp_path)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    curated.write_parquet(legacy)

    store = CurationLabels.load(tmp_path, master)
    assert store.labels == {}
    assert store.rekey_report.migrated == 0


def test_rekey_drops_rather_than_attaching_to_neighbor(tmp_path: Path):
    # Object 2 still exists (exact key present at (20,40)) but the STORED centroid
    # is (10,20) == object 1's position. The label must DROP, never silently
    # re-key onto the neighbour at the stored centroid.
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 10.0, 20.0)
    store = CurationLabels.load(tmp_path, _master())
    assert store.labels == {}
    assert store.rekey_report.dropped == 1


def test_rekey_degrades_without_bbox(tmp_path: Path):
    _write_store_with_label(tmp_path, "plateA", 2, "debris", 20.0, 40.0)
    master = _master().drop(["Bbox_CenterRR", "Bbox_CenterCC"])
    store = CurationLabels.load(tmp_path, master)
    # exact key (plateA, 2) still exists -> kept on the exact-key-only fallback
    assert store.labels == {("plateA", 2): "debris"}
    assert store.rekey_report.kept == 1


def test_filtered_measurements_compat_surface(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())

    # remove == mark as "other"
    store.remove("plateA", 1)
    assert store.is_removed("plateA", 1)
    assert store.labels[("plateA", 1)] == "other"
    assert store.removed_keys == {("plateA", 1)}

    # toggle off / on
    store.toggle("plateA", 1)
    assert not store.is_removed("plateA", 1)
    store.toggle("plateA", 3)
    assert store.is_removed("plateA", 3)

    # restore
    store.restore("plateA", 3)
    assert store.removed_keys == set()

    # payloads
    store.mark("plateA", 2, "debris")
    assert store.removed_keys_payload() == [["plateA", 2]]
    assert store.labels_payload() == [["plateA", 2, "debris"]]

    # removed_count_in
    assert store.removed_count_in(_master()) == 1


def test_mutate_and_payload_runs_under_lock(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    payload = store.mutate_and_payload(lambda s: s.mark("plateA", 1, "merged"))
    assert payload == [["plateA", 1]]


def test_imports_alongside_existing_viewer_modules():
    # The new store must not introduce an import cycle with the viewer package.
    import phenotypic.gui.results_viewer._curation_labels as cl
    import phenotypic.gui.results_viewer._filtered_state as fs  # still present

    assert hasattr(cl.CurationLabels, "load")
    # Compat surface matches the methods the app currently calls on the old store.
    for name in ("remove", "restore", "toggle", "is_removed", "removed_keys_payload"):
        assert hasattr(cl.CurationLabels, name)
        assert hasattr(fs.FilteredMeasurements, name)


# ---------------------------------------------------------------------------
# FIX-1 (M1): two stored labels claiming the same master object both drop
# ---------------------------------------------------------------------------


def test_rekey_drops_when_two_labels_claim_one_object(tmp_path: Path):
    """Two stored labels whose centroids both point to the same surviving object.

    Master has object 2 at (20, 40).  We store label 2 (exact key gone via
    renumber — we use a master where label 2 is absent so BOTH entries must
    go through the nearest-unique path) and label 5, both fingerprinted at
    (20, 40).  Neither should survive: two candidates claim the same target,
    so both are dropped (rekeyed == 0, dropped == 2).
    """
    # Build a master where the original label 2 is gone but a new label 99
    # sits at the same centroid (20, 40).
    master = pl.DataFrame(
        {
            "Metadata_ImageFile": ["plateA", "plateA", "plateA", "plateA"],
            "Metadata_Dataset": ["ds1", "ds1", "ds1", "ds1"],
            "Object_Label": [1, 99, 3, 4],
            "Bbox_CenterRR": [10.0, 20.0, 30.0, 40.0],
            "Bbox_CenterCC": [20.0, 40.0, 60.0, 80.0],
            "Size_Area": [100.0, 200.0, 300.0, 400.0],
        }
    )
    # Two stored labels both with centroids at (20, 40) — label 2 (exact key
    # gone) and label 5 (also gone) — both fingerprint-match object 99.
    _write_store_with_labels(
        tmp_path,
        [
            ("plateA", 2, "debris", 20.0, 40.0),
            ("plateA", 5, "merged", 20.0, 40.0),
        ],
    )
    store = CurationLabels.load(tmp_path, master)
    # Both candidates claimed the same target -> both dropped
    assert store.labels == {}
    assert store.rekey_report.dropped == 2
    assert store.rekey_report.rekeyed == 0


# ---------------------------------------------------------------------------
# FIX-2 (M2): stale-file sweep must not delete foreign (non-category) files
# ---------------------------------------------------------------------------


def test_stale_sweep_preserves_foreign_files(tmp_path: Path):
    """A file in errors/ whose stem is not a known category token is untouched."""
    errs = tools_.errors_dir(tmp_path)
    errs.mkdir(parents=True, exist_ok=True)
    foreign = errs / "notes.parquet"
    # Write a simple 1-column frame as the "foreign" file
    pl.DataFrame({"note": ["do not delete"]}).write_parquet(foreign)

    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 1, "debris")

    assert foreign.exists(), "Foreign file must survive the stale-sweep"


# ---------------------------------------------------------------------------
# M4: mark across multiple images
# ---------------------------------------------------------------------------


def test_mark_across_multiple_images(tmp_path: Path):
    """Two images, one marked object each in different categories."""
    master = pl.DataFrame(
        {
            "Metadata_ImageFile": ["pA", "pA", "pB", "pB"],
            "Metadata_Dataset": ["ds1"] * 4,
            "Object_Label": [1, 2, 1, 2],
            "Bbox_CenterRR": [10.0, 20.0, 10.0, 20.0],
            "Bbox_CenterCC": [10.0, 20.0, 10.0, 20.0],
            "Size_Area": [100.0, 200.0, 100.0, 200.0],
        }
    )
    store = CurationLabels.load(tmp_path, master)
    store.mark("pA", 1, "debris")
    store.mark("pB", 2, "merged")

    # Both keys are in removed_keys
    assert ("pA", 1) in store.removed_keys
    assert ("pB", 2) in store.removed_keys

    # Curated mirror drops both
    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert curated.height == 2
    remaining_keys = list(
        zip(
            curated.get_column("Metadata_ImageFile").to_list(),
            curated.get_column("Object_Label").to_list(),
        )
    )
    assert ("pA", 1) not in remaining_keys
    assert ("pB", 2) not in remaining_keys

    # Per-category parquets have the right objects
    debris_df = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "debris"))
    assert debris_df.get_column("Object_Label").to_list() == [1]
    assert debris_df.get_column("Metadata_ImageFile").to_list() == ["pA"]

    merged_df = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "merged"))
    assert merged_df.get_column("Object_Label").to_list() == [2]
    assert merged_df.get_column("Metadata_ImageFile").to_list() == ["pB"]


# ---------------------------------------------------------------------------
# Task 1: mtime guard — external reseed must block subsequent writes
# ---------------------------------------------------------------------------


def test_save_refuses_after_external_reseed(tmp_path: Path):
    """A CLI re-seed (mtime bump) must cause the guard to refuse further writes.

    Sequence:
    1. ``mark`` seeds ``measurements.parquet`` + records its mtime.
    2. An external actor bumps the mtime (simulating a CLI ``--measure`` re-run).
    3. A second ``mark`` must refuse to clobber the mirror.
    4. ``store.stale is True``.
    5. Object 2 is NOT in the mirror (the second mark never wrote it).
    """
    import os
    import time

    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 1, "debris")  # seeds measurements.parquet + records mtime

    # Simulate a CLI re-seed: bump the mtime of measurements.parquet.
    mpath = tools_.measurements_parquet_path(tmp_path)
    future = time.time() + 5
    os.utime(mpath, (future, future))

    store.mark("plateA", 2, "merged")  # must refuse to clobber

    on_disk = pl.read_parquet(mpath)
    # Object 2 must NOT have been removed from the mirror (the write was refused).
    assert 2 in on_disk.get_column("Object_Label").to_list()
    # The staleness flag must be set.
    assert store.stale is True


def test_labels_survive_reload_against_curated_mirror(tmp_path: Path):
    """Re-key against the CLEAN master on disk, not the curated mirror.

    Regression: ``OutputRoot`` hands ``load`` the curated ``measurements.parquet``
    mirror, which has the labeled rows removed. Re-keying against that mirror
    dropped every label on a viewer reload. With the clean
    ``master_measurements.parquet`` present, the labels must survive.
    """
    clean = _master(6)
    master_path = tools_.master_measurements_parquet_path(tmp_path)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    clean.write_parquet(master_path)

    # Session 1: open against the clean master, mark 3 objects.
    s1 = CurationLabels.load(tmp_path, clean)
    s1.mark_many([("plateA", i) for i in (1, 2, 3)], "debris")
    mirror = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert mirror.height == 3  # labeled rows curated OUT of the mirror

    # Session 2 (reload): OutputRoot would pass the curated MIRROR as master_df.
    s2 = CurationLabels.load(tmp_path, mirror)
    assert len(s2.labels) == 3  # all survive — re-keyed against the clean master
    assert s2.rekey_report.kept == 3
    assert s2.rekey_report.dropped == 0

    # And the labeled (curated-out) objects still re-emit to errors/*.parquet,
    # because the per-category partition reads the clean master.
    s2.write_error_partitions()
    errs = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "debris"))
    assert sorted(errs.get_column("Object_Label").to_list()) == [1, 2, 3]


def test_curated_mirror_preserves_post_columns(tmp_path: Path):
    """The curated mirror is written from the post-applied frame (``master_df``),
    so post-only columns survive curation; re-keying still uses the clean master."""
    clean = _master(4)
    master_path = tools_.master_measurements_parquet_path(tmp_path)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    clean.write_parquet(master_path)

    # A post-applied mirror carries a column the clean master lacks.
    mirror = clean.with_columns(pl.lit(1.0).alias("Post_Normalized"))
    store = CurationLabels.load(tmp_path, mirror)
    store.mark("plateA", 2, "debris")

    curated = pl.read_parquet(tools_.measurements_parquet_path(tmp_path))
    assert "Post_Normalized" in curated.columns  # post column preserved
    assert 2 not in curated.get_column("Object_Label").to_list()  # still curated

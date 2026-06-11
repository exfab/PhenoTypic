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

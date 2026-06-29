"""Tests for ``CurationLabels.write_error_partitions`` (Dash-free, no mirror).

The CLI finalize path re-emits the per-category error parquets + the re-keyed
labels parquet headlessly, **without** rewriting the curated
``measurements.parquet`` mirror (that stays the GUI's live concern). This method
is the lock-held composition of ``_write_category_parquets`` +
``_write_labels_parquet`` that omits the mirror.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import phenotypic.sdk_ as tools_
from phenotypic.sdk_ import BundleLayout
from phenotypic.gui.results_viewer._curation_labels import CurationLabels


def _layout(tmp_path: Path) -> BundleLayout:
    """Full-run-style layout rooted at ``tmp_path`` (deliverables under it)."""
    return BundleLayout(deliverables_base=tmp_path / "deliverables", output_root=tmp_path)


def _master(n: int = 6) -> pl.DataFrame:
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


def test_write_error_partitions_writes_errors_and_labels_no_mirror(
    tmp_path: Path,
) -> None:
    """It writes errors/<cat>.parquet (with Curation_Category) + labels, no mirror."""
    store = CurationLabels.load(_layout(tmp_path), _master())
    # Mark across two categories WITHOUT going through mark()/_save_locked, so the
    # mirror is never seeded by the setup — we then assert the new method also
    # does not create it.
    store.labels[("plateA", 1)] = "debris"
    store.labels[("plateA", 2)] = "debris"
    store.labels[("plateA", 3)] = "background_noise"
    store.fingerprints[("plateA", 1)] = (10.0, 20.0)
    store.fingerprints[("plateA", 2)] = (20.0, 40.0)
    store.fingerprints[("plateA", 3)] = (30.0, 60.0)

    mirror = tools_.measurements_parquet_path(tmp_path)
    assert not mirror.exists()

    store.write_error_partitions()

    # Mirror NOT (re)written by this call.
    assert not mirror.exists()

    # Per-category parquets exist and carry the category column.
    debris = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "debris"))
    assert sorted(debris.get_column("Object_Label").to_list()) == [1, 2]
    assert debris.get_column("Curation_Category").to_list() == ["debris", "debris"]

    noise = pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "background_noise")
    )
    assert noise.get_column("Object_Label").to_list() == [3]
    assert noise.get_column("Curation_Category").to_list() == ["background_noise"]

    # Labels parquet exists and round-trips the full label set on reload.
    assert tools_.curation_labels_parquet_path(tmp_path).exists()
    reloaded = CurationLabels.load(_layout(tmp_path), _master())
    assert reloaded.labels == {
        ("plateA", 1): "debris",
        ("plateA", 2): "debris",
        ("plateA", 3): "background_noise",
    }


def test_write_error_partitions_leaves_existing_mirror_untouched(
    tmp_path: Path,
) -> None:
    """When a mirror already exists, the call does not rewrite it (mtime stable)."""
    store = CurationLabels.load(_layout(tmp_path), _master())
    # Seed the mirror once via the normal save path.
    store.mark("plateA", 1, "debris")
    mirror = tools_.measurements_parquet_path(tmp_path)
    assert mirror.exists()
    before_ns = mirror.stat().st_mtime_ns

    # Add another label directly (no save) then re-emit partitions only.
    store.labels[("plateA", 2)] = "debris"
    store.fingerprints[("plateA", 2)] = (20.0, 40.0)
    store.write_error_partitions()

    # Mirror untouched by write_error_partitions.
    assert mirror.stat().st_mtime_ns == before_ns
    # But the error partition reflects BOTH labeled objects.
    debris = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "debris"))
    assert sorted(debris.get_column("Object_Label").to_list()) == [1, 2]

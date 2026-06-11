"""Tests for the durable CurationLabels store."""

from pathlib import Path

import polars as pl
import pytest

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

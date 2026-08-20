# tests/unit/sdk_/test_bundle_layout.py
from pathlib import Path

import polars as pl
import pytest

from phenotypic.sdk_ import BundleLayout


def _seed_deliverables(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Metadata_Dataset": ["plate1"], "Object_Label": [1]}).write_parquet(
        base / "master_measurements.parquet"
    )


def test_detect_when_pointed_at_parent_containing_deliverables(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    layout = BundleLayout.detect(out)
    assert layout.deliverables_base == (out / "deliverables").resolve()
    assert layout.output_root == out.resolve()


def test_detect_when_pointed_at_deliverables_dir_standalone(tmp_path):
    base = tmp_path / "bundle" / "deliverables"
    _seed_deliverables(base)
    layout = BundleLayout.detect(base)
    assert layout.deliverables_base == base.resolve()
    # No sibling results/ -> standalone, no output_root.
    assert layout.output_root is None
    assert layout.has_results is False
    assert layout.results_dir is None


def test_detect_deliverables_subdir_with_sibling_results_promotes_parent(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    (out / "results" / "plate1" / "hdf").mkdir(parents=True)
    layout = BundleLayout.detect(out / "deliverables")
    assert layout.output_root == out.resolve()
    assert layout.has_results is True


def test_promotion_guard_requires_deliverables_name(tmp_path):
    # A standalone bundle NOT named "deliverables" must not adopt a sibling results/.
    base = tmp_path / "shared_bundle"
    _seed_deliverables(base)
    (tmp_path / "results" / "plate1").mkdir(parents=True)
    layout = BundleLayout.detect(base)
    assert layout.output_root is None


def test_detect_rejects_non_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        BundleLayout.detect(tmp_path)


def test_resolved_pipeline_config_path_prefers_canonical(tmp_path):
    """Canonical typed config present -> returned over a legacy sibling."""
    out = tmp_path / "run"
    base = out / "deliverables"
    _seed_deliverables(base)
    canonical = base / "pipeline.json.pht-pipe"
    legacy = base / "pipeline.json"
    canonical.write_text("{}", encoding="utf-8")
    legacy.write_text("{}", encoding="utf-8")
    layout = BundleLayout.detect(out)
    assert layout.resolved_pipeline_config_path == canonical


def test_resolved_pipeline_config_path_falls_back_to_legacy(tmp_path):
    """Only the legacy plain ``pipeline.json`` exists -> it is returned."""
    out = tmp_path / "run"
    base = out / "deliverables"
    _seed_deliverables(base)
    legacy = base / "pipeline.json"
    legacy.write_text("{}", encoding="utf-8")
    layout = BundleLayout.detect(out)
    assert layout.resolved_pipeline_config_path == legacy


def test_resolved_pipeline_config_path_defaults_to_canonical_when_neither(tmp_path):
    """Neither config present -> the canonical typed path (for fresh writes)."""
    out = tmp_path / "run"
    base = out / "deliverables"
    _seed_deliverables(base)
    layout = BundleLayout.detect(out)
    assert layout.resolved_pipeline_config_path == layout.pipeline_config_path
    assert not layout.resolved_pipeline_config_path.exists()


def test_deliverables_accessors_anchor_on_base(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    layout = BundleLayout.detect(out)
    base = out / "deliverables"
    assert layout.master_parquet == base / "master_measurements.parquet"
    assert layout.mirror_parquet == base / "measurements.parquet"
    assert layout.qc_duckdb == base / "qc" / "qc.duckdb"
    assert layout.curation_labels_parquet == base / "qc" / "curation_labels.parquet"
    assert layout.overlay_path("plate1", "img001") == base / "overlays" / "plate1" / "img001.png"


def test_qc_duckdb_path_under_deliverables_qc(tmp_path):
    from phenotypic.sdk_ import qc_duckdb_path, qc_dir

    assert qc_duckdb_path(tmp_path) == qc_dir(tmp_path) / "qc.duckdb"


def test_bundle_layout_qc_duckdb_accessor(tmp_path):
    from phenotypic.sdk_ import BundleLayout

    # Seed a deliverables bundle so BundleLayout.detect classifies the dir
    # (plan test omitted this; detect raises on an empty dir).
    _seed_deliverables(tmp_path / "deliverables")
    layout = BundleLayout.detect(tmp_path)
    assert layout.qc_duckdb == layout.qc_dir / "qc.duckdb"


def test_store_path_resolves_a_directory_not_a_file(tmp_path):
    """A store is a directory; an is_file() check would always return None."""
    from phenotypic.sdk_ import BundleLayout, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "img")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    _seed_deliverables(tmp_path / "deliverables")
    layout = BundleLayout.detect(tmp_path)
    assert layout.store_path("ds", "img") == store


def test_store_path_returns_none_when_absent(tmp_path):
    from phenotypic.sdk_ import BundleLayout

    _seed_deliverables(tmp_path / "deliverables")
    layout = BundleLayout.detect(tmp_path)
    assert layout.store_path("ds", "img") is None


def test_store_path_returns_none_for_a_file_at_the_store_path(tmp_path):
    """A stray file named like a store is not a store."""
    from phenotypic.sdk_ import BundleLayout, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "img")
    store.parent.mkdir(parents=True)
    store.write_bytes(b"")
    _seed_deliverables(tmp_path / "deliverables")
    layout = BundleLayout.detect(tmp_path)
    assert layout.store_path("ds", "img") is None


def test_store_path_is_none_for_a_standalone_bundle_with_no_output_root(tmp_path):
    from phenotypic.sdk_ import BundleLayout

    base = tmp_path / "bundle" / "deliverables"
    _seed_deliverables(base)
    layout = BundleLayout.detect(base)
    assert layout.output_root is None
    assert layout.store_path("ds", "img") is None

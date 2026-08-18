"""Unit tests for ``OutputRoot.discover`` standalone-bundle support.

Companion to ``tests/gui/results_viewer/test_output_root.py`` (the full-run
discovery suite). These exercise the Task 4 ``BundleLayout``-backed discovery:
booting from a deliverables-only bundle (no ``results/``) and lighting up the
full-run capabilities when ``results/`` is present.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import EXPERIMENT, IMAGE


def _seed_standalone_bundle(base: Path) -> None:
    """Deliverables-only: master + mirror + one overlay, NO results/, NO root qc/."""
    base.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate1", "plate1"],
            str(IMAGE.IMAGE_NAME): ["img001", "img001"],
            "Object_Label": [1, 2],
        }
    )
    df.write_parquet(base / "master_measurements.parquet")
    df.write_parquet(base / "measurements.parquet")
    ov = base / "overlays" / "plate1"
    ov.mkdir(parents=True)
    from PIL import Image as PILImage

    PILImage.new("RGB", (8, 8)).save(ov / "img001.png")


def test_discover_standalone_deliverables_only(tmp_path: Path) -> None:
    base = tmp_path / "bundle" / "deliverables"
    _seed_standalone_bundle(base)
    root = OutputRoot.discover(
        base,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )
    assert root.has_results is False
    assert "plate1" in root.master_df[str(EXPERIMENT.DATASET)].unique().to_list()
    # Overlay-backed picker still works.
    assert root.has_overlay("plate1", "img001") is True
    assert root.hdf_path("plate1", "img001") is None


def test_discover_full_run_lights_up_results(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _seed_standalone_bundle(out / "deliverables")
    (out / "results" / "plate1" / "hdf").mkdir(parents=True)
    (out / "results" / "plate1" / "hdf" / "img001.h5").write_bytes(b"")
    root = OutputRoot.discover(
        out,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )
    assert root.has_results is True
    assert root.hdf_path("plate1", "img001") is not None

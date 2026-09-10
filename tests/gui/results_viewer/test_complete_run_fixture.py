"""The e2e fixture helper publishes a run the Results viewer calls complete."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic._gui.results_viewer._mutation_guard import output_mutations_disabled
from phenotypic._gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import (
    dataset_overlays_dir,
    deliverables_dir,
    measurements_csv_path,
    zarr_store_path,
)
from tests._output_layout import (
    publish_complete_run_over_outputs,
    write_master,
    write_measurements_mirror,
)


def _seed(root: Path, images: list[str], dataset: str = "ds1") -> None:
    """Write a master and mirror listing ``images`` under one dataset."""
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": [dataset] * len(images),
            "Metadata_ImageName": images,
            "Object_Label": list(range(1, len(images) + 1)),
            "Shape_Area": [100.0 + index for index in range(len(images))],
        }
    )
    write_master(root, frame)
    write_measurements_mirror(root, frame)


def _digests(root: Path) -> dict[str, str]:
    """Return a content digest for every file below ``root``."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_published_outputs_are_a_complete_mutable_run(tmp_path: Path) -> None:
    """Completion evidence is written without touching the fixture's deliverables."""
    root = tmp_path / "run"
    _seed(root, ["plate_001.tif", "plate_002"])
    before = _digests(deliverables_dir(root))

    publish_complete_run_over_outputs(root, total_images=2)

    output = OutputRoot.discover(root, cache_root=tmp_path / "cache")
    assert output.run_completion == "complete", output.run_advisories
    assert not output_mutations_disabled(output)
    assert _digests(deliverables_dir(root)) == before


def test_a_store_the_fixture_wrote_is_reused_not_replaced(tmp_path: Path) -> None:
    """A real store survives publication byte for byte."""
    root = tmp_path / "run"
    _seed(root, ["plate_001"])
    store = zarr_store_path(root, "ds1", "plate_001")
    store.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image(pixels).save2zarr(store)
    before = _digests(store)

    publish_complete_run_over_outputs(root, total_images=1)

    assert _digests(store) == before
    output = OutputRoot.discover(root, cache_root=tmp_path / "cache")
    assert output.run_completion == "complete", output.run_advisories


def test_an_overlay_backed_fixture_keeps_its_overlay_as_the_pixel_source(tmp_path: Path) -> None:
    """No store is promoted over an overlay, so the viewer still crops from it."""
    root = tmp_path / "run"
    _seed(root, ["plate_001.tif"])
    overlay = dataset_overlays_dir(root, "ds1") / "plate_001.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (32, 32), (200, 0, 0)).save(overlay)
    before = _digests(deliverables_dir(root))

    publish_complete_run_over_outputs(root, total_images=1)

    output = OutputRoot.discover(root, cache_root=tmp_path / "cache")
    assert output.run_completion == "complete", output.run_advisories
    assert output.store_path("ds1", "plate_001") is None
    assert output.has_overlay("ds1", "plate_001")
    assert _digests(deliverables_dir(root)) == before


def test_a_missing_core_file_fails_loudly(tmp_path: Path) -> None:
    """A fixture without the CSV mirror is told what to write."""
    root = tmp_path / "run"
    _seed(root, ["plate_001"])
    measurements_csv_path(root).unlink()

    with pytest.raises(FileNotFoundError, match="measurements CSV"):
        publish_complete_run_over_outputs(root, total_images=1)


def test_a_miscounted_fixture_fails_loudly(tmp_path: Path) -> None:
    """The declared image count must match the master."""
    root = tmp_path / "run"
    _seed(root, ["plate_001", "plate_002"])

    with pytest.raises(AssertionError, match="master lists 2 images"):
        publish_complete_run_over_outputs(root, total_images=3)

"""Smoke tests for the colony-view ``/crops`` Flask blueprint.

Spins up a Dash app against a tmp output directory containing a tiny
synthetic master parquet plus a single overlay PNG, then hits the
``/crops`` endpoint via Flask's test client. Verifies happy path,
validation errors, and 404s without launching a real server.
"""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui._config import QC_CROPS_URL_SEGMENT
from phenotypic.gui._shared import tiles
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot

from tests._output_layout import write_master
from phenotypic.schema import IMAGE


@pytest.fixture()
def crop_app(tmp_path: Path):
    """Build a minimal output dir, Dash client, and source-path handles.

    Lays out:
      <tmp>/deliverables/master_measurements.parquet  (1 colony in dataset 'd1')
      <tmp>/deliverables/overlays/d1/img-1.png         (uniform red 100×100 PNG)
    """
    # 1. master_measurements.parquet (under deliverables/)
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            str(IMAGE.IMAGE_NAME): ["img-1"],
            "Object_Label": [7],
            "Bbox_CenterRR": [50],
            "Bbox_CenterCC": [50],
            "Bbox_MinRR": [40],
            "Bbox_MaxRR": [60],
            "Bbox_MinCC": [40],
            "Bbox_MaxCC": [60],
        }
    )
    write_master(tmp_path, master)

    # 2. overlay PNG
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (100, 100), (255, 0, 0)).save(
        overlay_dir / "img-1.png", format="PNG"
    )
    hdf_path = tmp_path / "results" / "d1" / "hdf" / "img-1.h5"
    hdf_path.parent.mkdir(parents=True)
    with h5py.File(hdf_path, "w") as handle:
        layers = handle.create_group("layers")
        layers.create_dataset(
            "rgb",
            data=np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8),
        )

    # 3. Build the app and hand back the Flask test client.
    output_root = OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )
    app = create_app(output_root)
    return (
        app.server.test_client(),
        output_root,
        overlay_dir / "img-1.png",
        hdf_path,
    )


@pytest.fixture()
def app_client(crop_app):
    """Return only the Flask client for ordinary route contract tests."""
    return crop_app[0]


def test_crop_route_happy_path_returns_png(app_client) -> None:
    """A valid /crops request returns a PNG of the requested size."""
    resp = app_client.get("/crops/d1/img-1/7.png?size=20")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    img = PILImage.open(io.BytesIO(resp.data))
    assert img.size == (20, 20)


def test_crop_route_rejects_missing_size(app_client) -> None:
    """No ``size=`` query param → 400."""
    resp = app_client.get("/crops/d1/img-1/7.png")
    assert resp.status_code == 400


def test_crop_route_rejects_oversized_size(app_client) -> None:
    """size > 4096 → 400 to head off DoS via huge allocations."""
    resp = app_client.get("/crops/d1/img-1/7.png?size=9999")
    assert resp.status_code == 400


def test_crop_route_returns_404_for_unknown_label(app_client) -> None:
    """A label the master frame doesn't know about → 404."""
    resp = app_client.get("/crops/d1/img-1/99.png?size=20")
    assert resp.status_code == 404


def test_crop_route_rejects_unknown_dataset(app_client) -> None:
    """A dataset that doesn't exist on disk → 404 (overlay not found)."""
    resp = app_client.get("/crops/no-such-dataset/img-1/7.png?size=20")
    assert resp.status_code == 404


def test_overlay_replacement_stales_colony_and_qc_crop_routes(crop_app) -> None:
    """Both shared crop namespaces reject a replaced overlay generation."""
    client, output_root, overlay_path, _hdf_path = crop_app
    bound_token = output_root.bound_image_source_token("d1", "img-1")
    replacement = overlay_path.with_name("replacement.png")
    PILImage.new("RGB", (100, 100), (0, 255, 0)).save(
        replacement,
        format="PNG",
    )
    replacement.replace(overlay_path)
    assert not output_root.image_source_token_is_current(
        "d1",
        "img-1",
        bound_token,
    )

    for segment in ("crops", QC_CROPS_URL_SEGMENT):
        response = client.get(f"/{segment}/d1/img-1/7.png?size=20")
        assert response.status_code == 409


def test_hdf_replacement_stales_colony_and_qc_crop_routes(crop_app) -> None:
    """Both shared crop namespaces reject a replaced HDF generation."""
    client, output_root, _overlay_path, hdf_path = crop_app
    bound_token = output_root.bound_image_source_token("d1", "img-1")
    replacement = hdf_path.with_name("replacement.h5")
    with h5py.File(replacement, "w") as handle:
        layers = handle.create_group("layers")
        layers.create_dataset(
            "rgb",
            data=np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8),
        )
    replacement.replace(hdf_path)
    assert not output_root.image_source_token_is_current(
        "d1",
        "img-1",
        bound_token,
    )

    for segment in ("crops", QC_CROPS_URL_SEGMENT):
        response = client.get(f"/{segment}/d1/img-1/7.png?size=20")
        assert response.status_code == 409


@pytest.mark.parametrize("segment", ["crops", QC_CROPS_URL_SEGMENT])
def test_crop_route_rechecks_snapshot_after_pixel_read(
    crop_app,
    monkeypatch: pytest.MonkeyPatch,
    segment: str,
) -> None:
    """A processing replacement during the read is rejected after cropping."""
    client, output_root, overlay_path, _hdf_path = crop_app
    bound_token = output_root.bound_image_source_token("d1", "img-1")
    real_crop = tiles.crop_colony

    def _crop_then_replace(*args, **kwargs):
        png = real_crop(*args, **kwargs)
        replacement = overlay_path.with_name("mid-read.png")
        PILImage.new("RGB", (100, 100), (0, 255, 0)).save(
            replacement,
            format="PNG",
        )
        replacement.replace(overlay_path)
        return png

    monkeypatch.setattr(tiles, "crop_colony", _crop_then_replace)

    response = client.get(f"/{segment}/d1/img-1/7.png?size=20")

    assert response.status_code == 409
    assert not output_root.image_source_token_is_current(
        "d1",
        "img-1",
        bound_token,
    )

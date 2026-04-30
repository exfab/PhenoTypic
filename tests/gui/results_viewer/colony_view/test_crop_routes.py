"""Smoke tests for the colony-view ``/crops`` Flask blueprint.

Spins up a Dash app against a tmp output directory containing a tiny
synthetic master parquet plus a single overlay PNG, then hits the
``/crops`` endpoint via Flask's test client. Verifies happy path,
validation errors, and 404s without launching a real server.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot


@pytest.fixture()
def app_client(tmp_path: Path):
    """Build a minimal output dir + Dash app and return the Flask test client.

    Lays out:
      <tmp>/master_measurements.parquet           (1 colony in dataset 'd1')
      <tmp>/results/d1/overlays/img-1.png         (uniform red 100×100 PNG)
    """
    # 1. master_measurements.parquet
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            "Metadata_ImageFile": ["img-1"],
            "ObjectLabel": [7],
            "Bbox_CenterRR": [50],
            "Bbox_CenterCC": [50],
            "Bbox_MinRR": [40],
            "Bbox_MaxRR": [60],
            "Bbox_MinCC": [40],
            "Bbox_MaxCC": [60],
        }
    )
    master.write_parquet(tmp_path / "master_measurements.parquet")

    # 2. overlay PNG
    overlay_dir = tmp_path / "results" / "d1" / "overlays"
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (100, 100), (255, 0, 0)).save(
        overlay_dir / "img-1.png", format="PNG"
    )

    # 3. Build the app and hand back the Flask test client.
    output_root = OutputRoot.discover(tmp_path)
    app = create_app(output_root)
    return app.server.test_client()


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

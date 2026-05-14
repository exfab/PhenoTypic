"""Verify the builder tiles blueprint serves DZI manifests for staged PNGs.

The blueprint mounts at ``/tiles`` on the builder Flask app — the path
the server sees AFTER the hub :class:`DispatcherMiddleware` strips the
``/builder/`` mount prefix. This test drives Flask's
:class:`flask.testing.FlaskClient` directly (no dispatcher in the
loop), so the test URLs use the bare blueprint path. Browser-facing
URLs are ``<requests_pathname_prefix>tiles/...`` — see
:func:`phenotypic.gui.builder._point_picker._dzi_url`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.builder._app import create_app


@pytest.fixture
def app_with_tmp_root(tmp_path):
    """Builder app whose ``image_root`` is *tmp_path* so the cache lives there."""
    return create_app(image_root=tmp_path)


def _stage_png(
    tmp_path: Path,
    session_id: str,
    source: str,
    w: int = 64,
    h: int = 64,
) -> Path:
    """Place a synthetic PNG at the cache location the blueprint expects."""
    cache_dir = tmp_path / ".phenotypic-gui" / "builder_tiles" / session_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    png_path = cache_dir / f"{source}.png"
    rng = np.random.default_rng(seed=0)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    PILImage.fromarray(arr).save(png_path)
    return png_path


def test_dzi_manifest_served(app_with_tmp_root, tmp_path):
    """A staged PNG is tiled lazily and the manifest XML round-trips."""
    sid = "deadbeef-1234-5678-9abc-deadbeef0000"
    _stage_png(tmp_path, sid, "rgb")
    client = app_with_tmp_root.server.test_client()
    resp = client.get(f"/tiles/{sid}/rgb.dzi")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<Image" in body
    # The DZI manifest declares the deepzoom XML namespace; capitalisation
    # varies between backends but ``deepzoom`` always appears somewhere.
    assert "deepzoom" in body.lower()


def test_unknown_source_rejected(app_with_tmp_root):
    """``<source>`` outside the rgb/intermediate allow-list returns 404."""
    sid = "deadbeef-1234-5678-9abc-deadbeef0000"
    client = app_with_tmp_root.server.test_client()
    resp = client.get(f"/tiles/{sid}/badsource.dzi")
    assert resp.status_code == 404


def test_unsafe_session_id_rejected(app_with_tmp_root):
    """Path-traversal payloads in ``<session_id>`` get rejected without disk reads.

    Werkzeug normalises encoded slashes ahead of routing, so the only payloads
    that actually reach our handler are those that survive as one path
    component — ``..`` (parent), dot-prefixed, special chars, etc. We exercise
    a representative sample.
    """
    client = app_with_tmp_root.server.test_client()
    # ``..`` is a literal parent-dir reference per :func:`_is_safe_path_component`.
    resp_dotdot = client.get("/tiles/../rgb.dzi")
    assert resp_dotdot.status_code == 404
    # Leading-dot id (``.hidden``) — also rejected.
    resp_hidden = client.get("/tiles/.hidden/rgb.dzi")
    assert resp_hidden.status_code == 404
    # Special chars not in the safe charset.
    resp_special = client.get("/tiles/has$dollar/rgb.dzi")
    assert resp_special.status_code == 404


def test_too_short_session_id_rejected(app_with_tmp_root, tmp_path):
    """Session ids shorter than 8 chars are rejected even if charset-safe."""
    # Stage a PNG under the short id directly so we can prove the rejection
    # is policy-based, not "PNG missing".
    _stage_png(tmp_path, "short", "rgb")
    client = app_with_tmp_root.server.test_client()
    resp = client.get("/tiles/short/rgb.dzi")
    assert resp.status_code == 404


def test_missing_png_returns_404(app_with_tmp_root):
    """Missing source PNG returns 404 rather than a partially-tiled directory."""
    client = app_with_tmp_root.server.test_client()
    resp = client.get(
        "/tiles/deadbeef-1234-5678-9abc-deadbeef0000/rgb.dzi"
    )
    assert resp.status_code == 404


def test_tile_endpoint_serves_after_manifest(app_with_tmp_root, tmp_path):
    """After the manifest is fetched, a tile from the pyramid is reachable."""
    sid = "deadbeef-1234-5678-9abc-deadbeef0000"
    _stage_png(tmp_path, sid, "rgb")
    client = app_with_tmp_root.server.test_client()
    # Generate the pyramid via the manifest endpoint.
    manifest_resp = client.get(f"/tiles/{sid}/rgb.dzi")
    assert manifest_resp.status_code == 200

    # Look up which level files actually exist (the tiler picks levels
    # based on image dimensions; level 0 is the smallest pyramid layer).
    cache_dir = tmp_path / ".phenotypic-gui" / "builder_tiles" / sid
    files_dir = cache_dir / "rgb_files"
    assert files_dir.is_dir(), "tile pyramid should have been generated"
    level_dirs = sorted(p for p in files_dir.iterdir() if p.is_dir())
    assert level_dirs, "no pyramid levels generated"
    sample_level = level_dirs[0]
    sample_tile = next(sample_level.glob("*.png"))
    rel_level = sample_level.name
    rel_filename = sample_tile.name

    tile_resp = client.get(
        f"/tiles/{sid}/rgb_files/{rel_level}/{rel_filename}"
    )
    assert tile_resp.status_code == 200
    assert tile_resp.mimetype == "image/png"


def test_tile_endpoint_rejects_unsafe_filename(app_with_tmp_root, tmp_path):
    """Tile filenames must match the strict ``<col>_<row>.png`` pattern."""
    sid = "deadbeef-1234-5678-9abc-deadbeef0000"
    _stage_png(tmp_path, sid, "rgb")
    client = app_with_tmp_root.server.test_client()
    # Make sure tiles exist so a real route resolution would otherwise succeed.
    client.get(f"/tiles/{sid}/rgb.dzi")

    bad_resp = client.get(
        f"/tiles/{sid}/rgb_files/0/not-a-tile.png"
    )
    assert bad_resp.status_code == 404

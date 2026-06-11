import dash
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr
from phenotypic.gui.browse import _tile_routes
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture
def app_and_root(monkeypatch, tmp_path):
    # Redirect the ephemeral cache into the test's tmp dir.
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(cache))
    # Sandbox root with one image.
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "plates" / "b7").mkdir(parents=True)
    img = sandbox_root / "plates" / "b7" / "A1.png"
    PILImage.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(img)
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    # Dash 4.x runs validate_layout in a before_request hook; a None layout
    # would 500 every request before it reaches the blueprint. Give it a
    # trivial layout so the tile routes are actually exercised.
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    return app.server.test_client(), "plates/b7/A1.png"


def test_manifest_then_tile(app_and_root):
    client, rel = app_and_root
    token = sr.encode_token(rel)
    manifest = client.get(f"/tiles/{token}.dzi")
    assert manifest.status_code == 200
    assert b"<Image" in manifest.data
    tile = client.get(f"/tiles/{token}_files/0/0_0.png")
    assert tile.status_code == 200
    assert tile.mimetype == "image/png"


def test_malformed_token_404(app_and_root):
    client, _ = app_and_root
    assert client.get("/tiles/not%20a%20token.dzi").status_code == 404


def test_escape_token_404(app_and_root):
    client, _ = app_and_root
    token = sr.encode_token("../../etc/passwd")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_raw_unavailable_422(app_and_root, monkeypatch, tmp_path):
    client, _ = app_and_root

    def _boom(original, cache_png):
        raise sr.SourceRenderUnavailable("nope")

    monkeypatch.setattr(_tile_routes._source_render, "normalize_to_png", _boom)
    # A token that resolves to the existing image so render is attempted.
    token = sr.encode_token("plates/b7/A1.png")
    assert client.get(f"/tiles/{token}.dzi").status_code == 422

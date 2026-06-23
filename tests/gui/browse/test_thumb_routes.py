"""Browse thumbnail route smoke tests (Flask test client)."""
from __future__ import annotations

import io
from pathlib import Path

import dash
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render, _thumb_routes
from phenotypic.gui.shell._sandbox import SandboxRoot


def _client(monkeypatch, tmp_path: Path):
    # Redirect the ephemeral cache into tmp_path so init_cache() never wipes
    # the real system temp dir (the established browse-test idiom).
    monkeypatch.setattr(
        _source_render.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    # A source image inside the sandbox root.
    (tmp_path / "imgs").mkdir()
    src = tmp_path / "imgs" / "plateA.png"
    PILImage.new("RGB", (200, 100), (0, 128, 0)).save(src, format="PNG")

    sandbox = SandboxRoot.from_path(tmp_path)  # canonical constructor (resolves)
    _source_render.init_cache()
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # REQUIRED: a layout-less Dash app 500s on first
    _thumb_routes.register(app, sandbox)
    token = _source_render.encode_token("imgs/plateA.png")
    return app.server.test_client(), token


def test_thumb_happy_path(monkeypatch, tmp_path: Path) -> None:
    client, token = _client(monkeypatch, tmp_path)
    resp = client.get(f"/thumb/{token}?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128


def test_thumb_unknown_token_is_404(monkeypatch, tmp_path: Path) -> None:
    client, _token = _client(monkeypatch, tmp_path)
    resp = client.get("/thumb/not-a-real-token?size=128")
    assert resp.status_code == 404

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr
from phenotypic.gui.browse._app import create_app
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "imgs"
    root.mkdir()
    PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "a.png")
    return SandboxRoot.from_path(root)


def test_create_app_serves_layout_and_tiles(sandbox):
    app = create_app(sandbox, url_prefix="/")
    client = app.server.test_client()
    # Dash layout endpoint responds.
    assert client.get("/_dash-layout").status_code == 200
    # Tile blueprint is mounted.
    token = sr.encode_token("a.png")
    assert client.get(f"/tiles/{token}.dzi").status_code == 200


def test_create_app_injects_app_prefix(sandbox):
    app = create_app(sandbox, url_prefix="/browse/")
    assert "window.__phenotypicAppPrefix" in app.index_string
    assert "/browse/" in app.index_string


def test_create_app_serves_thumbnail_route(monkeypatch, tmp_path) -> None:
    import io
    from PIL import Image as PILImage
    from phenotypic.gui.browse._app import create_app
    from phenotypic.gui.browse import _source_render
    from phenotypic.gui.shell._sandbox import SandboxRoot

    monkeypatch.setattr(
        _source_render.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    (tmp_path / "imgs").mkdir()
    PILImage.new("RGB", (120, 60), (1, 2, 3)).save(tmp_path / "imgs" / "p.png")
    app = create_app(SandboxRoot.from_path(tmp_path))
    client = app.server.test_client()
    token = _source_render.encode_token("imgs/p.png")
    resp = client.get(f"/thumb/{token}?size=64")
    assert resp.status_code == 200
    assert PILImage.open(io.BytesIO(resp.data)).size[0] <= 64

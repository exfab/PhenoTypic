import tempfile

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic._gui.browse import _source_render as sr
from phenotypic._gui.browse._app import create_app
from phenotypic._gui._shared import viv_script_urls
from phenotypic._gui.shell._sandbox import SandboxRoot


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    # Patch ``tempfile`` directly, not through ``_source_render``.
    #
    # This used to read ``sr.tempfile``, which worked only because that module
    # imported ``tempfile`` for its own ephemeral cache. P6 Task 7 deleted that
    # cache and the import with it, and every fixture reaching the module's
    # attribute broke -- a consumer with no import edge, which no import-graph
    # walk can see.
    #
    # The redirect itself is NOT vestigial. ``_cache.resolve_cache_location``
    # falls back to ``tempfile.mkdtemp(prefix="phenotypic-browse-")`` with no
    # ``dir=`` when neither the sandbox nor the user-cache tier is writable,
    # and that consults ``gettempdir()``. Patching it keeps that last resort
    # inside ``tmp_path`` instead of the real system temp dir.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "cache"))
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


def test_create_app_serves_the_shared_viv_facade_after_its_bundle(sandbox):
    app = create_app(sandbox, url_prefix="/browse/")

    assert app.config.external_scripts == viv_script_urls("/browse/")
    assert app.config.external_scripts == [
        "/browse/_viv/viv-bundle.min.js",
        "/browse/_viv/viv_viewer.js",
    ]
    client = app.server.test_client()
    assert client.get("/_viv/viv-bundle.min.js").status_code == 200
    assert client.get("/_viv/viv_viewer.js").status_code == 200

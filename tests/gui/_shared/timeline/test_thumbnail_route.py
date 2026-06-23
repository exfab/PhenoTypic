"""Flask-test-client smoke tests for register_thumbnail_route."""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import dash
import pytest
from PIL import Image as PILImage

from phenotypic.gui._shared.timeline import _thumbnail
from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    register_thumbnail_route,
)


@pytest.fixture()
def client(tmp_path: Path):
    src = tmp_path / "src.png"
    PILImage.new("RGB", (200, 100), (0, 0, 255)).save(src, format="PNG")

    def resolve_source(identity: str) -> Path:
        if identity == "raw":
            raise ThumbUnavailable("no rawpy")
        if identity == "missing":
            raise FileNotFoundError(identity)
        return src

    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # REQUIRED: a layout-less Dash app 500s on first
    # request (before_request → validate_layout → NoLayoutException). Matches the
    # established idiom in tests/gui/browse/test_tile_routes.py:30.
    register_thumbnail_route(
        app, segment="thumb", resolve_source=resolve_source, cache_base=tmp_path / "cache"
    )
    return app.server.test_client()


def test_happy_path_returns_bucketed_png(client) -> None:
    resp = client.get("/thumb/img-1?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128  # longest edge == snapped bucket


def test_missing_size_is_400(client) -> None:
    assert client.get("/thumb/img-1").status_code == 400


def test_thumb_unavailable_is_422(client) -> None:
    assert client.get("/thumb/raw?size=128").status_code == 422


def test_missing_source_is_404(client) -> None:
    assert client.get("/thumb/missing?size=128").status_code == 404


def test_second_request_is_served_from_cache(client, tmp_path: Path) -> None:
    assert client.get("/thumb/img-1?size=128").status_code == 200
    cache_files = list((tmp_path / "cache").glob("*.png"))
    assert len(cache_files) == 1
    # A second identical request reuses the cached file (no new file written).
    assert client.get("/thumb/img-1?size=128").status_code == 200
    assert list((tmp_path / "cache").glob("*.png")) == cache_files


def test_concurrent_same_key_renders_once(tmp_path: Path, monkeypatch) -> None:
    # Background-warm fires many concurrent fetches; two requests for the SAME
    # thumbnail must not both decode+downscale. The per-source render lock
    # serialises them so exactly one render happens and one cache file lands.
    src = tmp_path / "src.png"
    PILImage.new("RGB", (200, 100), (0, 0, 255)).save(src, format="PNG")

    calls: list[int] = []

    def _slow_render(src_png: Path, size: int) -> bytes:
        calls.append(1)
        time.sleep(0.2)  # widen the window so both threads race the cache check
        buf = io.BytesIO()
        PILImage.new("RGB", (size, size // 2), (0, 0, 255)).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(_thumbnail, "downscale_to_thumb", _slow_render)

    def resolve_source(identity: str) -> Path:
        return src

    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    register_thumbnail_route(
        app, segment="thumb", resolve_source=resolve_source, cache_base=tmp_path / "cache"
    )
    server = app.server

    statuses: list[int] = []

    def _fire() -> None:
        with server.test_client() as c:
            statuses.append(c.get("/thumb/img-1?size=128").status_code)

    threads = [threading.Thread(target=_fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses == [200, 200]
    assert len(calls) == 1  # exactly one render under concurrency
    assert len(list((tmp_path / "cache").glob("*.png"))) == 1

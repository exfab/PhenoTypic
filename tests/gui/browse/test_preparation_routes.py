import dash
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render
from phenotypic.gui.browse._cache import BrowseCache, CacheLocation
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._preparation_routes import (
    BrowsePreparationApi,
    register,
)
from phenotypic.gui.browse._source_probe import probe_source
from phenotypic.gui.shell._sandbox import SandboxRoot


def _client(tmp_path):
    sandbox_path = tmp_path / "sandbox"
    sandbox_path.mkdir()
    source = sandbox_path / "plate.png"
    PILImage.new("RGB", (8, 8), "red").save(source)
    sandbox = SandboxRoot.from_path(sandbox_path)
    cache = BrowseCache(CacheLocation(tmp_path / "cache", "sandbox", True))
    manager = BrowsePreparationManager(cache)
    api = BrowsePreparationApi(sandbox, cache, manager)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    register(app, api)
    revision = probe_source(
        source,
        sandbox_root=sandbox_path,
        relative_path="plate.png",
    )
    item = {
        "token": _source_render.encode_token("plate.png"),
        "revision": revision.cache_key,
    }
    return app.server.test_client(), api, manager, item


def test_dataset_start_status_stop_are_tab_scoped(tmp_path):
    client, _api, manager, item = _client(tmp_path)
    try:
        started = client.post(
            "/api/browse/dataset/start",
            json={"client_id": "tab-a", "generation": 1, "items": [item]},
        )
        assert started.status_code == 200
        assert started.get_json()["total"] == 1

        other = client.get("/api/browse/dataset/status?client_id=tab-b")
        assert other.get_json()["state"] == "idle"

        stopped = client.post(
            "/api/browse/dataset/stop",
            json={"client_id": "tab-a"},
        )
        assert stopped.status_code == 200
    finally:
        manager.close()


def test_nearby_rejects_stale_revision(tmp_path):
    client, _api, manager, item = _client(tmp_path)
    try:
        item["revision"] = "0" * 64
        response = client.post(
            "/api/browse/nearby",
            json={"client_id": "tab-a", "generation": 1, "items": [item]},
        )
        assert response.status_code == 409
        assert response.get_json() == {"error": "invalid or stale image"}
    finally:
        manager.close()


def test_cache_clear_preserves_current_revision(tmp_path):
    client, api, manager, item = _client(tmp_path)
    try:
        revision = item["revision"]
        source_revision = probe_source(
            api.sandbox.root / "plate.png",
            sandbox_root=api.sandbox.root,
            relative_path="plate.png",
        )
        handle = api.select("tab-a", 1, source_revision)
        assert handle.complete.wait(5)
        response = client.post(
            "/api/browse/cache/clear",
            json={"client_id": "tab-a", "current_revision": revision},
        )
        assert response.status_code == 200
        assert api.cache.entry(revision).dzi_ready
    finally:
        manager.close()


def test_preparation_routes_reject_malformed_json(tmp_path):
    client, _api, manager, _item = _client(tmp_path)
    try:
        assert (
            client.post("/api/browse/dataset/start", data=b"nope").status_code
            == 400
        )
        assert (
            client.get("/api/browse/dataset/status?client_id=").status_code
            == 400
        )
    finally:
        manager.close()

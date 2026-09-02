import base64
import os
from pathlib import Path

import dash
import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse import _source_render as sr
from phenotypic.gui.browse import _tile_routes
from phenotypic.gui.browse._source_probe import probe_source
from phenotypic.gui.shell._sandbox import SandboxRoot


def _write_fake_ngff_image_group(store: Path, member: str) -> None:
    """Write the minimum real group/array metadata the route validates."""
    group = store / member
    (group / "0").mkdir(parents=True, exist_ok=True)
    (group / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"group","attributes":{"ome":{'
        '"version":"0.5","multiscales":[{"datasets":[{"path":"0"}]}]}}}',
        encoding="utf-8",
    )
    (group / "0" / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"array"}', encoding="utf-8"
    )


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
    return app.server.test_client(), "plates/b7/A1.png", sandbox_root


def test_manifest_then_tile(app_and_root):
    client, rel, _ = app_and_root
    token = sr.encode_token(rel)
    manifest = client.get(f"/tiles/{token}.dzi")
    assert manifest.status_code == 200
    assert b"<Image" in manifest.data
    tile = client.get(f"/tiles/{token}_files/0/0_0.png")
    assert tile.status_code == 200
    assert tile.mimetype == "image/png"


def test_revisioned_preview_manifest_and_tile_are_immutable(app_and_root):
    client, rel, sandbox_root = app_and_root
    token = sr.encode_token(rel)
    revision = probe_source(
        sandbox_root / rel,
        sandbox_root=sandbox_root,
        relative_path=rel,
    )

    assert (
        client.get(
            f"/assets/{token}/{revision.cache_key}/preview-if-ready.png"
        ).status_code
        == 404
    )
    preview = client.get(f"/assets/{token}/{revision.cache_key}/preview.png")
    assert preview.status_code == 200
    assert (
        preview.headers["Cache-Control"]
        == "private, max-age=31536000, immutable"
    )

    manifest = client.get(f"/assets/{token}/{revision.cache_key}/image.dzi")
    assert manifest.status_code == 200
    assert "cache;dur=" in manifest.headers["Server-Timing"]
    assert "queue;dur=" in manifest.headers["Server-Timing"]
    assert (
        manifest.headers["Cache-Control"]
        == "private, max-age=31536000, immutable"
    )

    tile = client.get(
        f"/assets/{token}/{revision.cache_key}/image_files/0/0_0.png"
    )
    assert tile.status_code == 200
    assert (
        tile.headers["Cache-Control"] == "private, max-age=31536000, immutable"
    )


def test_revisioned_asset_rejects_stale_revision(app_and_root):
    client, rel, _ = app_and_root
    token = sr.encode_token(rel)
    stale = "0" * 64
    response = client.get(f"/assets/{token}/{stale}/image.dzi")
    assert response.status_code == 409
    assert response.get_json() == {"error": "source image changed"}


def test_published_plain_zarr_store_is_served_as_generation_addressed_bytes(
    monkeypatch, tmp_path
) -> None:
    """Browse hands Viv store bytes; it must not build a PNG/DZI pyramid."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(cache))
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    chunk = store / "rgb" / "0" / "c" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"chunk-bytes")
    _write_fake_ngff_image_group(store, "rgb")
    root = store / "zarr.json"
    root.write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"rgb"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    client = app.server.test_client()
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")

    response = client.get(
        f"/assets/{token}/{revision.cache_key}/zarr/rgb/0/c/0"
    )

    assert response.status_code == 200
    assert response.data == b"chunk-bytes"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Cache-Control"].endswith("immutable")
    ranged = client.get(
        f"/assets/{token}/{revision.cache_key}/zarr/rgb/0/c/0",
        headers={"Range": "bytes=0-4"},
    )
    assert ranged.status_code == 206
    assert ranged.data == b"chunk"
    assert ranged.headers["Content-Range"] == "bytes 0-4/11"
    assert not list(cache.rglob("*.dzi"))


def test_published_store_range_does_not_materialize_the_member(
    monkeypatch, tmp_path
) -> None:
    """A range request streams the shard instead of reading all of it."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    chunk = store / "rgb" / "0" / "c" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"0123456789")
    _write_fake_ngff_image_group(store, "rgb")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"rgb"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")
    original_read_bytes = Path.read_bytes

    def reject_chunk_materialization(path: Path) -> bytes:
        if path == chunk:
            raise AssertionError("route materialized a complete Zarr member")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_chunk_materialization)

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/rgb/0/c/0",
        headers={"Range": "bytes=2-5"},
    )

    assert response.status_code == 206
    assert response.data == b"2345"


def test_published_store_route_exposes_only_declared_image_roots(
    tmp_path,
) -> None:
    """Embedded measurement tables are not part of the image-byte API."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    image_member = store / "original" / "0" / "c" / "0"
    table_member = store / "tables" / "measurements" / "table.parquet"
    image_member.parent.mkdir(parents=True)
    table_member.parent.mkdir(parents=True)
    image_member.write_bytes(b"image-metadata")
    table_member.write_bytes(b"private-table")
    _write_fake_ngff_image_group(store, "original")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"original"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")
    client = app.server.test_client()

    image = client.get(
        f"/assets/{token}/{revision.cache_key}/zarr/original/0/c/0"
    )
    table = client.get(
        f"/assets/{token}/{revision.cache_key}/zarr/"
        "tables/measurements/table.parquet"
    )

    assert image.status_code == 200
    assert image.data == b"image-metadata"
    assert table.status_code == 404


def test_store_declaration_cannot_authorize_reserved_tables_root(
    tmp_path,
) -> None:
    """A crafted series map must not turn measurement tables into images."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    table_group = store / "tables"
    table_group.mkdir(parents=True)
    (table_group / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"group","attributes":{"ome":{'
        '"version":"0.5","multiscales":[{'
        '"datasets":[{"path":"0"}]}]}}}',
        encoding="utf-8",
    )
    (table_group / "0").mkdir()
    (table_group / "0" / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"array"}', encoding="utf-8"
    )
    secret = table_group / "measurements.parquet"
    secret.write_bytes(b"private-table")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"tables"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/"
        "tables/measurements.parquet"
    )

    assert response.status_code == 404
    assert response.data != b"private-table"


def test_store_declaration_cannot_authorize_an_arbitrary_directory(
    tmp_path,
) -> None:
    """A declared series must itself carry NGFF image-group metadata."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    payload = store / "payload"
    payload.mkdir(parents=True)
    victim = payload / "secret.bin"
    victim.write_bytes(b"not-an-image-group")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"payload"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/payload/secret.bin"
    )

    assert response.status_code == 404
    assert response.data != b"not-an-image-group"


def test_label_declaration_requires_an_ngff_image_label_group(
    tmp_path,
) -> None:
    """An ordinary image group cannot be relabelled to widen authorization."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    _write_fake_ngff_image_group(store, "labelish")
    victim = store / "labelish" / "0" / "c" / "0"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"not-a-label-group")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{},"labels":{"objmap":"labelish"}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/labelish/0/c/0"
    )

    assert response.status_code == 404
    assert response.data != b"not-a-label-group"


def test_store_member_open_rejects_component_swapped_to_external_symlink(
    monkeypatch, tmp_path
) -> None:
    """Opening is anchored, so a validation/open swap cannot escape."""
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("no-follow opens are unavailable")
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    member = store / "rgb" / "0" / "c" / "0"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"safe-store-member")
    (store / "rgb" / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"group","attributes":{"ome":{'
        '"version":"0.5","multiscales":[{"datasets":[{"path":"0"}]}]}}}',
        encoding="utf-8",
    )
    (store / "rgb" / "0" / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"array"}', encoding="utf-8"
    )
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"rgb"},"labels":{}}}}',
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside_member = outside / "0" / "c" / "0"
    outside_member.parent.mkdir(parents=True)
    outside_member.write_bytes(b"external-victim")
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")
    original_open = os.open
    swapped = False

    def swap_component_then_open(
        path, flags, mode=0o777, *, dir_fd=None
    ) -> int:
        nonlocal swapped
        if path == "rgb" and dir_fd is not None and not swapped:
            swapped = True
            (store / "rgb").rename(store / "rgb-original")
            (store / "rgb").symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_component_then_open)

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/rgb/0/c/0"
    )

    assert swapped
    assert response.status_code == 404
    assert response.data != b"external-victim"


@pytest.mark.parametrize(
    "member",
    ["", "%2E", "rgb//0", "%00", "%FF"],
)
def test_malformed_store_member_fails_closed(
    tmp_path, member: str
) -> None:
    """Malformed URL members never escape to Dash or raise a server error."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/{member}",
        follow_redirects=False,
    )

    assert response.status_code in {400, 404}


def test_published_store_route_maps_unstable_root_to_conflict(
    monkeypatch, tmp_path
) -> None:
    """A root change during member open fails closed as a stale generation."""
    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "plate.zarr"
    member = store / "rgb" / "0" / "c" / "0"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"image-metadata")
    _write_fake_ngff_image_group(store, "rgb")
    (store / "zarr.json").write_text(
        '{"attributes":{"phenotypic":{'
        '"store_schema_version":3,'
        '"publication_protocol":"root-last-immutable-v1",'
        '"series":{"rgb":"rgb"},"labels":{}}}}',
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=sandbox_root)
    token = sr.encode_token("plate.zarr")
    observations = iter(
        [
            revision.store_revision,
            revision.store_revision,
            OSError("root changed during revision inspection"),
        ]
    )

    def unstable_publication(
        _store: Path, *, root_dir_fd: int | None = None
    ) -> str:
        del root_dir_fd
        observed = next(observations)
        if isinstance(observed, OSError):
            raise observed
        assert observed is not None
        return observed

    monkeypatch.setattr(
        _tile_routes, "store_publication_token", unstable_publication
    )

    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/rgb/0/c/0"
    )

    assert response.status_code == 409
    assert response.get_json() == {"error": "source image changed"}


def test_mutable_third_party_store_fails_closed_without_asset_rescan(
    monkeypatch, tmp_path
) -> None:
    """No publication token means no immutable multi-request Viv source."""
    from phenotypic.sdk_ import _io_constants as io

    sandbox_root = tmp_path / "sandbox"
    store = sandbox_root / "third-party.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    sandbox = SandboxRoot.from_path(sandbox_root)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    token = sr.encode_token("third-party.zarr")
    monkeypatch.setattr(
        io,
        "_store_revision_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("asset request recursively traversed the store")
        ),
    )

    response = app.server.test_client().get(
        f"/assets/{token}/{'0' * 64}/zarr/zarr.json"
    )

    assert response.status_code == 422
    assert "publication token" in response.get_json()["error"]


def test_malformed_token_404(app_and_root):
    client, _, _ = app_and_root
    assert client.get("/tiles/not%20a%20token.dzi").status_code == 404


def test_escape_token_404(app_and_root):
    client, _, _ = app_and_root
    token = sr.encode_token("../../etc/passwd")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_raw_unavailable_422(app_and_root, monkeypatch, tmp_path):
    client, _, _ = app_and_root

    def _boom(original, cache_png):
        raise sr.SourceRenderUnavailable("nope")

    monkeypatch.setattr(_tile_routes._source_render, "normalize_to_png", _boom)
    # A token that resolves to the existing image so render is attempted.
    token = sr.encode_token("plates/b7/A1.png")
    resp = client.get(f"/tiles/{token}.dzi")
    assert resp.status_code == 422
    # The client body is a fixed message; the exception text ("nope") never
    # leaks into the response.
    assert resp.get_json() == {
        "error": "source image cannot be rendered on this platform"
    }


# ---------------------------------------------------------------------------
# Adversarial traversal regression guards. Every case below already rejects
# correctly today; these lock in the containment so a future regression in
# ``sandbox.resolve``, the token guard, or the tile-filename guard can't pass
# silently. The manifest route maps malformed/escape/non-file tokens to 404.
# ---------------------------------------------------------------------------
def test_absolute_path_token_404(app_and_root):
    # An absolute path is joined-then-resolved by the sandbox; it escapes the
    # root and must 404 rather than reading /etc/passwd.
    client, _, _ = app_and_root
    token = sr.encode_token("/etc/passwd")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_nul_byte_token_404(app_and_root):
    # A NUL byte in the decoded path makes the filesystem ``lstat`` raise
    # ValueError; the route must catch it as a 404, not a 500.
    client, _, _ = app_and_root
    token = sr.encode_token("plates/b7/A1.png\x00.txt")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_non_utf8_decode_token_404(app_and_root):
    # Valid base64url that decodes to non-UTF8 bytes: ``decode_token`` raises,
    # and the broad-except in ``_resolve_original`` maps that to 404.
    client, _, _ = app_and_root
    token = base64.urlsafe_b64encode(b"\xff\xfe\xff").decode().rstrip("=")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_directory_token_404(app_and_root):
    # An in-sandbox real directory resolves fine but ``is_file()`` rejects it.
    client, _, _ = app_and_root
    token = sr.encode_token("plates/b7")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_symlink_escape_token_404(app_and_root, tmp_path):
    # A symlink INSIDE the sandbox pointing OUTSIDE it: ``sandbox.resolve``
    # follows the link, sees the resolved target escapes the root, and raises
    # ValueError -> 404. This is the vector the design spec explicitly calls
    # out.
    client, _, sandbox_root = app_and_root
    outside = tmp_path / "outside_secret.png"
    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(outside)
    link = sandbox_root / "plates" / "b7" / "escape.png"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")
    token = sr.encode_token("plates/b7/escape.png")
    assert client.get(f"/tiles/{token}.dzi").status_code == 404


def test_tile_filename_traversal_404(app_and_root):
    # The tile endpoint's filename guard (secure_filename + ``\d+_\d+\.png``)
    # rejects traversal-style and wrong-extension filenames that arrive as a
    # single path segment. The manifest must run first so the cache dirs exist
    # (otherwise the "tile cache missing" branch would 404 for the wrong
    # reason).
    #
    # NOTE: a raw/encoded-slash traversal like ``..%2f..%2fsecret.png`` does
    # NOT reach this endpoint at all — Werkzeug decodes ``%2f`` to ``/`` during
    # URL normalization, so the request falls through to Dash's ``<path:path>``
    # catch-all (which serves its app-shell HTML, never a sandbox file). The
    # blueprint's filename guard only governs single-segment filenames, so we
    # assert it against those.
    client, rel, _ = app_and_root
    token = sr.encode_token(rel)
    assert client.get(f"/tiles/{token}.dzi").status_code == 200
    # Encoded-dot traversal that stays a single segment -> reaches the guard.
    resp_dots = client.get(f"/tiles/{token}_files/0/%2e%2e")
    assert resp_dots.status_code == 404
    assert (
        resp_dots.mimetype == "application/json"
    )  # our guard, not Dash's shell
    # Wrong extension (not ``\d+_\d+\.png``).
    resp_ext = client.get(f"/tiles/{token}_files/0/0_0.jpg")
    assert resp_ext.status_code == 404
    assert resp_ext.mimetype == "application/json"

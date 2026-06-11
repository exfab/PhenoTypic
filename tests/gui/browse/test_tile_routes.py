import base64
import os

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
    assert resp_dots.mimetype == "application/json"  # our guard, not Dash's shell
    # Wrong extension (not ``\d+_\d+\.png``).
    resp_ext = client.get(f"/tiles/{token}_files/0/0_0.jpg")
    assert resp_ext.status_code == 404
    assert resp_ext.mimetype == "application/json"

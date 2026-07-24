"""Timeline thumbnail route: (dataset, stem) → cached downscaled overlay PNG."""
from __future__ import annotations

import io
from pathlib import Path

import dash
import polars as pl
from PIL import Image as PILImage

from phenotypic.gui._config import BROWSE_THUMB_URL_SEGMENT, VIEWER_THUMB_URL_SEGMENT
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _thumb_routes
from tests._output_layout import write_master, write_measurements_mirror
from phenotypic.schema import METADATA


def test_viewer_and_browse_thumb_segments_are_distinct() -> None:
    # S2: the two surfaces' thumbnail routes must not collide if ever co-mounted
    # on one Flask server; the Browse segment is "thumb", the viewer's is
    # "timeline-thumb" (Phase 1 _config).
    assert VIEWER_THUMB_URL_SEGMENT == "timeline-thumb"
    assert BROWSE_THUMB_URL_SEGMENT == "thumb"
    assert VIEWER_THUMB_URL_SEGMENT != BROWSE_THUMB_URL_SEGMENT


def _output_root(tmp_path: Path) -> OutputRoot:
    cli_out = tmp_path / "out"
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["ds", "ds"],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "Metadata_ImageNumber": pl.Series([1, 2], dtype=pl.Int64),
            "Object_Label": [1, 2],
            "Size_Area": [1.0, 2.0],
        }
    )
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / "ds"
    overlays.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (200, 100), (0, 64, 128)).save(overlays / "a.png")
    PILImage.new("RGB", (200, 100), (0, 64, 128)).save(overlays / "b.png")
    return OutputRoot.discover(cli_out)


def _client(tmp_path: Path):
    root = _output_root(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # layout-less Dash 500s on first request
    _thumb_routes.register(app, root)
    return app.server.test_client(), root


def test_cell_ref_round_trips() -> None:
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    assert _thumb_routes.decode_cell_ref(ident) == ("ds", "a")


def test_thumb_happy_path_returns_bucketed_png(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    resp = client.get(f"/timeline-thumb/{ident}?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128


def test_thumb_unknown_pair_is_404(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    ident = _thumb_routes.encode_cell_ref("ds", "does-not-exist")
    assert client.get(f"/timeline-thumb/{ident}?size=128").status_code == 404


def test_thumb_unsafe_identity_is_404(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    assert client.get("/timeline-thumb/..%2F..%2Fetc/passwd?size=128").status_code == 404


def test_thumb_cache_persists_under_viewer_cache(tmp_path: Path) -> None:
    client, root = _client(tmp_path)
    source_before = {
        path.relative_to(root.root).as_posix(): path.read_bytes()
        for path in root.root.rglob("*")
        if path.is_file()
    }
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    assert client.get(f"/timeline-thumb/{ident}?size=128").status_code == 200
    cache_dir = root.viewer_cache_dir / "timeline_thumbs"
    assert cache_dir.is_dir()
    assert list(cache_dir.glob("*.png"))  # a cached thumbnail was written
    assert not (root.root / ".viewer_cache").exists()
    source_after = {
        path.relative_to(root.root).as_posix(): path.read_bytes()
        for path in root.root.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before

"""Flask-test-client render check: the Timeline grid's data-src resolves 200.

No-browser integration lane (Task 6 Step 1b): with a time column present, the
rendered grid's first populated cell carries a ``data-src`` pointing at
``/<VIEWER_THUMB_URL_SEGMENT>/...`` that resolves 200 through the registered
thumbnail route. Proves the ``url_builder`` prefix + ``encode_cell_ref`` wiring
end-to-end without a browser. Named with ``timeline`` so ``-k timeline`` catches it.
"""
from __future__ import annotations

import io
from pathlib import Path

import dash
import polars as pl
from PIL import Image as PILImage

from phenotypic.gui._config import VIEWER_THUMB_URL_SEGMENT
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _thumb_routes
from phenotypic.gui.results_viewer.timeline_view._layout import (
    build_timeline_grid_component,
)
from tests._output_layout import write_master, write_measurements_mirror
from phenotypic.schema import METADATA


def _output_root(tmp_path: Path) -> OutputRoot:
    cli_out = tmp_path / "out"
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["ds", "ds"],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "Metadata_ImageNumber": pl.Series([1, 2], dtype=pl.Int64),
            "Metadata_PlateNum": ["1", "2"],
            "Object_Label": [1, 2],
            "Size_Area": [1.0, 2.0],
        }
    )
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / "ds"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (200, 100), (0, 64, 128)).save(overlays / f"{stem}.png")
    return OutputRoot.discover(
        cli_out,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )


def _walk(component):
    stack = [component]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)


def _first_data_src(component) -> str | None:
    for node in _walk(component):
        to_json = getattr(node, "to_plotly_json", None)
        if to_json is None:
            continue
        props = to_json().get("props", {})
        src = props.get("data-src")
        if isinstance(src, str) and src:
            return src
    return None


def test_timeline_thumb_url_resolves(tmp_path: Path) -> None:
    root = _output_root(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # layout-less Dash 500s on first request
    _thumb_routes.register(app, root)

    component, show_empty, _n = build_timeline_grid_component(
        root,
        root.master_df,
        row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    assert show_empty is False

    data_src = _first_data_src(component)
    assert data_src is not None, "grid emitted no data-src"
    # url_builder prefixes the mount ("/" by default) + the viewer thumb segment.
    assert f"/{VIEWER_THUMB_URL_SEGMENT}/" in data_src, data_src

    client = app.server.test_client()
    resp = client.get(data_src)
    assert resp.status_code == 200, (data_src, resp.status_code)
    assert resp.mimetype == "image/png"
    PILImage.open(io.BytesIO(resp.data))  # decodes

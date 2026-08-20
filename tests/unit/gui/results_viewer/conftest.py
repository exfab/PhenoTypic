"""Fixtures for the Results-viewer unit suite.

Hosts :func:`live_viewer`: a full-run output root plus a registered tile
blueprint, driven through Flask's test client. It is what the tile-cache
invalidation suite republishes stores under.

**Why not Playwright.** The plan specified this fixture in
``tests/e2e/gui/conftest.py``. ``tests/e2e`` is not in ``testpaths``
(``pyproject.toml:218``) and is gated on ``PLAYWRIGHT=1``, so a fixture built
there would guard a lane that does not run. Everything these tests assert --
that a promote changes the bytes the tile route serves, that an unpublished
store reuses the cache, that an undecodable store answers 422 -- is
observable at the Flask boundary, which is where the DZI cache, the source
token, and the content token all live. A browser adds no evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import dash
import numpy as np
import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import Image
from phenotypic.gui.results_viewer import _tile_routes
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

from tests._output_layout import write_master


def _image_with_objmap(value: int) -> Image:
    """Build a 64x64 image whose objmap is a ``value``-sized labelled square.

    The square's EXTENT varies with *value*, not only its label id.
    ``_label_map_to_rgb`` colours via ``skimage.color.label2rgb``, which
    assigns colours by the RANK of the unique labels present, not by their
    values -- so a single label ``1`` and a single label ``7`` render to
    byte-identical pixels. A republish that changed only the id would be
    invisible to the renderer, and a test built on it would fail no matter
    how correct the cache invalidation was.
    """
    image = Image(arr=np.zeros((64, 64, 3), dtype=np.uint8))
    labeled = np.zeros((64, 64), dtype=np.int32)
    edge = 8 + 4 * value
    labeled[8:edge, 8:edge] = value
    image.objmap[:] = labeled
    return image


@dataclass
class LiveViewer:  # noqa: D101 - documented below
    """A registered tile route over a real, republishable output root."""

    client: object
    root: Path
    output_root: OutputRoot

    def request_tile(self, dataset: str, stem: str, *, layer: str):
        """Return the raw ``.dzi`` manifest response for one (image, layer)."""
        return self.client.get(f"/tiles/{dataset}/{stem}.dzi?layer={layer}")

    def get_tile(self, dataset: str, stem: str, *, layer: str) -> bytes:
        """Return the rendered source PNG bytes behind one served pyramid.

        The manifest response is XML describing the pyramid, and it is
        identical whatever the pixels are -- so asserting on it would pass
        against a stale tile. The source PNG the DZI was tiled from is the
        artifact that actually carries the pixels.
        """
        response = self.request_tile(dataset, stem, layer=layer)
        assert response.status_code == 200, response.get_data(as_text=True)
        cache_dir = _tile_routes._dzi_cache_dir_for(
            self.output_root.cache_dir, dataset, stem, layer
        )
        return (cache_dir / f"{stem}.png").read_bytes()

    def republish_with_objmap(
        self, dataset: str, stem: str, *, value: int
    ) -> Path:
        """Rewrite the image's objmap and publish it through ``save2zarr``.

        A promote, not an in-place write: nothing in the design opens a
        promoted store for writing, so a promote is the only way a store's
        contents ever change.
        """
        store = zarr_store_path(self.root, dataset, stem)
        image = _image_with_objmap(value)
        return image.save2zarr(store)

    def rebind(self) -> None:
        """Re-discover the output and re-register the routes.

        The viewer binds to a snapshot at discovery. Anything that changes a
        store's identity afterwards makes that binding stale, and every
        pixel route answers ``409`` until the user Refreshes -- which is the
        intended contract, not an obstacle. This is that Refresh.
        """
        self.output_root = OutputRoot.discover(
            self.root,
            cache_root=self.root.parent / ".test-phenotypic-viewer-cache",
        )
        app = dash.Dash(f"rebound-{id(self)}-{id(self.output_root)}")
        app.layout = dash.html.Div()
        _tile_routes.register(app, self.output_root)
        self.client = app.server.test_client()

    def corrupt_schema_version(self, dataset: str, stem: str) -> None:
        """Stamp a future ``store_schema_version`` into the store's root."""
        root = zarr_store_path(self.root, dataset, stem) / STORE_ROOT_JSON
        payload = json.loads(root.read_text(encoding="utf-8"))
        payload["attributes"][PhenotypicAttr.ROOT][
            PhenotypicAttr.STORE_SCHEMA_VERSION
        ] = 999
        root.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def live_viewer(tmp_path: Path) -> LiveViewer:
    """A viewer bound to a full run with one store-backed image."""
    dataset, stem = "d1", "img001"
    write_master(
        tmp_path,
        pl.DataFrame(
            {
                "Metadata_Dataset": [dataset],
                str(IMAGE.IMAGE_NAME): [stem],
                "Object_Label": [1],
                "Bbox_CenterRR": [16.0],
                "Bbox_CenterCC": [16.0],
            }
        ),
    )
    (tmp_path / "results" / dataset / "measurements").mkdir(parents=True)

    image = _image_with_objmap(1)
    store = zarr_store_path(tmp_path, dataset, stem)
    store.parent.mkdir(parents=True, exist_ok=True)
    image.save2zarr(store)

    overlay_dir = tmp_path / "deliverables" / "overlays" / dataset
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (64, 64), (10, 80, 160)).save(
        overlay_dir / f"{stem}.png", format="PNG"
    )

    output_root = OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )
    app = dash.Dash(__name__)
    # Dash 4.x validates the layout in a before_request hook; a trivial
    # layout keeps that from 500-ing before the request reaches the
    # blueprint.
    app.layout = dash.html.Div()
    _tile_routes.register(app, output_root)
    return LiveViewer(
        client=app.server.test_client(),
        root=tmp_path,
        output_root=output_root,
    )

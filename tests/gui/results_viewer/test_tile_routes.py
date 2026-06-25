"""Tests for :mod:`phenotypic.gui.results_viewer._tile_routes`.

Task 9 gave the DZI deep-zoom cache a *layer* dimension: the manifest and
tile routes now key their pyramid by ``(dataset, stem, layer)`` so the same
image can cache an ``rgb`` pyramid alongside an ``objmap`` one without
collision. ``?layer=`` selects which full-res HDF layer sources the pixels
(default ``rgb`` for a full run); a standalone deliverables bundle (no
per-image HDF) — or an explicit ``?layer=overlay`` — tiles the baked overlay
PNG instead.

The pure resolution helpers (:func:`_dzi_cache_dir_for`,
:func:`_resolve_dzi_layer`) are unit-tested directly; the manifest/tile
route wiring is smoke-tested through a Flask test client against a tmp output
dir carrying a real per-image HDF (built via ``Image(...).save2hdf5``),
mirroring the neighbouring ``colony_view/test_crop_routes.py`` pattern but
registering only the tile blueprint on a bare Dash app (the
``browse/test_tile_routes.py`` pattern).
"""

from __future__ import annotations

from pathlib import Path

import dash
import numpy as np
import polars as pl
import pytest

from phenotypic.gui.results_viewer import _tile_routes
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._tile_routes import (
    _OVERLAY_LAYER,
    _dzi_cache_dir_for,
    _resolve_dzi_layer,
)

from tests._output_layout import write_master


# ---------------------------------------------------------------------------
# _dzi_cache_dir_for — per-(image, layer) cache directory
# ---------------------------------------------------------------------------


def test_dzi_cache_dir_for_appends_layer_as_last_component(tmp_path: Path) -> None:
    """The layer is the final path component of ``<cache_root>/<ds>/<stem>/<layer>``."""
    cache_dir = _dzi_cache_dir_for(tmp_path, "d1", "img001", "rgb")
    assert cache_dir == tmp_path / "d1" / "img001" / "rgb"
    assert cache_dir.name == "rgb"


def test_dzi_cache_dir_for_distinct_dirs_per_layer(tmp_path: Path) -> None:
    """Two layers for the same image resolve to distinct, layer-named dirs."""
    rgb_dir = _dzi_cache_dir_for(tmp_path, "d1", "img001", "rgb")
    objmap_dir = _dzi_cache_dir_for(tmp_path, "d1", "img001", "objmap")

    assert rgb_dir != objmap_dir
    assert rgb_dir.name == "rgb"
    assert objmap_dir.name == "objmap"
    # They share the same (dataset, stem) parent; only the leaf differs.
    assert rgb_dir.parent == objmap_dir.parent


# ---------------------------------------------------------------------------
# _resolve_dzi_layer — raw ?layer= value -> cache-dir key (or None)
# ---------------------------------------------------------------------------


def test_resolve_layer_none_full_run_defaults_to_rgb() -> None:
    """Omitted ``?layer=`` on a full run (has_results) defaults to ``rgb``."""
    assert (
        _resolve_dzi_layer(None, has_results=True, has_hdf=True) == "rgb"
    )


def test_resolve_layer_none_standalone_falls_back_to_overlay() -> None:
    """Omitted ``?layer=`` on a standalone bundle (no results) -> overlay sentinel."""
    assert (
        _resolve_dzi_layer(None, has_results=False, has_hdf=False)
        == _OVERLAY_LAYER
    )


def test_resolve_layer_hdf_layer_without_hdf_collapses_to_overlay() -> None:
    """A named HDF layer with no per-image HDF collapses to the overlay sentinel."""
    assert (
        _resolve_dzi_layer("objmap", has_results=True, has_hdf=False)
        == _OVERLAY_LAYER
    )


def test_resolve_layer_invalid_value_returns_none() -> None:
    """An unrecognised ``?layer=`` value resolves to ``None`` (caller 404s)."""
    assert (
        _resolve_dzi_layer("not-a-layer", has_results=True, has_hdf=True)
        is None
    )


def test_resolve_layer_explicit_overlay_stays_overlay() -> None:
    """Explicit ``?layer=overlay`` resolves to the overlay sentinel even with an HDF."""
    assert (
        _resolve_dzi_layer(_OVERLAY_LAYER, has_results=True, has_hdf=True)
        == _OVERLAY_LAYER
    )


def test_resolve_layer_hdf_layer_with_hdf_passes_through() -> None:
    """A valid HDF layer with an HDF present passes through unchanged."""
    assert (
        _resolve_dzi_layer("objmap", has_results=True, has_hdf=True)
        == "objmap"
    )


# ---------------------------------------------------------------------------
# Manifest / tile route smoke — HDF-layer pyramid lands in the per-layer dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_and_root(tmp_path: Path):
    """Seed a full-run output dir with a real per-image HDF and register routes.

    Lays out::

        <tmp>/deliverables/master_measurements.parquet  (1 colony in dataset 'd1')
        <tmp>/results/d1/hdf/img001.h5                   (real Image w/ objmap)
        <tmp>/results/d1/measurements/                   (enables discovery)
        <tmp>/deliverables/overlays/d1/img001.png        (overlay fallback)

    Returns ``(test_client, output_root, dataset, stem)``.
    """
    from phenotypic import Image
    from PIL import Image as PILImage

    dataset, stem = "d1", "img001"

    # master_measurements.parquet under deliverables/.
    master = pl.DataFrame(
        {
            "Metadata_Dataset": [dataset],
            "Metadata_ImageFile": [stem],
            "Object_Label": [1],
        }
    )
    write_master(tmp_path, master)

    # Real per-image HDF with a non-trivial objmap (two labels).
    hdf_dir = tmp_path / "results" / dataset / "hdf"
    hdf_dir.mkdir(parents=True)
    (tmp_path / "results" / dataset / "measurements").mkdir(parents=True)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = (200, 40, 40)
    img = Image(arr=rgb)
    labeled = np.zeros((64, 64), dtype=np.int32)
    labeled[8:24, 8:24] = 1
    labeled[36:52, 36:52] = 2
    img.objmap[:] = labeled
    img.save2hdf5(str(hdf_dir / f"{stem}.h5"))

    # Overlay PNG (the standalone / explicit-overlay fallback source).
    overlay_dir = tmp_path / "deliverables" / "overlays" / dataset
    overlay_dir.mkdir(parents=True)
    PILImage.new("RGB", (64, 64), (10, 80, 160)).save(
        overlay_dir / f"{stem}.png", format="PNG"
    )

    output_root = OutputRoot.discover(tmp_path)
    app = dash.Dash(__name__)
    # Dash 4.x validates the layout in a before_request hook; a trivial layout
    # keeps that from 500-ing before the request reaches the blueprint.
    app.layout = dash.html.Div()
    _tile_routes.register(app, output_root)
    return app.server.test_client(), output_root, dataset, stem


def test_manifest_hdf_layer_tiles_into_per_layer_cache_dir(client_and_root) -> None:
    """An ``?layer=objmap`` manifest tiles the HDF layer into ``.../<stem>/objmap``."""
    client, output_root, dataset, stem = client_and_root

    resp = client.get(f"/tiles/{dataset}/{stem}.dzi?layer=objmap")
    assert resp.status_code == 200
    assert b"<Image" in resp.data

    layer_dir = _dzi_cache_dir_for(output_root.cache_dir, dataset, stem, "objmap")
    # The pyramid (manifest + tiles) landed in the per-layer cache dir.
    assert (layer_dir / f"{stem}.dzi").is_file()
    assert (layer_dir / f"{stem}_files").is_dir()
    # The rendered HDF-layer source PNG was written there too.
    assert (layer_dir / f"{stem}.png").is_file()

    # A matching tile request serves a PNG out of the SAME per-layer pyramid.
    tile = client.get(f"/tiles/{dataset}/{stem}_files/0/0_0.png?layer=objmap")
    assert tile.status_code == 200
    assert tile.mimetype == "image/png"


def test_manifest_default_and_objmap_use_distinct_cache_dirs(client_and_root) -> None:
    """Default (rgb) and ``?layer=objmap`` manifests populate separate dirs."""
    client, output_root, dataset, stem = client_and_root

    assert client.get(f"/tiles/{dataset}/{stem}.dzi").status_code == 200
    assert (
        client.get(f"/tiles/{dataset}/{stem}.dzi?layer=objmap").status_code == 200
    )

    rgb_dir = _dzi_cache_dir_for(output_root.cache_dir, dataset, stem, "rgb")
    objmap_dir = _dzi_cache_dir_for(output_root.cache_dir, dataset, stem, "objmap")
    assert (rgb_dir / f"{stem}.dzi").is_file()
    assert (objmap_dir / f"{stem}.dzi").is_file()
    assert rgb_dir != objmap_dir


def test_manifest_explicit_overlay_tiles_overlay_png(client_and_root) -> None:
    """``?layer=overlay`` tiles the baked overlay PNG into the overlay cache dir."""
    client, output_root, dataset, stem = client_and_root

    resp = client.get(f"/tiles/{dataset}/{stem}.dzi?layer=overlay")
    assert resp.status_code == 200

    overlay_dir = _dzi_cache_dir_for(
        output_root.cache_dir, dataset, stem, _OVERLAY_LAYER
    )
    assert (overlay_dir / f"{stem}.dzi").is_file()
    # No rendered HDF source PNG in the overlay branch — the overlay PNG is the
    # source, read straight from deliverables/overlays/.
    assert not (overlay_dir / f"{stem}.png").is_file()


def test_manifest_invalid_layer_returns_404(client_and_root) -> None:
    """An unrecognised ``?layer=`` value 404s rather than tiling."""
    client, _, dataset, stem = client_and_root
    resp = client.get(f"/tiles/{dataset}/{stem}.dzi?layer=bogus")
    assert resp.status_code == 404


def test_tile_endpoint_invalid_layer_returns_404(client_and_root) -> None:
    """The tile endpoint also rejects an unrecognised ``?layer=`` value."""
    client, _, dataset, stem = client_and_root
    # Generate the default pyramid first so the failure is the layer guard,
    # not a missing cache dir.
    assert client.get(f"/tiles/{dataset}/{stem}.dzi").status_code == 200
    resp = client.get(f"/tiles/{dataset}/{stem}_files/0/0_0.png?layer=bogus")
    assert resp.status_code == 404

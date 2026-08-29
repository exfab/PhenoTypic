"""The source spec is read from the store, never inferred.

The label-path test is spec section 8's check: a store whose primary series
is ``gray`` (no ``rgb``) must resolve its objmap through
``phenotypic.labels.objmap``, proving nothing hard-codes
``rgb/labels/objmap``.

Every fixture here is written by the real writer (see ``conftest.py``). A
hand-edited ``zarr.json`` would let these tests agree with a store layout no
writer produces, which is precisely the drift the contract exists to close.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import GridImage
from phenotypic.gui._shared.tiles import StoreUnreadable
from phenotypic.gui.results_viewer._store_source import build_source_spec


def test_series_come_from_the_store_in_primary_first_order(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["seriesPath"] == "rgb"
    assert spec["series"][0] == "rgb"
    assert set(spec["series"]) == {"rgb", "gray", "detect_mat"}


def test_an_original_series_is_listed(store_with_original):
    """``_write_store_part`` appends "original" when the image carries one.

    A spec that filtered the series list to a literal set would silently drop
    a layer the writer legitimately produced -- and the byte route's readable
    set is derived from the same block, so the two would disagree and the
    Layers panel would offer a series the route 404s.
    """
    spec = build_source_spec(store_with_original, "/zarr/ds/orig.ome.zarr")
    assert "original" in spec["series"]
    assert spec["seriesPath"] == "rgb"


def test_label_path_is_read_not_constructed(gray_only_store):
    spec = build_source_spec(gray_only_store, "/zarr/ds/grayplate.ome.zarr")
    assert spec["seriesPath"] == "gray"
    assert not spec["labelPath"].startswith("rgb/")
    assert spec["labelPath"] == "gray/labels/objmap"


def test_a_store_with_no_label_image_yields_no_label_path(label_less_store):
    """``labels`` is omitted entirely, not emitted empty (ngff_.py:576-581).

    Most builder-preview stores are like this, because
    ``save_intermediate_zarr`` sets ``write_objmap = "objmap" in layers``.
    ``block["labels"]["objmap"]`` would raise ``KeyError`` on every one of
    them.
    """
    spec = build_source_spec(label_less_store, "/zarr/ds/prev.ome.zarr")
    assert spec["labelPath"] is None


def test_pyramid_ladder_is_read_not_recomputed(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["pyramid"]["levels"] >= 1
    assert spec["pyramid"]["downsample"]["label"] == "nearest"
    assert spec["pyramid"]["downsample"]["image"] == "mean"


def test_the_spec_is_a_valid_facade_source_spec(rgb_store):
    """``setSource`` validates ``storeUrl`` and ``seriesPath`` and throws.

    The facade's own contract (``_assets/viv_viewer.js``) names those two
    keys, so the dict this function returns is handed to it unmodified. A
    rename on either side would fail here rather than in a browser.
    """
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["storeUrl"] == "/zarr/ds/plate.ome.zarr"
    assert set(spec) == {
        "storeUrl",
        "token",
        "series",
        "seriesPath",
        "labelPath",
        "labelColorDomain",
        "pyramid",
    }


def test_label_color_domain_has_a_safe_generic_fallback(rgb_store):
    """Non-grid objmaps still receive a useful categorical hue domain."""
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["labelColorDomain"] == [0, 255]


def test_grid_capacity_bounds_the_label_color_domain(tmp_path):
    """Grid labels use the declared plate capacity for stable categorical hues."""
    image = GridImage(
        arr=np.zeros((64, 96, 3), dtype=np.uint8),
        nrows=8,
        ncols=12,
    )
    store = image.save2zarr(tmp_path / "grid.ome.zarr")

    spec = build_source_spec(store, "/zarr/ds/grid.ome.zarr")

    assert spec["labelColorDomain"] == [0, 96]


def test_the_token_identifies_this_generation_of_the_store(rgb_store):
    """Same value the byte route compares an incoming URL segment against."""
    from phenotypic.gui.results_viewer._zarr_routes import (
        store_generation_token,
    )

    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["token"] == store_generation_token(rgb_store)


def test_a_store_this_build_cannot_decode_raises_rather_than_opening(
    rgb_store,
):
    """Plate and Colony must agree about an undecodable store.

    ``_readable_block`` raises ``StoreUnreadable``; ``crop_colony``
    deliberately does not catch it and the caller turns it into a 422. A raw
    ``json.loads`` here would let Plate open a store Colony refuses, with the
    two surfaces disagreeing about one store.
    """
    import json

    from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

    root = rgb_store / STORE_ROOT_JSON
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"][PhenotypicAttr.ROOT][
        PhenotypicAttr.STORE_SCHEMA_VERSION
    ] = 999
    root.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StoreUnreadable):
        build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")


# ---- task 3.4: a mid-run zeros objmap is valid, and distinguishable ------


def test_an_all_zero_objmap_is_a_valid_source_not_an_error(stage1_store):
    """A store between Stage 1 and Stage 3 holds a zeros objmap.

    Backend behaviour (landed): Stage 2 is read-only, so the in-store objmap
    stays zeros until Stage 3 re-promotes. The Layers panel must offer the
    label layer normally -- an empty segmentation is the correct rendering of
    a correct store, not a condition to surface as a fault.
    """
    spec = build_source_spec(stage1_store, "/zarr/ds/stage1plate.ome.zarr")
    assert spec["labelPath"]
    assert "error" not in spec




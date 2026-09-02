"""Level selection follows the store's RECORDED ladder, ceil boundary included.

Backend section 1.3: levels halve until ``max(H, W) <= 512``, so
``levels = ceil(log2(max(H, W) / 512)) + 1``. A draft used ``floor``, which
terminates one level early and leaves a 4000x3000 plate's smallest level at
1000x750. That regression is what these tests exist to catch, so every
assertion here runs against a store written by the real writer -- never
against a formula restated in the test body.

**There is no ``select_pyramid_level`` exercise here, and its absence is the
point.** Once the Plate surface reads chunks in the browser, the level in use
is deck.gl's per-frame choice and the server selects nothing: the whole
server-side selection stack lost its only caller chain and was retired with
this change. What survives is a claim about what the WRITER recorded, which
holds whoever does the selecting -- and phase 5 asserts the browser's choice
against this same ladder.
"""

from __future__ import annotations

import json

import pytest

from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

from tests.unit.gui.results_viewer.conftest import _level_shapes


@pytest.mark.parametrize(
    ("extent", "expected_levels"),
    [(512, 1), (513, 2), (1024, 2), (1025, 3), (4000, 4)],
)
def test_the_written_store_records_the_ceil_ladder(
    store_at_extent, extent, expected_levels
):
    """The STORE's recorded ladder, not a formula re-derived here.

    ``floor`` would give 512->1, 513->1, 1025->2, 4000->3 -- so 513, 1025 and
    4000 each fail under the regression, and 513/1025 are the ceil boundaries
    specifically.
    """
    # Read the recorded block with plain ``json`` rather than reaching into
    # ``_shared/tiles.py``'s private ``_readable_block``: the claim here is
    # about what the WRITER recorded, so going through the reader's private
    # path would weaken it.
    store = store_at_extent(extent)
    root = json.loads((store / STORE_ROOT_JSON).read_text(encoding="utf-8"))
    block = root["attributes"][PhenotypicAttr.ROOT]
    assert block[PhenotypicAttr.PYRAMID]["levels"] == expected_levels


@pytest.mark.parametrize(
    ("extent", "expected_levels"),
    [(512, 1), (513, 2), (1024, 2), (1025, 3), (4000, 4)],
)
def test_the_recorded_ladder_is_the_one_actually_on_disk(
    store_at_extent, extent, expected_levels
):
    """Each declared level exists and halves, down to a level within stop_px.

    The recorded count is only worth reading if it describes the arrays that
    were written. A ``floor`` ladder would stop one level early, leaving the
    coarsest level's longest edge ABOVE 512 -- which is what the last
    assertion refuses.
    """
    store = store_at_extent(extent)
    shapes = _level_shapes(store, "rgb")

    assert len(shapes) == expected_levels
    for finer, coarser in zip(shapes, shapes[1:]):
        assert max(coarser[-2:]) == max(1, (max(finer[-2:]) + 1) // 2)
    assert max(shapes[-1][-2:]) <= 512
    if expected_levels > 1:
        assert max(shapes[-2][-2:]) > 512


def test_the_label_pyramid_matches_the_image_pyramid(store_at_extent):
    """One ladder per store, not one per series.

    ``levels`` is written once into ``attributes.phenotypic.pyramid`` and
    every series and the label image are written to that same count, so a
    client can resolve a label level from the image's level index.
    """
    store = store_at_extent(1025)
    assert _level_shapes(store, "rgb/labels/objmap") == [
        shape[-2:] for shape in _level_shapes(store, "rgb")
    ]

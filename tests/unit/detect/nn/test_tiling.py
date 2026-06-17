"""Shared tiling module (Spec 2b, Task 3).

The fixed-geometric tiling was extracted from ``_sam3_detector`` to ``_tiling``
so the semantic detectors reuse it. These tests exercise the new module path
directly (the SAM3 suite still imports the re-exported names and stays green),
plus the semantic union stitch.
"""

import numpy as np

from phenotypic.detect.nn._tiling import (
    _plan_tiles,
    _Tile,
    stitch_semantic_tiles,
)


# ---------------------------------------------------------------------------
# Planning (same contract as the SAM3 tiling tests, against _tiling directly)
# ---------------------------------------------------------------------------


class TestPlanTiles:
    def test_plan_fixed_tiles_cover_with_overlap(self):
        tiles = _plan_tiles((3000, 3000), tile_px=1008, overlap=0.15)
        assert all(t.h <= 1008 and t.w <= 1008 for t in tiles)
        covered = np.zeros((3000, 3000), bool)
        for t in tiles:
            covered[t.y0:t.y1, t.x0:t.x1] = True
        assert covered.all()

    def test_small_image_is_single_tile(self):
        tiles = _plan_tiles((500, 500), tile_px=1008, overlap=0.15)
        assert len(tiles) == 1
        t = tiles[0]
        assert (t.y0, t.x0, t.y1, t.x1) == (0, 0, 500, 500)

    def test_tile_dims_match_bounds(self):
        for t in _plan_tiles((2500, 1800), tile_px=1008, overlap=0.2):
            assert t.h == t.y1 - t.y0
            assert t.w == t.x1 - t.x0
            assert t.y1 <= 2500 and t.x1 <= 1800

    def test_sam3_detector_reexports_tiling(self):
        # Back-compat: _sam3_detector re-exports the extracted names.
        from phenotypic.detect.nn import _sam3_detector

        assert _sam3_detector._plan_tiles is _plan_tiles
        assert _sam3_detector._Tile is _Tile


# ---------------------------------------------------------------------------
# Semantic union stitch
# ---------------------------------------------------------------------------


class TestStitchSemanticTiles:
    def test_two_overlapping_tiles_union(self):
        # Two tiles whose overlap each mark part of a shared region; the union
        # covers both contributions (no NMS for semantic — just OR).
        tiles = [_Tile(0, 0, 6, 6), _Tile(0, 4, 6, 10)]
        m0 = np.zeros((6, 6), bool)
        m0[2:5, 3:6] = True  # right edge of tile 0
        m1 = np.zeros((6, 6), bool)
        m1[2:5, 0:3] = True  # left edge of tile 1 (overlaps tile 0's columns)
        full = stitch_semantic_tiles(tiles, [m0, m1], out_shape=(6, 10))
        assert full.dtype == bool and full.shape == (6, 10)
        # Tile 0 contribution at absolute cols 3..6.
        assert full[3, 3] and full[3, 5]
        # Tile 1 contribution at absolute cols 4..7 (x0=4 + local 0..3).
        assert full[3, 4] and full[3, 6]
        # Overlap region ORed, corners outside both stay false.
        assert not full[0, 0] and not full[5, 9]

    def test_single_tile_passthrough(self):
        tiles = [_Tile(0, 0, 4, 4)]
        m = np.zeros((4, 4), bool)
        m[1:3, 1:3] = True
        full = stitch_semantic_tiles(tiles, [m], out_shape=(4, 4))
        assert np.array_equal(full, m)

    def test_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError, match="tiles"):
            stitch_semantic_tiles([_Tile(0, 0, 2, 2)], [], out_shape=(2, 2))

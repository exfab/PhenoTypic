"""Shared tiling module (Spec 2b, Task 3; centroid-in-core merge, Task 5).

The fixed-geometric tiling was extracted from ``_sam3_detector`` to ``_tiling``
so the semantic detectors reuse it, and the cross-tile instance merge followed.
These tests exercise the new module path directly (the SAM3 suite still imports
the re-exported names and stays green): tile planning, the semantic union
stitch, the legacy IoU-NMS merge, and the centroid-in-core merge that replaces
it for tiled instance detection.
"""

import numpy as np
import pytest

from phenotypic.detect.nn._tiling import (
    _merge_tiles_iou_nms,
    _plan_tiles,
    _Tile,
    assign_by_centroid_core,
    owning_tile_index,
    stitch_semantic_tiles,
    tile_overlap_px,
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
        with pytest.raises(ValueError, match="tiles"):
            stitch_semantic_tiles([_Tile(0, 0, 2, 2)], [], out_shape=(2, 2))


# ---------------------------------------------------------------------------
# Instance merge: IoU-NMS (moved here from _sam3_detector)
# ---------------------------------------------------------------------------


class TestMergeTilesIouNms:
    def test_merge_dedups_overlapping_instances(self):
        a = np.zeros((10, 10), np.uint16)
        a[2:6, 2:6] = 1
        b = np.zeros((10, 10), np.uint16)
        b[2:6, 2:6] = 1  # same blob from neighbour tile
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 1

    def test_merge_keeps_distinct_instances(self):
        a = np.zeros((10, 10), np.uint16)
        a[1:3, 1:3] = 1
        b = np.zeros((10, 10), np.uint16)
        b[7:9, 7:9] = 1  # disjoint blob
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 2

    def test_sam3_detector_reexports_the_merge(self):
        from phenotypic.detect.nn import _sam3_detector

        assert _sam3_detector._merge_tiles_iou_nms is _merge_tiles_iou_nms
        assert _sam3_detector._iou is not None

    def test_fragment_survives_iou_nms(self):
        """Documents the bug centroid-in-core exists to fix.

        IoU(whole, fragment) equals the fragment's area fraction, so a
        fragment covering <= iou_thresh of its parent is never suppressed.
        """
        whole = np.zeros((20, 20), np.uint16)
        whole[5:15, 5:15] = 1  # 100 px
        frag = np.zeros((20, 20), np.uint16)
        frag[5:15, 5:9] = 1  # 40 px, IoU == 0.4 <= 0.5
        merged = _merge_tiles_iou_nms([whole, frag], iou_thresh=0.5)
        assert merged.max() == 2  # the fragment survives as a second instance


# ---------------------------------------------------------------------------
# Instance merge: centroid-in-core (Task 5)
# ---------------------------------------------------------------------------


class TestTileOverlapPx:
    def test_overlap_of_two_tiles(self):
        assert tile_overlap_px([_Tile(0, 0, 100, 100), _Tile(0, 80, 100, 180)]) == 20

    def test_single_tile_has_no_overlap(self):
        assert tile_overlap_px([_Tile(0, 0, 10, 10)]) == 0

    def test_abutting_tiles_do_not_overlap(self):
        assert tile_overlap_px([_Tile(0, 0, 10, 10), _Tile(0, 10, 10, 20)]) == 0

    def test_smallest_pairwise_overlap_wins(self):
        tiles = _plan_tiles((100, 180), tile_px=100, overlap=0.2)
        assert tile_overlap_px(tiles) == 20


class TestOwningTileIndex:
    def test_whole_and_fragment_resolve_to_the_same_tile(self):
        tiles = _plan_tiles((100, 180), tile_px=100, overlap=0.2)
        assert [(t.y0, t.x0, t.y1, t.x1) for t in tiles] == [
            (0, 0, 100, 100),
            (0, 80, 100, 180),
        ]
        # Whole colony centred at x=79.5 (tile 0 only contains it).
        assert owning_tile_index(tiles, (49.5, 79.5)) == 0
        # Its fragment in tile 1 is centred at x=84.5 — still nearer tile 0's
        # centre (x=50) than tile 1's (x=130), so tile 1 declines to claim it.
        assert owning_tile_index(tiles, (49.5, 84.5)) == 0

    def test_border_tile_core_reaches_the_image_edge(self):
        tiles = _plan_tiles((100, 180), tile_px=100, overlap=0.2)
        assert owning_tile_index(tiles, (2.0, 178.0)) == 1
        assert owning_tile_index(tiles, (2.0, 1.0)) == 0

    def test_single_tile_always_owns(self):
        assert owning_tile_index([_Tile(0, 0, 50, 50)], (25.0, 25.0)) == 0


class TestAssignByCentroidCore:
    def _two_tiles(self):
        # 100x180 image, two 100x100 tiles overlapping by 20 px.
        return [_Tile(0, 0, 100, 100), _Tile(0, 80, 100, 180)]

    def test_fragment_regression_one_colony_stays_one(self):
        """A colony fully inside tile A also appears as a fragment in tile B.

        Under ``_merge_tiles_iou_nms(iou_thresh=0.5)`` the fragment survives
        (IoU == area fraction) and paints OVER the colony. Centroid-in-core
        must yield exactly one instance with its area intact.
        """
        tiles = self._two_tiles()
        # Colony spans image cols 70..90 -> inside A (0..100); B (80..180)
        # sees only cols 80..90 as a fragment.
        a = np.zeros((100, 100), dtype=np.uint16)
        a[40:60, 70:90] = 1  # whole, tile-local
        b = np.zeros((100, 100), dtype=np.uint16)
        b[40:60, 0:10] = 1  # fragment, tile-local

        merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        labels = [lbl for lbl in np.unique(merged) if lbl]
        assert len(labels) == 1
        assert int((merged == labels[0]).sum()) == 20 * 20

    def test_instance_claimed_by_exactly_one_tile(self):
        tiles = self._two_tiles()
        # Colony wholly inside the overlap band, cols 82..88: both tiles see it
        # whole, so both emit an identical instance.
        a = np.zeros((100, 100), dtype=np.uint16)
        a[10:20, 82:88] = 1
        b = np.zeros((100, 100), dtype=np.uint16)
        b[10:20, 2:8] = 1
        merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        assert len([lbl for lbl in np.unique(merged) if lbl]) == 1

    def test_single_tile_relabels_contiguously(self):
        t = [_Tile(0, 0, 50, 50)]
        om = np.zeros((50, 50), dtype=np.uint16)
        om[5:10, 5:10] = 7
        om[20:25, 20:25] = 9
        merged = assign_by_centroid_core(t, [om], (50, 50))
        assert sorted(int(lbl) for lbl in np.unique(merged) if lbl) == [1, 2]

    def test_labels_are_assigned_largest_first(self):
        t = [_Tile(0, 0, 50, 50)]
        om = np.zeros((50, 50), dtype=np.uint16)
        om[0:2, 0:2] = 1  # 4 px, small
        om[10:20, 10:20] = 2  # 100 px, large
        merged = assign_by_centroid_core(t, [om], (50, 50))
        assert merged[15, 15] == 1  # largest gets label 1
        assert merged[0, 0] == 2

    def test_empty_input_returns_empty_objmap(self):
        tiles = self._two_tiles()
        blank = np.zeros((100, 100), dtype=np.uint16)
        merged = assign_by_centroid_core(tiles, [blank, blank.copy()], (100, 180))
        assert merged.shape == (100, 180)
        assert merged.dtype == np.uint16
        assert not merged.any()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="tiles"):
            assign_by_centroid_core(self._two_tiles(), [], (100, 180))

    def test_overlap_guard_warns_when_colony_exceeds_overlap(self):
        tiles = self._two_tiles()  # overlap_px == 20
        a = np.zeros((100, 100), dtype=np.uint16)
        a[10:70, 30:90] = 1  # d == 60 > 20
        b = np.zeros((100, 100), dtype=np.uint16)
        with pytest.warns(UserWarning, match="overlap"):
            merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        assert len([lbl for lbl in np.unique(merged) if lbl]) == 1  # not deleted

    def test_no_overlap_warning_when_instance_fits(self, recwarn):
        tiles = self._two_tiles()
        a = np.zeros((100, 100), dtype=np.uint16)
        a[40:50, 40:50] = 1  # d == 10 <= 20
        b = np.zeros((100, 100), dtype=np.uint16)
        assign_by_centroid_core(tiles, [a, b], (100, 180))
        assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]

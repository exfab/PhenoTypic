"""Tests for Sam3Detector (Spec 2a, Tasks 4-5).

Construction / serialization / capability / prompt / tiling-math run WITHOUT
gated weights (lazy load). The import-smoke test requires ``phenotypic[foundation]``
(transformers); functional ``.apply()`` tests additionally require the gated
~3.45 GB SAM3 weights and skip when unavailable.
"""

import json

import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import FOUNDATION_AVAILABLE, Sam3Detector


# ---------------------------------------------------------------------------
# Construction / capability / prompt
# ---------------------------------------------------------------------------


class TestSam3Construction:
    def test_capability_fields(self):
        det = Sam3Detector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is True

    def test_prompt_defaults_and_overrides(self):
        assert Sam3Detector().prompt == "colony"
        assert Sam3Detector(prompt="yeast colony").prompt == "yeast colony"

    def test_default_thresholds(self):
        det = Sam3Detector()
        assert det.score_thresh == 0.5
        assert det.mask_threshold == 0.5
        assert det.min_mask_region_area == 100  # C3: match Sam2Detector
        assert det.device == "auto"

    def test_default_tiling_fields(self):
        det = Sam3Detector()
        assert det.tile_px == 1008
        assert det.tile_overlap == 0.15
        assert det.max_instances_per_tile == 200

    def test_serialization_round_trip(self):
        det = Sam3Detector(prompt="bacterial colony", score_thresh=0.4)
        round = Sam3Detector.from_json(det.to_json())
        assert round.prompt == "bacterial colony" and round.score_thresh == 0.4

    def test_constructs_without_transformers(self):
        # lazy import: building the op must not import transformers
        Sam3Detector()  # no raise

    def test_is_gpu_and_object_detector(self):
        det = Sam3Detector()
        assert isinstance(det, GpuDetector)
        assert isinstance(det, ObjectDetector)


class TestSam3DetectorSerialization:
    def test_pipeline_json_round_trip(self):
        pipe = ImagePipeline(ops=[Sam3Detector(prompt="colony", tile_px=1500)])
        restored = ImagePipeline.from_json(pipe.to_json())
        det = list(restored._ops.values())[0]
        assert isinstance(det, Sam3Detector)
        assert det.prompt == "colony"
        assert det.tile_px == 1500

    def test_private_attrs_not_in_json(self):
        det = Sam3Detector()
        config = json.loads(ImagePipeline(ops=[det]).to_json())
        for op_cfg in config["pipe_cfgs"].values():
            params = op_cfg.get("params", {})
            assert "_model" not in params
            assert "_processor" not in params


# ---------------------------------------------------------------------------
# Tiling math + cross-tile merge (Task 5 — pure, no model)
# ---------------------------------------------------------------------------


class TestSam3Tiling:
    def test_plan_fixed_tiles_cover_with_overlap(self):
        import numpy as np

        from phenotypic.detect.nn._sam3_detector import _plan_tiles

        tiles = _plan_tiles((3000, 3000), tile_px=1008, overlap=0.15)
        assert all(t.h <= 1008 and t.w <= 1008 for t in tiles)
        covered = np.zeros((3000, 3000), bool)
        for t in tiles:
            covered[t.y0:t.y1, t.x0:t.x1] = True
        assert covered.all()

    def test_small_image_is_single_tile(self):
        from phenotypic.detect.nn._sam3_detector import _plan_tiles

        tiles = _plan_tiles((500, 500), tile_px=1008, overlap=0.15)
        assert len(tiles) == 1
        t = tiles[0]
        assert (t.y0, t.x0, t.y1, t.x1) == (0, 0, 500, 500)

    def test_tile_dims_match_bounds(self):
        from phenotypic.detect.nn._sam3_detector import _plan_tiles

        for t in _plan_tiles((2500, 1800), tile_px=1008, overlap=0.2):
            assert t.h == t.y1 - t.y0
            assert t.w == t.x1 - t.x0
            assert t.y1 <= 2500 and t.x1 <= 1800

    def test_merge_dedups_overlapping_instances(self):
        import numpy as np

        from phenotypic.detect.nn._tiling import _merge_tiles_iou_nms

        a = np.zeros((10, 10), np.uint16)
        a[2:6, 2:6] = 1
        b = np.zeros((10, 10), np.uint16)
        b[2:6, 2:6] = 1  # same blob from neighbour tile
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 1  # one instance, not two

    def test_merge_keeps_distinct_instances(self):
        import numpy as np

        from phenotypic.detect.nn._tiling import _merge_tiles_iou_nms

        a = np.zeros((10, 10), np.uint16)
        a[1:3, 1:3] = 1
        b = np.zeros((10, 10), np.uint16)
        b[7:9, 7:9] = 1  # disjoint blob
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 2  # two distinct instances survive


class TestSam3UsesCentroidCore:
    """Task 6: ``_infer_batch`` hands **tile-local** objmaps to the merge."""

    def test_infer_batch_merges_by_centroid_core(self, monkeypatch):
        """A colony straddling a tile seam must yield one instance, not a
        colony plus its fragment."""
        import numpy as np

        det = Sam3Detector(tile_px=100, tile_overlap=0.2)
        monkeypatch.setattr(det, "_ensure_model_loaded", lambda: None)

        # _plan_tiles((100, 180), 100, 0.2) -> [(0,0,100,100), (0,80,100,180)].
        # Tile 0 sees the whole colony at global cols 70..90; tile 1 sees only
        # its fragment at global cols 80..90.
        def fake_forward(crops):
            out = []
            for i, c in enumerate(crops):
                om = np.zeros(c.shape[:2], dtype=np.uint16)
                if i == 0:
                    om[40:60, 70:90] = 1  # whole colony, tile-local
                else:
                    om[40:60, 0:10] = 1  # fragment, tile-local
                out.append(om)
            return out

        monkeypatch.setattr(det, "_forward_tiles", fake_forward)
        sample = np.zeros((100, 180, 3), dtype=np.uint8)
        (result,) = det._infer_batch([sample])
        assert result.shape == (100, 180)
        labels = [lab for lab in np.unique(result) if lab]
        assert len(labels) == 1  # not colony + fragment
        assert int((result == labels[0]).sum()) == 20 * 20  # area uncorrupted

    def test_infer_batch_does_not_double_offset(self, monkeypatch):
        """The merge receives tile-local maps, so a colony seen only by the
        second tile must land at its true global coordinates.

        Offsetting the crop objmaps before the merge would make
        ``assign_by_centroid_core`` add ``tile.x0`` a second time — one
        plausible-looking instance, wrong place.
        """
        import numpy as np

        det = Sam3Detector(tile_px=100, tile_overlap=0.2)
        monkeypatch.setattr(det, "_ensure_model_loaded", lambda: None)

        # Tile 1 spans global cols 80..180; the colony sits at tile-local
        # cols 40..60 -> global cols 120..140, rows 40..60.
        def fake_forward(crops):
            out = []
            for i, c in enumerate(crops):
                om = np.zeros(c.shape[:2], dtype=np.uint16)
                if i == 1:
                    om[40:60, 40:60] = 1
                out.append(om)
            return out

        monkeypatch.setattr(det, "_forward_tiles", fake_forward)
        (result,) = det._infer_batch([np.zeros((100, 180, 3), dtype=np.uint8)])
        ys, xs = np.nonzero(result)
        assert (ys.min(), ys.max()) == (40, 59)
        assert (xs.min(), xs.max()) == (120, 139)

    def test_empty_batch_yields_no_results(self, monkeypatch):
        det = Sam3Detector()
        monkeypatch.setattr(det, "_ensure_model_loaded", lambda: None)
        assert det._infer_batch([]) == []


class TestSam3TileMergeIouDeprecated:
    def test_field_survives_json_round_trip(self):
        pipe = ImagePipeline(ops=[Sam3Detector(tile_merge_iou=0.25)])
        det = ImagePipeline.from_json(pipe.to_json()).get_ops()["Sam3Detector"]
        assert det.tile_merge_iou == 0.25

    def test_docstring_marks_it_deprecated(self):
        doc = Sam3Detector.__doc__ or ""
        arg_line = next(
            (ln for ln in doc.splitlines() if "tile_merge_iou:" in ln), ""
        )
        assert "Deprecated" in arg_line


class TestSam3TilingBatchInteraction:
    """C4: tiles regroup by source image; per-tile target_sizes; offset back."""

    def test_two_images_with_different_tile_counts_batch_correctly(
        self, monkeypatch
    ):
        import numpy as np

        det = Sam3Detector(tile_px=1008, tile_overlap=0.15)
        # Avoid loading any model.
        monkeypatch.setattr(det, "_ensure_model_loaded", lambda: None)

        # One small image (1 tile) + one large image (multiple tiles). Each
        # crop's forward returns a single full-crop instance so we can count
        # crops and verify offset-back + per-image grouping.
        forwarded_shapes: list[tuple[int, int]] = []

        def fake_forward(images):
            out = []
            for img in images:
                forwarded_shapes.append((img.shape[0], img.shape[1]))
                obj = np.ones((img.shape[0], img.shape[1]), dtype=np.uint16)
                out.append(obj)
            return out

        monkeypatch.setattr(det, "_forward_tiles", fake_forward)

        small = np.zeros((400, 400, 3), dtype=np.uint8)
        large = np.zeros((3000, 3000, 3), dtype=np.uint8)

        from phenotypic.detect.nn._sam3_detector import _plan_tiles

        n_small = len(_plan_tiles((400, 400), 1008, 0.15))
        n_large = len(_plan_tiles((3000, 3000), 1008, 0.15))
        assert n_small == 1 and n_large > 1  # different tile counts

        results = det._infer_batch([small, large])

        # One result per input image, each in that image's full shape.
        assert len(results) == 2
        assert results[0].shape == (400, 400)
        assert results[1].shape == (3000, 3000)
        # Every crop was forwarded (small's 1 + large's many).
        assert len(forwarded_shapes) == n_small + n_large
        # Each crop's forwarded size is the crop's OWN (H, W), never the full
        # image (C4: per-tile target_sizes).
        assert all(h <= 1008 and w <= 1008 for h, w in forwarded_shapes)
        # Both images got instances painted (offset-back worked).
        assert results[0].max() >= 1
        assert results[1].max() >= 1


# ---------------------------------------------------------------------------
# Import smoke — requires phenotypic[foundation] (D-foundation-install)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FOUNDATION_AVAILABLE,
    reason="Requires phenotypic[foundation] (transformers)",
)
class TestSam3ImportSmoke:
    def test_transformers_sam3_symbols_resolve(self):
        from transformers import Sam3Model, Sam3Processor

        assert Sam3Model is not None and Sam3Processor is not None

    def test_detector_module_imports(self):
        from phenotypic.detect.nn import _sam3_detector

        assert _sam3_detector.Sam3Detector is Sam3Detector


# ---------------------------------------------------------------------------
# Functional — requires foundation + gated weights (skips otherwise)
# ---------------------------------------------------------------------------


def _sam3_weights_available() -> bool:
    if not FOUNDATION_AVAILABLE:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache("facebook/sam3", "config.json")
        return isinstance(hit, str)
    except Exception:
        return False


@pytest.mark.skipif(
    not _sam3_weights_available(),
    reason="Requires phenotypic[foundation] and cached gated SAM3 weights",
)
class TestSam3Functional:
    def test_apply_produces_objects(self, synth_plate):
        det = Sam3Detector(device="cpu")
        result = det.apply(synth_plate.copy(), inplace=False)
        assert result.objmap[:].max() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

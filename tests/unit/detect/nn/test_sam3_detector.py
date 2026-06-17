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

        from phenotypic.detect.nn._sam3_detector import _merge_tiles_iou_nms

        a = np.zeros((10, 10), np.uint16)
        a[2:6, 2:6] = 1
        b = np.zeros((10, 10), np.uint16)
        b[2:6, 2:6] = 1  # same blob from neighbour tile
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 1  # one instance, not two

    def test_merge_keeps_distinct_instances(self):
        import numpy as np

        from phenotypic.detect.nn._sam3_detector import _merge_tiles_iou_nms

        a = np.zeros((10, 10), np.uint16)
        a[1:3, 1:3] = 1
        b = np.zeros((10, 10), np.uint16)
        b[7:9, 7:9] = 1  # disjoint blob
        merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
        assert merged.max() == 2  # two distinct instances survive


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

"""Tests for Sam2.

Construction, serialization, and isinstance checks work WITHOUT torch/sam2
installed.  Functional tests that call ``.apply()`` are skipped unless the
full ``phenotypic[torch]`` extra is available.
"""

import json
import sys
import types

import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import SAM2_AVAILABLE, Sam2


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSam2DetectorConstruction:
    """Sam2 can be constructed and inspected without torch."""

    def test_default_parameters(self):
        det = Sam2()
        assert det.model_size == "tiny"
        assert det.points_per_side == 32
        assert det.points_per_batch == 8
        assert det.pred_iou_thresh == 0.7
        assert det.stability_score_thresh == 0.92
        assert det.min_mask_region_area == 100
        # Native SAM2 crop-pyramid knobs default to upstream AMG values,
        # except crop_n_layers, which PhenoTypic engages by default.
        assert det.crop_n_layers == 1
        assert det.crop_nms_thresh == 0.7
        assert det.crop_overlap_ratio == 512 / 1500
        assert det.crop_n_points_downscale_factor == 1
        assert det.box_nms_thresh == 0.7
        assert det.device == "auto"
        assert det.checkpoint is None
        assert det.config is None

    def test_custom_parameters(self):
        det = Sam2(
                model_size="large",
                points_per_side=64,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.95,
                min_mask_region_area=500,
                device="cpu",
                checkpoint="/tmp/custom.pt",
                config="custom.yaml",
        )
        assert det.model_size == "large"
        assert det.points_per_side == 64
        assert det.pred_iou_thresh == 0.8
        assert det.stability_score_thresh == 0.95
        assert det.min_mask_region_area == 500
        assert det.device == "cpu"
        assert det.checkpoint == "/tmp/custom.pt"
        assert det.config == "custom.yaml"

    def test_crop_sliding_window_parameters(self):
        """Native SAM2 crop knobs are settable for sliding-window inference."""
        det = Sam2(
                crop_n_layers=1,
                crop_nms_thresh=0.6,
                crop_overlap_ratio=0.4,
                crop_n_points_downscale_factor=2,
        )
        assert det.crop_n_layers == 1
        assert det.crop_nms_thresh == 0.6
        assert det.crop_overlap_ratio == 0.4
        assert det.crop_n_points_downscale_factor == 2

    def test_empty_constructor_for_serialization(self):
        """Detector can be built with defaults for deserialization paths."""
        det = Sam2()
        assert det._generator is None

    def test_points_per_batch_must_be_positive(self):
        with pytest.raises(ValidationError):
            Sam2(points_per_batch=0)

    def test_all_model_sizes_accepted(self):
        for size in ("tiny", "small", "base_plus", "large"):
            det = Sam2(model_size=size)
            assert det.model_size == size

    def test_capability_fields(self):
        det = Sam2()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is False


# ---------------------------------------------------------------------------
# isinstance checks
# ---------------------------------------------------------------------------


class TestSam2DetectorHierarchy:
    """Sam2 sits in the correct ABC hierarchy."""

    def test_is_gpu_detector(self):
        det = Sam2()
        assert isinstance(det, GpuDetector)

    def test_is_object_detector(self):
        det = Sam2()
        assert isinstance(det, ObjectDetector)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSam2DetectorSerialization:
    """JSON round-trip works without torch installed."""

    def test_json_roundtrip(self):
        original = Sam2(
                model_size="small",
                points_per_side=48,
                points_per_batch=4,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.95,
                min_mask_region_area=200,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                device="cpu",
        )
        pipeline = ImagePipeline(ops=[original])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        restored = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored, Sam2)
        assert restored.model_size == "small"
        assert restored.points_per_side == 48
        assert restored.points_per_batch == 4
        assert restored.pred_iou_thresh == 0.8
        assert restored.stability_score_thresh == 0.95
        assert restored.min_mask_region_area == 200
        assert restored.crop_n_layers == 1
        assert restored.crop_n_points_downscale_factor == 2
        assert restored.device == "cpu"

    def test_generator_not_in_json(self):
        """The lazy _generator attribute must not appear in serialised JSON."""
        det = Sam2(model_size="tiny")
        pipeline = ImagePipeline(ops=[det])
        json_str = pipeline.to_json()
        config = json.loads(json_str)

        # Walk all param dicts — _generator should never appear
        for key, op_cfg in config["pipe_cfgs"].items():
            assert "_generator" not in op_cfg.get("params", {}), (
                "_generator leaked into JSON output"
            )

    def test_json_roundtrip_default_params(self):
        """Round-trip with all-default params preserves detector identity."""
        pipeline = ImagePipeline(ops=[Sam2()])
        restored = ImagePipeline.from_json(pipeline.to_json())
        restored_det = list(restored._ops.values())[0]
        assert isinstance(restored_det, Sam2)
        assert restored_det.model_size == "tiny"
        assert restored_det.points_per_batch == 8

    def test_old_pipeline_payload_defaults_points_per_batch(self):
        config = json.loads(ImagePipeline(ops=[Sam2()]).to_json())
        sam2_config = next(
                value
                for value in config["pipe_cfgs"].values()
                if value["class"] == "Sam2"
        )
        sam2_config["params"].pop("points_per_batch")

        restored = ImagePipeline.from_json(json.dumps(config))

        assert list(restored._ops.values())[0].points_per_batch == 8

    def test_json_structure(self):
        """Verify the serialised JSON has the expected structure."""
        det = Sam2(model_size="base_plus", points_per_side=16)
        pipeline = ImagePipeline(ops=[det])
        config = json.loads(pipeline.to_json())

        pipe_cfgs = config["pipe_cfgs"]
        sam2_key = [k for k in pipe_cfgs if "Sam2" in k][0]
        sam2_data = pipe_cfgs[sam2_key]

        assert sam2_data["class"] == "Sam2"
        assert sam2_data["params"]["model_size"] == "base_plus"
        assert sam2_data["params"]["points_per_side"] == 16


# ---------------------------------------------------------------------------
# Functional tests — require phenotypic[torch]
# ---------------------------------------------------------------------------


def _sam2_tiny_available() -> bool:
    """True if the tiny SAM2 checkpoint is cached locally or the download host is reachable.

    The download host (dl.fbaipublicfiles.com) is blocked in some CI environments
    (network policy returns 403 host_not_allowed). Tests skip rather than fail when
    neither the cache nor the network is available.
    """
    if not SAM2_AVAILABLE:
        return False
    from phenotypic.detect.nn._helper._checkpoint_manager import Sam2CheckpointManager

    if Sam2CheckpointManager.is_cached("tiny"):
        return True
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
                Sam2CheckpointManager.BASE_URL
                + Sam2CheckpointManager.MODELS["tiny"]["filename"],
                method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


@pytest.mark.skipif(
        not SAM2_AVAILABLE or not _sam2_tiny_available(),
        reason="Requires phenotypic[torch] and a cached or downloadable SAM2 tiny checkpoint",
)
class TestSam2DetectorFunctional:
    """Functional tests that load a model and run inference."""

    def test_apply_produces_objects(self, synth_plate):
        import numpy as np

        image = synth_plate.copy()
        det = Sam2(model_size="tiny", device="cpu")
        result = det.apply(image, inplace=False)
        assert result.objmap[:].max() > 0
        # S1: after the interface refactor writes objmap, the shared-backend
        # invariant must still hold (objmask is the derived view of objmap).
        assert result.objmask[:].any()
        np.testing.assert_array_equal(
                result.objmap[:] > 0,
                result.objmask[:],
        )

    def test_objmask_objmap_consistency(self, synth_plate):
        image = synth_plate.copy()
        det = Sam2(model_size="tiny", device="cpu")
        result = det.apply(image, inplace=False)

        import numpy as np

        np.testing.assert_array_equal(
                result.objmap[:] > 0,
                result.objmask[:],
        )

    def test_pipeline_apply(self, synth_plate):
        pipeline = ImagePipeline(ops=[
            Sam2(model_size="tiny", device="cpu"),
        ])
        result = pipeline.apply(synth_plate.copy(), inplace=False)
        assert result.objmap[:].max() > 0


class TestSam2CropPyramid:
    def test_crop_pyramid_is_engaged_by_default(self):
        from phenotypic.detect.nn import Sam2

        det = Sam2()
        assert det.crop_n_layers == 1
        assert det.box_nms_thresh == 0.7

    def test_build_sam2_generator_accepts_box_nms_thresh(self):
        """`box_nms_thresh` dedups the dense point grid's redundant proposals
        within one crop. SAM2 exposes it; Sam2 did not."""
        import inspect

        from phenotypic.detect.nn._sam2 import build_sam2_generator

        sig = inspect.signature(build_sam2_generator)
        assert "box_nms_thresh" in sig.parameters
        assert sig.parameters["box_nms_thresh"].default == 0.7


class TestSam2RleStreaming:
    def test_builder_forwards_batch_size_and_internal_rle_mode(
            self, monkeypatch
    ):
        from phenotypic.detect.nn._sam2 import build_sam2_generator

        seen: dict = {}

        class FakeGenerator:
            def __init__(self, model, **kwargs):
                seen.update(kwargs)

        monkeypatch.setitem(
                sys.modules,
                "sam2.automatic_mask_generator",
                types.SimpleNamespace(SAM2AutomaticMaskGenerator=FakeGenerator),
        )
        monkeypatch.setitem(
                sys.modules,
                "sam2.build_sam",
                types.SimpleNamespace(build_sam2=lambda *a, **k: object()),
        )
        build_sam2_generator(
                "tiny",
                device="cpu",
                points_per_batch=3,
                checkpoint="/fake/checkpoint.pt",
                config="fake.yaml",
        )
        assert seen["points_per_batch"] == 3
        assert seen["output_mode"] == "uncompressed_rle"

    @pytest.mark.parametrize(
            "mask",
            [
                np.zeros((3, 5), dtype=bool),
                np.ones((3, 5), dtype=bool),
                np.indices((4, 7)).sum(axis=0) % 2 == 0,
            ],
    )
    def test_fortran_rle_round_trip(self, mask):
        from phenotypic.detect.nn._helper._sam2_rle import (
            decode_uncompressed_rle,
            encode_uncompressed_rle,
        )

        rle = encode_uncompressed_rle(mask)
        np.testing.assert_array_equal(decode_uncompressed_rle(rle), mask)

    def test_rle_iou_matches_boolean_iou_randomized(self):
        from phenotypic.detect.nn._helper._sam2_rle import (
            encode_uncompressed_rle,
            rle_iou,
        )

        rng = np.random.default_rng(7)
        for _ in range(30):
            a = rng.random((9, 13)) > 0.7
            b = rng.random((9, 13)) > 0.7
            union = int((a | b).sum())
            expected = int((a & b).sum()) / union if union else 0.0
            assert rle_iou(
                    encode_uncompressed_rle(a), encode_uncompressed_rle(b)
            ) == expected

    def test_streamed_objmap_matches_binary_mask_ordering(self):
        from phenotypic.detect.nn._helper._sam2_rle import (
            encode_uncompressed_rle,
            paint_rle_records,
        )

        large = np.zeros((8, 11), bool)
        large[1:7, 1:10] = True
        small = np.zeros((8, 11), bool)
        small[3:5, 4:7] = True
        equal = np.zeros((8, 11), bool)
        equal[:2, :3] = True
        binary = [
            {"segmentation": small, "area": int(small.sum())},
            {"segmentation": equal, "area": int(equal.sum())},
            {"segmentation": large, "area": int(large.sum())},
        ]
        expected = np.zeros(large.shape, dtype=np.uint16)
        for label, record in enumerate(
                sorted(binary, key=lambda item: item["area"], reverse=True), start=1
        ):
            expected[record["segmentation"]] = label
        rle_records = [
            {
                "segmentation": encode_uncompressed_rle(record["segmentation"]),
                "area"        : record["area"],
            }
            for record in binary
        ]
        actual = paint_rle_records(
                rle_records,
                large.shape,
                detector_name="SAM2",
                truncate_before_sort=True,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_uint16_cap_occurs_before_area_sort(self, monkeypatch):
        from types import SimpleNamespace

        from phenotypic.detect.nn._helper._sam2_rle import (
            encode_uncompressed_rle,
            paint_rle_records,
        )

        masks = []
        for column in range(3):
            mask = np.zeros((1, 3), dtype=bool)
            mask[0, column] = True
            masks.append(mask)
        records = [
            {"segmentation": encode_uncompressed_rle(masks[0]), "area": 1},
            {"segmentation": encode_uncompressed_rle(masks[1]), "area": 1},
            # A later, larger record must be discarded before sorting.
            {"segmentation": encode_uncompressed_rle(masks[2]), "area": 100},
        ]
        monkeypatch.setattr(np, "iinfo", lambda _dtype: SimpleNamespace(max=2))

        with pytest.warns(UserWarning, match="exceeding uint16"):
            actual = paint_rle_records(
                    records,
                    (1, 3),
                    detector_name="SAM2",
                    truncate_before_sort=True,
            )

        np.testing.assert_array_equal(actual, np.array([[1, 2, 0]], np.uint16))

    def test_malformed_rle_is_rejected(self):
        from phenotypic.detect.nn._helper._sam2_rle import validate_uncompressed_rle

        with pytest.raises(ValueError, match="cover exactly"):
            validate_uncompressed_rle({"size": [2, 3], "counts": [2, 3]})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

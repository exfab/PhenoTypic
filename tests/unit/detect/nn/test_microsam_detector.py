"""Tests for MicroSamDetector.

Construction, serialization, and isinstance checks work WITHOUT torch/micro_sam
installed.  Functional tests that call ``.apply()`` are skipped unless the
full ``phenotypic[torch]`` extra is available.
"""

import json

import pytest

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.detect.nn import MICROSAM_AVAILABLE, MicroSamDetector


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMicroSamDetectorConstruction:
    """MicroSamDetector can be constructed and inspected without torch."""

    def test_default_parameters(self):
        det = MicroSamDetector()
        assert det.model_type == "vit_b_lm"
        assert det.device == "auto"

    def test_custom_parameters(self):
        det = MicroSamDetector(model_type="vit_l_lm", device="cpu")
        assert det.model_type == "vit_l_lm"
        assert det.device == "cpu"

    def test_empty_constructor_for_serialization(self):
        """Detector can be built with defaults for deserialization paths."""
        det = MicroSamDetector()
        assert det._predictor is None

    def test_all_model_types_accepted(self):
        for mt in (
            "vit_t", "vit_b", "vit_l", "vit_h",
            "vit_t_lm", "vit_b_lm", "vit_l_lm",
            "vit_b_em_organelles", "vit_l_em_organelles",
        ):
            det = MicroSamDetector(model_type=mt)
            assert det.model_type == mt


# ---------------------------------------------------------------------------
# isinstance checks
# ---------------------------------------------------------------------------


class TestMicroSamDetectorHierarchy:
    """MicroSamDetector sits in the correct ABC hierarchy."""

    def test_is_gpu_detector(self):
        det = MicroSamDetector()
        assert isinstance(det, GpuDetector)

    def test_is_object_detector(self):
        det = MicroSamDetector()
        assert isinstance(det, ObjectDetector)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMicroSamDetectorSerialization:
    """JSON round-trip works without torch installed."""

    def test_json_roundtrip(self):
        original = MicroSamDetector(model_type="vit_l_lm", device="cpu")
        pipeline = ImagePipeline(ops=[original])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        restored = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored, MicroSamDetector)
        assert restored.model_type == "vit_l_lm"
        assert restored.device == "cpu"

    def test_predictor_not_in_json(self):
        """The lazy _predictor attribute must not appear in serialised JSON."""
        det = MicroSamDetector()
        pipeline = ImagePipeline(ops=[det])
        json_str = pipeline.to_json()
        config = json.loads(json_str)

        for key, op_cfg in config["pipe_cfgs"].items():
            assert "_predictor" not in op_cfg.get("params", {}), (
                "_predictor leaked into JSON output"
            )

    def test_json_roundtrip_default_params(self):
        """Round-trip with all-default params preserves detector identity."""
        pipeline = ImagePipeline(ops=[MicroSamDetector()])
        restored = ImagePipeline.from_json(pipeline.to_json())
        restored_det = list(restored._ops.values())[0]
        assert isinstance(restored_det, MicroSamDetector)
        assert restored_det.model_type == "vit_b_lm"

    def test_json_structure(self):
        """Verify the serialised JSON has the expected structure."""
        det = MicroSamDetector(model_type="vit_t_lm", device="cpu")
        pipeline = ImagePipeline(ops=[det])
        config = json.loads(pipeline.to_json())

        pipe_cfgs = config["pipe_cfgs"]
        msam_key = [k for k in pipe_cfgs if "MicroSamDetector" in k][0]
        msam_data = pipe_cfgs[msam_key]

        assert msam_data["class"] == "MicroSamDetector"
        assert msam_data["params"]["model_type"] == "vit_t_lm"
        assert msam_data["params"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# Functional tests — require phenotypic[torch]
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not MICROSAM_AVAILABLE,
    reason="Requires phenotypic[torch] (micro_sam + torch)",
)
class TestMicroSamDetectorFunctional:
    """Functional tests that load a model and run inference."""

    def test_apply_produces_objects(self, synth_plate):
        image = synth_plate.copy()
        det = MicroSamDetector(model_type="vit_b_lm", device="cpu")
        result = det.apply(image, inplace=False)
        assert result.objmap[:].max() > 0
        assert result.objmask[:].any()

    def test_objmask_objmap_consistency(self, synth_plate):
        image = synth_plate.copy()
        det = MicroSamDetector(model_type="vit_b_lm", device="cpu")
        result = det.apply(image, inplace=False)

        import numpy as np

        np.testing.assert_array_equal(
            result.objmap[:] > 0,
            result.objmask[:],
        )

    def test_pipeline_apply(self, synth_plate):
        pipeline = ImagePipeline(ops=[
            MicroSamDetector(model_type="vit_b_lm", device="cpu"),
        ])
        result = pipeline.apply(synth_plate.copy(), inplace=False)
        assert result.objmap[:].max() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

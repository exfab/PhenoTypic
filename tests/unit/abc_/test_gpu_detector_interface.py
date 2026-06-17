"""Tests for the GpuDetector batched/streaming interface (Spec 1, Plan 1).

All tests construct detectors WITHOUT torch — the interface and the CPU
``_FakeGpuDetector`` exercise the engine contract with no GPU dependency.
"""

from typing import get_args

from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind


class TestTypingAliases:
    def test_input_layer_values(self):
        assert set(get_args(GpuInputLayer)) == {"rgb", "gray", "detect_mat"}

    def test_output_kind_values(self):
        assert set(get_args(GpuOutputKind)) == {"instance", "semantic"}


from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn import Sam2Detector


class TestCapabilityFields:
    def test_defaults_on_existing_detector(self):
        det = Sam2Detector()
        assert det.input_layer == "rgb"
        assert det.supports_batching is False
        assert det.output_kind == "instance"

    def test_fields_are_serializable_pydantic_fields(self):
        # capability markers are real fields (not ClassVar) -> in model_fields
        assert "input_layer" in GpuDetector.model_fields
        assert "supports_batching" in GpuDetector.model_fields
        assert "output_kind" in GpuDetector.model_fields


import numpy as np


class TestPreprocess:
    def test_2d_layer_stacked_to_3_channels(self):
        det = Sam2Detector()
        gray = np.zeros((4, 5), dtype=np.float32)
        out = det.preprocess(gray)
        assert out.shape == (4, 5, 3)

    def test_rgb_passthrough(self):
        det = Sam2Detector()
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        out = det.preprocess(rgb)
        assert out.shape == (4, 5, 3)
        assert out is rgb  # no copy for already-3-channel input

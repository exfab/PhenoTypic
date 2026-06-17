"""Tests for the GpuDetector batched/streaming interface (Spec 1, Plan 1).

All tests construct detectors WITHOUT torch — the interface and the CPU
``_FakeGpuDetector`` exercise the engine contract with no GPU dependency.
"""

from typing import get_args

import numpy as np

from phenotypic.abc_ import GpuDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect.nn import Sam2Detector
from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind
from tests._fakes.fake_gpu_detector import FakeGpuDetector as _FakeGpuDetector


class TestTypingAliases:
    def test_input_layer_values(self):
        assert set(get_args(GpuInputLayer)) == {"rgb", "gray", "detect_mat"}

    def test_output_kind_values(self):
        assert set(get_args(GpuOutputKind)) == {"instance", "semantic"}


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


class TestInferBatchDefault:
    def test_collate_passthrough(self):
        det = _FakeGpuDetector()
        samples = [np.zeros((2, 2, 3)), np.ones((2, 2, 3))]
        assert det.collate(samples) == samples

    def test_infer_batch_loops_infer_one(self):
        det = _FakeGpuDetector(output_kind="instance")
        a = np.zeros((3, 3, 3), dtype=np.float32)
        a[1, 1, :] = 1.0
        results = det.infer_batch([a, a])
        assert len(results) == 2
        assert results[0].dtype == np.uint16
        assert results[0].max() == 1  # one labeled blob

    def test_infer_batch_loads_model(self):
        det = _FakeGpuDetector()
        det.infer_batch([np.zeros((2, 2, 3))])
        assert det._loaded is True


class TestOperateRoutes:
    def test_instance_route_writes_objmap(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(output_kind="instance", threshold=0.3)
        out = det.apply(image, inplace=False)
        assert out.objmap[:].max() >= 1
        # objmask is the derived view of objmap
        np.testing.assert_array_equal(out.objmap[:] > 0, out.objmask[:])

    def test_semantic_route_writes_objmask(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(output_kind="semantic", threshold=0.3)
        out = det.apply(image, inplace=False)
        assert out.objmask[:].any()

    def test_input_layer_detect_mat_is_read_and_stacked(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(input_layer="detect_mat", output_kind="instance",
                               threshold=0.3)
        # detect_mat is 2D -> preprocess stacks to (H,W,3); must not raise
        out = det.apply(image, inplace=False)
        assert out.objmap[:].shape == image.shape[:2]

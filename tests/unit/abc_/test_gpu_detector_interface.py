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


from pydantic import PrivateAttr
from skimage.measure import label as _sk_label


class _FakeGpuDetector(GpuDetector):
    """CPU-only GpuDetector for interface tests (no torch).

    Thresholds the (stacked) input and either labels it (instance) or returns
    the binary mask (semantic). ``supports_batching``/``output_kind``/
    ``input_layer`` are overrideable per test.
    """

    threshold: float = 0.5
    _loaded: bool = PrivateAttr(default=False)

    def _ensure_model_loaded(self) -> None:
        self._loaded = True

    def _infer_one(self, sample):
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        peak = gray.max()
        mask = gray > (self.threshold * peak) if peak > 0 else gray > 0
        if self.output_kind == "instance":
            return _sk_label(mask).astype(np.uint16)
        return mask


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

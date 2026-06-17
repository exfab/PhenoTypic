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

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import CompositeEnhance, GaussianBlur, MedianFilter


def _three_branch_image() -> tuple[Image, np.ndarray, np.ndarray, np.ndarray]:
    """An image plus the three constant detect_mat planes used as branches.

    Each ``SetPlane`` enhancer below overwrites ``detect_mat`` with a flat
    value, so the combined result is a deterministic per-pixel reduction of
    the three constants -- ideal for asserting the exact mode arithmetic.
    """
    base = np.zeros((8, 8), dtype=float)
    image = Image(arr=base)
    return image, np.full((8, 8), 0.2), np.full((8, 8), 0.6), np.full((8, 8), 0.4)


class _SetPlane(GaussianBlur):
    """Test-only enhancer that floods ``detect_mat`` with a constant value."""

    value: float = 0.0

    def _operate(self, image):
        image.detect_mat[:] = np.full_like(image.detect_mat[:], self.value)
        return image


def _branches() -> list[_SetPlane]:
    return [_SetPlane(value=0.2), _SetPlane(value=0.6), _SetPlane(value=0.4)]


class TestCombinationModes:
    def test_max_is_default(self):
        image, *_ = _three_branch_image()
        result = CompositeEnhance(ops=_branches()).apply(image)
        assert np.allclose(result.detect_mat[:], 0.6)

    def test_min(self):
        image, *_ = _three_branch_image()
        result = CompositeEnhance(ops=_branches(), mode="min").apply(image)
        assert np.allclose(result.detect_mat[:], 0.2)

    def test_mean(self):
        image, *_ = _three_branch_image()
        result = CompositeEnhance(ops=_branches(), mode="mean").apply(image)
        assert np.allclose(result.detect_mat[:], (0.2 + 0.6 + 0.4) / 3)

    def test_median(self):
        image, *_ = _three_branch_image()
        result = CompositeEnhance(ops=_branches(), mode="median").apply(image)
        assert np.allclose(result.detect_mat[:], 0.4)


class TestClipping:
    def test_clip_off_by_default_allows_out_of_range(self):
        image = Image(arr=np.zeros((8, 8), dtype=float))
        result = CompositeEnhance(
            ops=[_SetPlane(value=1.5), _SetPlane(value=-0.3)],
        ).apply(image)
        assert result.detect_mat[:].max() > 1.0

    def test_clip_clamps_to_unit_interval(self):
        image = Image(arr=np.zeros((8, 8), dtype=float))
        result = CompositeEnhance(
            ops=[_SetPlane(value=1.5), _SetPlane(value=-0.3)],
            mode="min",
            clip=True,
        ).apply(image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


class TestBranchTypes:
    def test_nested_pipeline_branch(self):
        image, *_ = _three_branch_image()
        pipe = ImagePipeline(pipe_cfgs=[_SetPlane(value=0.6)])
        result = CompositeEnhance(
            ops=[_SetPlane(value=0.2), pipe],
        ).apply(image)
        assert np.allclose(result.detect_mat[:], 0.6)

    def test_none_slot_is_skipped(self):
        image, *_ = _three_branch_image()
        result = CompositeEnhance(
            ops=[_SetPlane(value=0.2), None, _SetPlane(value=0.6)],
        ).apply(image)
        assert np.allclose(result.detect_mat[:], 0.6)

    def test_empty_enhancers_raises(self):
        image = Image(arr=np.zeros((8, 8), dtype=float))
        with pytest.raises(Exception, match="At least one enhancer"):
            CompositeEnhance(ops=[]).apply(image)

    def test_all_none_enhancers_raises(self):
        image = Image(arr=np.zeros((8, 8), dtype=float))
        with pytest.raises(Exception, match="At least one enhancer"):
            CompositeEnhance(ops=[None, None]).apply(image)


class TestIntegrityAndDefaults:
    def test_rgb_and_gray_unchanged(self):
        from phenotypic.data import load_synth_yeast_plate

        image = load_synth_yeast_plate()
        rgb_before = image.rgb[:].copy()
        gray_before = image.gray[:].copy()
        result = CompositeEnhance(
            ops=[GaussianBlur(sigma=1.0), MedianFilter()],
        ).apply(image)
        assert np.array_equal(result.rgb[:], rgb_before)
        assert np.array_equal(result.gray[:], gray_before)

    def test_constructs_with_no_args(self):
        op = CompositeEnhance()
        assert op.mode == "max"
        assert op.clip is False
        assert len(op.ops) == 2
        assert isinstance(op.ops[0], GaussianBlur)
        assert isinstance(op.ops[1], MedianFilter)

    def test_explicit_none_enhancers_maps_to_default(self):
        op = CompositeEnhance(ops=None)
        assert isinstance(op.ops[0], GaussianBlur)
        assert isinstance(op.ops[1], MedianFilter)


class TestSerialization:
    def test_roundtrip_preserves_branch_subclasses_and_mode(self):
        op = CompositeEnhance(
            ops=[GaussianBlur(sigma=1.5), MedianFilter()],
            mode="mean",
            clip=True,
        )
        restored = CompositeEnhance.from_json(op.to_json())
        assert isinstance(restored, CompositeEnhance)
        assert restored.mode == "mean"
        assert restored.clip is True
        assert isinstance(restored.ops[0], GaussianBlur)
        assert restored.ops[0].sigma == 1.5
        assert isinstance(restored.ops[1], MedianFilter)

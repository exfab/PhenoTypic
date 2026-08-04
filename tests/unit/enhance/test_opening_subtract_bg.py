"""Tests for SubtractOpening.

Tests defaults, shape/dtype preservation, rgb/gray immutability, non-negative
output, mathematical correctness (tophat = src - opening), uniform image yields
zeros, small bright spot preserved, all shapes work, pipeline integration, and
serialization roundtrip.
"""

import cv2
import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import SubtractOpening


# -- Defaults ----------------------------------------------------------------


class TestDefaults:
    """Verify default parameter values."""

    def test_default_shape(self):
        op = SubtractOpening()
        assert op.shape == "disk"

    def test_default_width(self):
        op = SubtractOpening()
        assert op.width == 51

    def test_custom_values(self):
        op = SubtractOpening(shape="square", width=31)
        assert op.shape == "square"
        assert op.width == 31


# -- Shape / dtype preservation ----------------------------------------------


class TestOutputInvariants:
    """Output detect_mat has same shape and dtype; rgb/gray unchanged."""

    @pytest.fixture
    def gray_image(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        return Image(arr=arr)

    @pytest.fixture
    def rgb_image(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64, 3)).astype(np.float64)
        return Image(arr=arr)

    def test_shape_preserved(self, gray_image):
        op = SubtractOpening(width=11)
        result = op.apply(gray_image)
        assert result.detect_mat[:].shape == gray_image.detect_mat[:].shape

    def test_dtype_preserved(self, gray_image):
        original_dtype = gray_image.detect_mat[:].dtype
        op = SubtractOpening(width=11)
        result = op.apply(gray_image)
        assert result.detect_mat[:].dtype == original_dtype

    def test_rgb_immutability(self, rgb_image):
        original_rgb = rgb_image.rgb[:].copy()
        op = SubtractOpening(width=11)
        op.apply(rgb_image)
        np.testing.assert_array_equal(rgb_image.rgb[:], original_rgb)

    def test_gray_immutability(self, gray_image):
        original_gray = gray_image.gray[:].copy()
        op = SubtractOpening(width=11)
        op.apply(gray_image)
        np.testing.assert_array_equal(gray_image.gray[:], original_gray)


# -- Non-negative output -----------------------------------------------------


class TestNonNegativeOutput:
    """Top-hat result must be >= 0 everywhere."""

    def test_output_non_negative(self):
        rng = np.random.default_rng(99)
        arr = rng.random((64, 64)).astype(np.float64)
        image = Image(arr=arr)
        op = SubtractOpening(width=15)
        result = op.apply(image)
        assert result.detect_mat[:].min() >= 0.0


# -- Mathematical correctness -----------------------------------------------


class TestMathematicalCorrectness:
    """MORPH_TOPHAT == src - morphological opening."""

    def test_tophat_equals_src_minus_opening(self):
        rng = np.random.default_rng(7)
        arr = rng.random((80, 80)).astype(np.float64) * 0.8 + 0.1
        image = Image(arr=arr)

        op = SubtractOpening(shape="square", width=11)
        result = op.apply(image)

        # Compute expected manually: src - opening(src)
        kernel = op._make_footprint(shape="square", width=11)
        opened = cv2.morphologyEx(
            src=image.gray[:], op=cv2.MORPH_OPEN, kernel=kernel
        )
        expected = image.gray[:] - opened

        np.testing.assert_allclose(
            result.detect_mat[:], expected, atol=1e-10
        )


# -- Uniform image yields zeros ---------------------------------------------


class TestUniformImage:
    """Uniform image has no local structure: top-hat should be ~zero."""

    def test_uniform_yields_zeros(self):
        arr = np.full((64, 64), 0.5, dtype=np.float64)
        image = Image(arr=arr)
        op = SubtractOpening(width=11)
        result = op.apply(image)
        np.testing.assert_allclose(result.detect_mat[:], 0.0, atol=1e-10)


# -- Small bright spot preserved ---------------------------------------------


class TestSmallBrightSpot:
    """A small bright spot on dark background should survive top-hat."""

    def test_bright_spot_preserved(self):
        arr = np.zeros((64, 64), dtype=np.float64)
        arr[30:34, 30:34] = 1.0  # 4x4 bright patch
        image = Image(arr=arr)

        op = SubtractOpening(shape="square", width=11)
        result = op.apply(image)

        # The bright spot region should retain significant intensity
        spot = result.detect_mat[30:34, 30:34]
        assert spot.mean() > 0.5


# -- All shapes work ---------------------------------------------------------


class TestAllShapes:
    """Each supported shape produces valid output."""

    @pytest.mark.parametrize("shape", ["square", "diamond", "disk"])
    def test_shape_runs(self, shape):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64)
        image = Image(arr=arr)

        op = SubtractOpening(shape=shape, width=11)
        result = op.apply(image)

        assert result.detect_mat[:].shape == (64, 64)
        assert result.detect_mat[:].min() >= 0.0


# -- Pipeline integration ---------------------------------------------------


class TestPipelineIntegration:
    """SubtractOpening works inside an ImagePipeline."""

    def test_in_pipeline(self):
        from phenotypic.enhance import BlurGauss

        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        image = Image(arr=arr)

        pipeline = ImagePipeline(ops=[
            SubtractOpening(shape="disk", width=11),
            BlurGauss(sigma=1.0),
        ])
        result = pipeline.apply(image)
        assert result.detect_mat[:].shape == image.detect_mat[:].shape
        assert result.detect_mat[:].min() >= 0.0


# -- Serialization roundtrip ------------------------------------------------


class TestSerialization:
    """to_json / from_json preserves all parameters."""

    def test_roundtrip_preserves_params(self):
        pipeline = ImagePipeline(ops=[
            SubtractOpening(shape="diamond", width=31),
        ])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 1
        op = ops[0]
        assert isinstance(op, SubtractOpening)
        assert op.shape == "diamond"
        assert op.width == 31

    def test_default_params_roundtrip(self):
        pipeline = ImagePipeline(ops=[SubtractOpening()])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        op = list(loaded._ops.values())[0]
        assert op.shape == "disk"
        assert op.width == 51

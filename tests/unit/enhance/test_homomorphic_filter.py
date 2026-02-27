"""Tests for HomomorphicFilter and the module-level homomorphic_filter function.

Tests defaults, shape/dtype preservation, rgb/gray immutability, output clipped
to [0, 1], module-level function on raw arrays, uniform image behaviour,
mathematical correctness (log-domain decomposition), pipeline integration, and
serialization roundtrip.
"""

import cv2
import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance._homomorphic_filter import HomomorphicFilter, homomorphic_filter


# -- Defaults ----------------------------------------------------------------


class TestDefaults:
    """Verify default parameter values."""

    def test_default_sigma(self):
        op = HomomorphicFilter()
        assert op.sigma == 200.0

    def test_default_gamma_low(self):
        op = HomomorphicFilter()
        assert op.gamma_low == 0.5

    def test_default_gamma_high(self):
        op = HomomorphicFilter()
        assert op.gamma_high == 1.5

    def test_default_eps(self):
        op = HomomorphicFilter()
        assert op.eps == 1e-6

    def test_custom_values(self):
        op = HomomorphicFilter(sigma=100.0, gamma_low=0.3, gamma_high=2.0, eps=1e-8)
        assert op.sigma == 100.0
        assert op.gamma_low == 0.3
        assert op.gamma_high == 2.0
        assert op.eps == 1e-8


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
        op = HomomorphicFilter(sigma=10.0)
        result = op.apply(gray_image)
        assert result.detect_mat[:].shape == gray_image.detect_mat[:].shape

    def test_dtype_float(self, gray_image):
        op = HomomorphicFilter(sigma=10.0)
        result = op.apply(gray_image)
        assert np.issubdtype(result.detect_mat[:].dtype, np.floating)

    def test_rgb_immutability(self, rgb_image):
        original_rgb = rgb_image.rgb[:].copy()
        op = HomomorphicFilter(sigma=10.0)
        op.apply(rgb_image)
        np.testing.assert_array_equal(rgb_image.rgb[:], original_rgb)

    def test_gray_immutability(self, gray_image):
        original_gray = gray_image.gray[:].copy()
        op = HomomorphicFilter(sigma=10.0)
        op.apply(gray_image)
        np.testing.assert_array_equal(gray_image.gray[:], original_gray)


# -- Output clipped to [0, 1] -----------------------------------------------


class TestOutputRange:
    """Filtered result must be in [0, 1]."""

    def test_output_clipped(self):
        rng = np.random.default_rng(99)
        arr = rng.random((64, 64)).astype(np.float64)
        image = Image(arr=arr)
        op = HomomorphicFilter(sigma=10.0, gamma_low=0.3, gamma_high=2.0)
        result = op.apply(image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0

    def test_output_clipped_extreme_gains(self):
        rng = np.random.default_rng(7)
        arr = rng.random((64, 64)).astype(np.float64)
        image = Image(arr=arr)
        op = HomomorphicFilter(sigma=10.0, gamma_low=0.0, gamma_high=3.0)
        result = op.apply(image)
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


# -- Module-level function works on raw arrays -------------------------------


class TestModuleLevelFunction:
    """homomorphic_filter() works directly on numpy arrays."""

    def test_returns_ndarray(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float32)
        result = homomorphic_filter(arr, sigma=10.0)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(42)
        arr = rng.random((80, 100)).astype(np.float32)
        result = homomorphic_filter(arr, sigma=10.0)
        assert result.shape == arr.shape

    def test_output_range(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float32)
        result = homomorphic_filter(arr, sigma=10.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_accepts_float64(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64)
        result = homomorphic_filter(arr, sigma=10.0)
        assert result.shape == arr.shape


# -- Uniform image behaviour -------------------------------------------------


class TestUniformImage:
    """A uniform image has no reflectance structure; output should stay ~uniform."""

    def test_uniform_stays_approximately_uniform(self):
        arr = np.full((64, 64), 0.5, dtype=np.float64)
        result = homomorphic_filter(arr, sigma=10.0)
        # All output values should be very close to each other
        assert result.std() < 1e-6

    def test_uniform_image_via_class(self):
        arr = np.full((64, 64), 0.5, dtype=np.float64)
        image = Image(arr=arr)
        op = HomomorphicFilter(sigma=10.0)
        result = op.apply(image)
        assert result.detect_mat[:].std() < 1e-6


# -- Mathematical correctness -----------------------------------------------


class TestMathematicalCorrectness:
    """Verify the log-domain decomposition matches the expected formula."""

    def test_log_domain_decomposition(self):
        """Result matches manual step-by-step computation."""
        rng = np.random.default_rng(7)
        arr = rng.random((80, 80)).astype(np.float32) * 0.8 + 0.1

        sigma = 15.0
        gamma_low = 0.4
        gamma_high = 1.8
        eps = 1e-6

        result = homomorphic_filter(
            arr, sigma=sigma, gamma_low=gamma_low, gamma_high=gamma_high, eps=eps,
        )

        # Manual computation
        log_image = np.log(arr.astype(np.float32) + eps)
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        low_pass = cv2.GaussianBlur(
            log_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
        )
        high_pass = log_image - low_pass
        filtered_log = gamma_low * low_pass + gamma_high * high_pass
        expected = np.clip(np.exp(filtered_log) - eps, 0.0, 1.0)

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_identity_gains(self):
        """With gamma_low=1 and gamma_high=1 the filter is ~identity."""
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float32) * 0.8 + 0.1  # avoid clipping

        result = homomorphic_filter(
            arr, sigma=10.0, gamma_low=1.0, gamma_high=1.0,
        )
        # Should be very close to the original (only eps rounding)
        np.testing.assert_allclose(result, arr, atol=1e-4)

    def test_ksize_always_odd(self):
        """Kernel size must be odd for cv2.GaussianBlur."""
        # sigma that produces even ksize before correction
        for sigma in [1.0, 2.0, 3.5, 10.0, 50.0, 200.0]:
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            assert ksize % 2 == 1


# -- Pipeline integration ---------------------------------------------------


class TestPipelineIntegration:
    """HomomorphicFilter works inside an ImagePipeline."""

    def test_in_pipeline(self):
        from phenotypic.enhance import GaussianBlur

        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        image = Image(arr=arr)

        pipeline = ImagePipeline([
            HomomorphicFilter(sigma=10.0),
            GaussianBlur(sigma=1.0),
        ])
        result = pipeline.apply(image)
        assert result.detect_mat[:].shape == image.detect_mat[:].shape
        assert result.detect_mat[:].min() >= 0.0
        assert result.detect_mat[:].max() <= 1.0


# -- Serialization roundtrip ------------------------------------------------


class TestSerialization:
    """to_json / from_json preserves all parameters."""

    def test_roundtrip_preserves_params(self):
        pipeline = ImagePipeline([
            HomomorphicFilter(sigma=100.0, gamma_low=0.3, gamma_high=2.0, eps=1e-8),
        ])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 1
        op = ops[0]
        assert isinstance(op, HomomorphicFilter)
        assert op.sigma == 100.0
        assert op.gamma_low == 0.3
        assert op.gamma_high == 2.0
        assert op.eps == 1e-8

    def test_default_params_roundtrip(self):
        pipeline = ImagePipeline([HomomorphicFilter()])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        op = list(loaded._ops.values())[0]
        assert isinstance(op, HomomorphicFilter)
        assert op.sigma == 200.0
        assert op.gamma_low == 0.5
        assert op.gamma_high == 1.5
        assert op.eps == 1e-6

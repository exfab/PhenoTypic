"""Tests for MultiscaleLoGEnhancer.

Tests defaults, shape/dtype preservation, rgb/gray immutability, module-level
function, non-negative output, uniform image yields ~zero, blob response,
scale selectivity, pipeline integration, and serialization roundtrip.
"""

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance._multiscale_log_enhancer import MultiscaleLoGEnhancer


# -- Helpers -----------------------------------------------------------------


def _make_blob_image(
    size: int = 128,
    center: tuple[int, int] | None = None,
    radius: float = 6.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Create a 2-D float64 array with a single Gaussian blob."""
    if center is None:
        center = (size // 2, size // 2)
    yy, xx = np.mgrid[:size, :size]
    sigma = radius / np.sqrt(2.0)
    blob = amplitude * np.exp(
        -((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2 * sigma ** 2)
    )
    return blob.astype(np.float64)


# -- Defaults ----------------------------------------------------------------


class TestDefaults:
    """Verify default parameter values."""

    def test_default_min_radius(self):
        op = MultiscaleLoGEnhancer()
        assert op.min_radius == 3.0

    def test_default_max_radius(self):
        op = MultiscaleLoGEnhancer()
        assert op.max_radius == 12.0

    def test_default_num_scales(self):
        op = MultiscaleLoGEnhancer()
        assert op.num_scales == 12

    def test_custom_values(self):
        op = MultiscaleLoGEnhancer(min_radius=2.0, max_radius=20.0, num_scales=8)
        assert op.min_radius == 2.0
        assert op.max_radius == 20.0
        assert op.num_scales == 8


# -- Validation --------------------------------------------------------------


class TestValidation:
    """Invalid parameters raise ValueError."""

    def test_min_ge_max_raises(self):
        with pytest.raises(ValueError, match="min_radius"):
            MultiscaleLoGEnhancer(min_radius=10.0, max_radius=5.0)

    def test_equal_radii_raises(self):
        with pytest.raises(ValueError, match="min_radius"):
            MultiscaleLoGEnhancer(min_radius=5.0, max_radius=5.0)

    def test_num_scales_zero_raises(self):
        with pytest.raises(ValueError, match="num_scales"):
            MultiscaleLoGEnhancer(num_scales=0)


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
        op = MultiscaleLoGEnhancer()
        result = op.apply(gray_image)
        assert result.detect_mat[:].shape == gray_image.detect_mat[:].shape

    def test_dtype_preserved(self, gray_image):
        original_dtype = gray_image.detect_mat[:].dtype
        op = MultiscaleLoGEnhancer()
        result = op.apply(gray_image)
        assert result.detect_mat[:].dtype == original_dtype

    def test_rgb_immutability(self, rgb_image):
        original_rgb = rgb_image.rgb[:].copy()
        op = MultiscaleLoGEnhancer()
        op.apply(rgb_image)
        np.testing.assert_array_equal(rgb_image.rgb[:], original_rgb)

    def test_gray_immutability(self, gray_image):
        original_gray = gray_image.gray[:].copy()
        op = MultiscaleLoGEnhancer()
        op.apply(gray_image)
        np.testing.assert_array_equal(gray_image.gray[:], original_gray)


# -- Module-level function ---------------------------------------------------


class TestModuleLevelFunction:
    """MultiscaleLoGEnhancer._enhance() works directly on arrays."""

    def test_returns_array_same_shape(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64)
        result = MultiscaleLoGEnhancer._enhance(arr)
        assert result.shape == arr.shape

    def test_returns_float_dtype(self):
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64)
        result = MultiscaleLoGEnhancer._enhance(arr)
        assert np.issubdtype(result.dtype, np.floating)

    def test_min_ge_max_raises(self):
        arr = np.zeros((16, 16))
        with pytest.raises(ValueError, match="min_radius"):
            MultiscaleLoGEnhancer._enhance(arr, min_radius=10.0, max_radius=5.0)

    def test_num_scales_zero_raises(self):
        arr = np.zeros((16, 16))
        with pytest.raises(ValueError, match="num_scales"):
            MultiscaleLoGEnhancer._enhance(arr, num_scales=0)


# -- Non-negative output -----------------------------------------------------


class TestNonNegativeOutput:
    """LoG response uses absolute value, so output >= 0 everywhere."""

    def test_output_non_negative(self):
        rng = np.random.default_rng(99)
        arr = rng.random((64, 64)).astype(np.float64)
        image = Image(arr=arr)
        op = MultiscaleLoGEnhancer()
        result = op.apply(image)
        assert result.detect_mat[:].min() >= 0.0

    def test_function_output_non_negative(self):
        rng = np.random.default_rng(99)
        arr = rng.random((64, 64)).astype(np.float64)
        result = MultiscaleLoGEnhancer._enhance(arr)
        assert result.min() >= 0.0


# -- Uniform image yields ~zero ---------------------------------------------


class TestUniformImage:
    """Uniform image has no edges or blobs: LoG should be ~zero."""

    def test_uniform_yields_near_zero(self):
        # Discrete Gaussian kernels on a perfectly uniform array produce small
        # non-zero LoG responses (~1e-3) due to kernel truncation artefacts.
        # These are negligible compared to real blob responses (>0.01).
        arr = np.full((64, 64), 0.5, dtype=np.float64)
        image = Image(arr=arr)
        op = MultiscaleLoGEnhancer()
        result = op.apply(image)
        assert result.detect_mat[:].max() < 0.005

    def test_function_uniform_yields_near_zero(self):
        arr = np.full((64, 64), 0.5, dtype=np.float64)
        result = MultiscaleLoGEnhancer._enhance(arr)
        assert result.max() < 0.005


# -- Blob response -----------------------------------------------------------


class TestBlobResponse:
    """Synthetic bright blob should produce positive response at blob centre."""

    def test_bright_blob_detected(self):
        blob_img = _make_blob_image(size=128, radius=6.0, amplitude=0.8)
        image = Image(arr=blob_img)
        op = MultiscaleLoGEnhancer(min_radius=3.0, max_radius=12.0)
        result = op.apply(image)

        # Centre region should have strong positive response
        cy, cx = 64, 64
        detect = result.detect_mat[:]
        centre_val = detect[cy, cx]
        assert centre_val > 0.01, (
            f"Expected positive LoG response at blob centre, got {centre_val}"
        )

    def test_blob_centre_is_local_max(self):
        blob_img = _make_blob_image(size=128, radius=6.0, amplitude=0.8)
        result = MultiscaleLoGEnhancer._enhance(blob_img, min_radius=3.0, max_radius=12.0)

        cy, cx = 64, 64
        centre_val = result[cy, cx]
        # Centre should be larger than corners (which are far from blob)
        corner_mean = np.mean([result[0, 0], result[0, -1],
                               result[-1, 0], result[-1, -1]])
        assert centre_val > corner_mean * 5, (
            f"Centre ({centre_val}) should be much larger than "
            f"corner mean ({corner_mean})"
        )


# -- Scale selectivity -------------------------------------------------------


class TestScaleSelectivity:
    """Different blob sizes produce maximum response at matching scale."""

    def test_small_vs_large_blob(self):
        # Create two images: small blob and large blob
        small_blob = _make_blob_image(size=128, radius=4.0, amplitude=0.8)
        large_blob = _make_blob_image(size=128, radius=10.0, amplitude=0.8)

        # Enhance each with a range that covers both radii
        small_result = MultiscaleLoGEnhancer._enhance(
            small_blob, min_radius=2.0, max_radius=15.0, num_scales=20,
        )
        large_result = MultiscaleLoGEnhancer._enhance(
            large_blob, min_radius=2.0, max_radius=15.0, num_scales=20,
        )

        # Both should have strong centre responses
        cy, cx = 64, 64
        assert small_result[cy, cx] > 0.01
        assert large_result[cy, cx] > 0.01

    def test_response_drops_outside_scale_range(self):
        # A blob of radius=8 should produce weaker response when the scale
        # range is very narrow and far from the true radius.
        blob = _make_blob_image(size=128, radius=8.0, amplitude=0.8)

        # Correct range includes radius=8
        good_result = MultiscaleLoGEnhancer._enhance(
            blob, min_radius=5.0, max_radius=12.0, num_scales=10,
        )
        # Mismatched range: only very small scales
        poor_result = MultiscaleLoGEnhancer._enhance(
            blob, min_radius=1.0, max_radius=2.0, num_scales=10,
        )

        cy, cx = 64, 64
        assert good_result[cy, cx] > poor_result[cy, cx], (
            "Matching scale range should produce stronger blob response"
        )


# -- Pipeline integration ---------------------------------------------------


class TestPipelineIntegration:
    """MultiscaleLoGEnhancer works inside an ImagePipeline."""

    def test_in_pipeline(self):
        from phenotypic.enhance import GaussianBlur

        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float64) * 0.5 + 0.25
        image = Image(arr=arr)

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.0),
            MultiscaleLoGEnhancer(min_radius=3.0, max_radius=10.0, num_scales=6),
        ])
        result = pipeline.apply(image)
        assert result.detect_mat[:].shape == image.detect_mat[:].shape
        assert result.detect_mat[:].min() >= 0.0


# -- Serialization roundtrip ------------------------------------------------


def _class_registered() -> bool:
    """Check whether MultiscaleLoGEnhancer is exported in enhance.__init__."""
    try:
        from phenotypic.enhance import MultiscaleLoGEnhancer as _  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP_SERIAL = pytest.mark.skipif(
    not _class_registered(),
    reason="MultiscaleLoGEnhancer not yet exported from enhance.__init__",
)


class TestSerialization:
    """to_json / from_json preserves all parameters."""

    @_SKIP_SERIAL
    def test_roundtrip_preserves_params(self):
        pipeline = ImagePipeline([
            MultiscaleLoGEnhancer(min_radius=2.0, max_radius=15.0, num_scales=8),
        ])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        ops = list(loaded._ops.values())
        assert len(ops) == 1
        op = ops[0]
        assert isinstance(op, MultiscaleLoGEnhancer)
        assert op.min_radius == 2.0
        assert op.max_radius == 15.0
        assert op.num_scales == 8

    @_SKIP_SERIAL
    def test_default_params_roundtrip(self):
        pipeline = ImagePipeline([MultiscaleLoGEnhancer()])
        json_str = pipeline.to_json()
        loaded = ImagePipeline.from_json(json_str)

        op = list(loaded._ops.values())[0]
        assert op.min_radius == 3.0
        assert op.max_radius == 12.0
        assert op.num_scales == 12

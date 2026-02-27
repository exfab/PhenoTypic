"""Tests for InoculumDetector.

Covers default parameters, custom parameters, detection on synthetic data,
threshold methods, background subtraction, GMM toggle, GridImage support,
pipeline integration, serialization roundtrip, and detect_mat immutability.
"""

import pytest
import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import InoculumDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inoculum_image(size=200):
    """Create a synthetic grayscale image with bright Gaussian blobs.

    Returns a float64 array in [0, 1] suitable for ``Image()``.
    """
    rng = np.random.default_rng(42)
    arr = rng.random((size, size)).astype(np.float64) * 0.1 + 0.1  # dim bg

    yy, xx = np.mgrid[:size, :size]
    for cy, cx, r in [(50, 50, 15), (50, 150, 12), (150, 100, 18)]:
        blob = 0.7 * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / np.sqrt(2)) ** 2)
        )
        arr += blob

    return np.clip(arr, 0, 1)


def _make_test_image(size=200):
    """Wrap the synthetic array in an ``Image`` object."""
    arr = _make_inoculum_image(size)
    return Image((arr * 255).astype(np.uint8))


# Smaller parameter values for speed in tests
_FAST_PARAMS = dict(
    homomorphic_sigma=30.0,
    opening_width=5,
    log_min_radius=5.0,
    log_max_radius=20.0,
    log_num_scales=3,
    background_tophat_width=30,
    gmm_morph_open_radius=1,
    gmm_min_core_area=5,
)


# ---------------------------------------------------------------------------
# Default / custom parameter tests
# ---------------------------------------------------------------------------


class TestInoculumDetectorDefaults:
    """Verify that default parameter values match the documented spec."""

    def test_default_parameters(self):
        det = InoculumDetector()
        assert det.thresh_method == "otsu"
        assert det.subtract_background is False
        assert det.background_tophat_width == 300
        assert det.homomorphic_sigma == 300.0
        assert det.homomorphic_gamma_low == 0.5
        assert det.homomorphic_gamma_high == 1.5
        assert det.opening_shape == "disk"
        assert det.opening_width == 50
        assert det.log_min_radius == 25.0
        assert det.log_max_radius == 50.0
        assert det.log_num_scales == 5
        assert det.enable_gmm is True
        assert det.gmm_n_components == 2
        assert det.gmm_separation_threshold == 0.9
        assert det.gmm_min_core_area == 30
        assert det.gmm_morph_open_radius == 10
        assert det.gmm_morph_close_radius == 2

    def test_custom_parameters(self):
        det = InoculumDetector(
            thresh_method="triangle",
            subtract_background=True,
            background_tophat_width=100,
            homomorphic_sigma=150.0,
            homomorphic_gamma_low=0.3,
            homomorphic_gamma_high=2.0,
            opening_shape="square",
            opening_width=30,
            log_min_radius=10.0,
            log_max_radius=40.0,
            log_num_scales=8,
            enable_gmm=False,
            gmm_n_components=3,
            gmm_separation_threshold=0.5,
            gmm_min_core_area=50,
            gmm_morph_open_radius=5,
            gmm_morph_close_radius=3,
        )
        assert det.thresh_method == "triangle"
        assert det.subtract_background is True
        assert det.background_tophat_width == 100
        assert det.homomorphic_sigma == 150.0
        assert det.homomorphic_gamma_low == 0.3
        assert det.homomorphic_gamma_high == 2.0
        assert det.opening_shape == "square"
        assert det.opening_width == 30
        assert det.log_min_radius == 10.0
        assert det.log_max_radius == 40.0
        assert det.log_num_scales == 8
        assert det.enable_gmm is False
        assert det.gmm_n_components == 3
        assert det.gmm_separation_threshold == 0.5
        assert det.gmm_min_core_area == 50
        assert det.gmm_morph_open_radius == 5
        assert det.gmm_morph_close_radius == 3


# ---------------------------------------------------------------------------
# Detection on synthetic data
# ---------------------------------------------------------------------------


class TestInoculumDetectorDetection:
    """Core detection behaviour on synthetic images."""

    def test_detection_produces_nonzero_objmap(self):
        """Applying the detector should label at least one object."""
        image = _make_test_image()
        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        result = det.apply(image, inplace=False)

        assert result.objmap[:].max() > 0, "Expected at least one labelled object"

    def test_detection_produces_nonzero_objmask(self):
        """objmask should have True pixels after detection."""
        image = _make_test_image()
        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        result = det.apply(image, inplace=False)

        assert result.objmask[:].any(), "Expected non-empty objmask"

    def test_detection_with_gmm_enabled(self):
        """Detection with GMM core extraction should still find objects."""
        image = _make_test_image()
        det = InoculumDetector(enable_gmm=True, **_FAST_PARAMS)
        result = det.apply(image, inplace=False)

        assert result.objmap[:].max() > 0

    def test_detection_with_gmm_disabled(self):
        """Detection with enable_gmm=False skips GMM and still works."""
        image = _make_test_image()
        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        result = det.apply(image, inplace=False)

        assert result.objmap[:].max() > 0

    def test_detection_with_background_subtraction(self):
        """Detection with subtract_background=True runs without error."""
        image = _make_test_image()
        det = InoculumDetector(
            subtract_background=True, enable_gmm=False, **_FAST_PARAMS,
        )
        result = det.apply(image, inplace=False)

        assert result.objmask[:].shape == image.detect_mat[:].shape


# ---------------------------------------------------------------------------
# Threshold method parametrisation
# ---------------------------------------------------------------------------


class TestInoculumDetectorThreshMethods:
    """All supported thresholding methods should run without error."""

    @pytest.mark.parametrize(
        "method",
        ["otsu", "mean", "local", "triangle", "minimum", "isodata", "li"],
    )
    def test_thresh_method(self, method):
        image = _make_test_image()
        det = InoculumDetector(
            thresh_method=method, enable_gmm=False, **_FAST_PARAMS,
        )
        result = det.apply(image, inplace=False)

        assert result.objmask[:].shape == image.detect_mat[:].shape


# ---------------------------------------------------------------------------
# detect_mat immutability
# ---------------------------------------------------------------------------


class TestInoculumDetectorImmutability:
    """ObjectDetector must not modify detect_mat."""

    def test_detect_mat_unchanged(self):
        image = _make_test_image()
        original_detect_mat = image.detect_mat[:].copy()

        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        det.apply(image, inplace=True)

        np.testing.assert_array_equal(
            image.detect_mat[:],
            original_detect_mat,
            err_msg="detect_mat must not be modified by ObjectDetector",
        )


# ---------------------------------------------------------------------------
# GridImage support
# ---------------------------------------------------------------------------


class TestInoculumDetectorGridImage:
    """Detection on GridImage (synth yeast plate)."""

    def test_gridimage_detection(self):
        image = load_synth_yeast_plate()
        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        result = det.apply(image, inplace=False)

        assert result.objmap[:].max() > 0, "Expected objects on GridImage"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestInoculumDetectorPipeline:
    """InoculumDetector in an ImagePipeline."""

    def test_pipeline_apply(self):
        pipeline = ImagePipeline([
            InoculumDetector(enable_gmm=False, **_FAST_PARAMS),
        ])
        image = _make_test_image()
        result = pipeline.apply(image, inplace=False)

        assert result.objmask[:].shape == image.detect_mat[:].shape


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestInoculumDetectorSerialization:
    """JSON serialization and restoration."""

    def test_json_roundtrip(self):
        original = InoculumDetector(
            thresh_method="triangle",
            subtract_background=True,
            background_tophat_width=100,
            homomorphic_sigma=150.0,
            homomorphic_gamma_low=0.3,
            homomorphic_gamma_high=2.0,
            opening_shape="square",
            opening_width=30,
            log_min_radius=10.0,
            log_max_radius=40.0,
            log_num_scales=8,
            enable_gmm=False,
            gmm_n_components=3,
            gmm_separation_threshold=0.5,
            gmm_min_core_area=50,
            gmm_morph_open_radius=5,
            gmm_morph_close_radius=3,
        )
        pipeline = ImagePipeline([original])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        restored = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored, InoculumDetector)
        assert restored.thresh_method == original.thresh_method
        assert restored.subtract_background == original.subtract_background
        assert restored.background_tophat_width == original.background_tophat_width
        assert restored.homomorphic_sigma == original.homomorphic_sigma
        assert restored.homomorphic_gamma_low == original.homomorphic_gamma_low
        assert restored.homomorphic_gamma_high == original.homomorphic_gamma_high
        assert restored.opening_shape == original.opening_shape
        assert restored.opening_width == original.opening_width
        assert restored.log_min_radius == original.log_min_radius
        assert restored.log_max_radius == original.log_max_radius
        assert restored.log_num_scales == original.log_num_scales
        assert restored.enable_gmm == original.enable_gmm
        assert restored.gmm_n_components == original.gmm_n_components
        assert restored.gmm_separation_threshold == original.gmm_separation_threshold
        assert restored.gmm_min_core_area == original.gmm_min_core_area
        assert restored.gmm_morph_open_radius == original.gmm_morph_open_radius
        assert restored.gmm_morph_close_radius == original.gmm_morph_close_radius

    def test_serialization_functional_equivalence(self):
        """Serialized and restored detector produces identical results."""
        original = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)
        pipeline = ImagePipeline([original])
        restored_pipeline = ImagePipeline.from_json(pipeline.to_json())

        image1 = _make_test_image()
        image2 = _make_test_image()

        result1 = pipeline.apply(image1, inplace=False)
        result2 = restored_pipeline.apply(image2, inplace=False)

        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestInoculumDetectorReproducibility:
    """Deterministic output for identical inputs."""

    def test_deterministic(self):
        image1 = _make_test_image()
        image2 = _make_test_image()

        det = InoculumDetector(enable_gmm=False, **_FAST_PARAMS)

        result1 = det.apply(image1, inplace=False)
        result2 = det.apply(image2, inplace=False)

        np.testing.assert_array_equal(result1.objmask[:], result2.objmask[:])


# ---------------------------------------------------------------------------
# Opening shapes
# ---------------------------------------------------------------------------


class TestInoculumDetectorOpeningShapes:
    """All opening shapes should work without error."""

    @pytest.mark.parametrize("shape", ["square", "diamond", "disk"])
    def test_opening_shape(self, shape):
        image = _make_test_image()
        det = InoculumDetector(
            opening_shape=shape, enable_gmm=False, **_FAST_PARAMS,
        )
        result = det.apply(image, inplace=False)
        assert result.objmask[:].shape == image.detect_mat[:].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

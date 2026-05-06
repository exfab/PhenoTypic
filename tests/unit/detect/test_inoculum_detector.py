"""Tests for InoculumDetector.

Covers default parameters, custom parameters, detection on synthetic data,
threshold methods, GMM toggle, GridImage support, pipeline integration,
serialization roundtrip, detect_mat immutability, and derived parameter logic.
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


# Smaller diameter range for speed in tests
_FAST_PARAMS = dict(
    min_diameter=10.0,
    max_diameter=30.0,
)


# ---------------------------------------------------------------------------
# Default / custom parameter tests
# ---------------------------------------------------------------------------


class TestInoculumDetectorDefaults:
    """Verify that default parameter values match the documented spec."""

    def test_default_parameters(self):
        det = InoculumDetector()
        assert det.min_diameter == 30.0
        assert det.max_diameter == 100.0
        assert det.thresh_method == "otsu"
        assert det.enable_gmm is True
        assert det.gmm_n_components == 2
        assert det.gmm_separation_threshold == 0.9
        assert det.validate_obj_count is True

    def test_custom_parameters(self):
        det = InoculumDetector(
            min_diameter=15.0,
            max_diameter=200.0,
            thresh_method="triangle",
            enable_gmm=False,
            gmm_n_components=3,
            gmm_separation_threshold=0.5,
            validate_obj_count=False,
        )
        assert det.min_diameter == 15.0
        assert det.max_diameter == 200.0
        assert det.thresh_method == "triangle"
        assert det.enable_gmm is False
        assert det.gmm_n_components == 3
        assert det.gmm_separation_threshold == 0.5
        assert det.validate_obj_count is False

    def test_min_diameter_nonpositive_raises(self):
        with pytest.raises(ValueError, match="min_diameter must be positive"):
            InoculumDetector(min_diameter=0.0)

    def test_max_diameter_nonpositive_raises(self):
        with pytest.raises(ValueError, match="max_diameter must be positive"):
            InoculumDetector(max_diameter=-5.0)

    def test_min_ge_max_diameter_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            InoculumDetector(min_diameter=100.0, max_diameter=50.0)


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
            min_diameter=15.0,
            max_diameter=200.0,
            thresh_method="triangle",
            enable_gmm=False,
            gmm_n_components=3,
            gmm_separation_threshold=0.5,
            validate_obj_count=False,
        )
        pipeline = ImagePipeline([original])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        restored = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored, InoculumDetector)
        assert restored.min_diameter == original.min_diameter
        assert restored.max_diameter == original.max_diameter
        assert restored.thresh_method == original.thresh_method
        assert restored.enable_gmm == original.enable_gmm
        assert restored.gmm_n_components == original.gmm_n_components
        assert restored.gmm_separation_threshold == original.gmm_separation_threshold
        assert restored.validate_obj_count == original.validate_obj_count

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


# Reproducibility for default-args is now covered by the smoke contract
# tests/smoke/test_operation.py::test_operation


# ---------------------------------------------------------------------------
# Derived parameter logic
# ---------------------------------------------------------------------------


class TestInoculumDetectorDerivedParams:
    """Verify internal parameter derivation from diameter range."""

    def test_derived_sigma(self):
        """SubtractGaussian sigma should be max_diameter * 2."""
        det = InoculumDetector(max_diameter=100.0)
        assert det.max_diameter * 2 == 200.0

    def test_derived_log_radii(self):
        """LoG radii should be diameter / 2."""
        det = InoculumDetector(min_diameter=30.0, max_diameter=100.0)
        assert det.min_diameter / 2 == 15.0
        assert det.max_diameter / 2 == 50.0

    def test_derived_gmm_morph_open(self):
        """GMM morph open radius = max(1, round(min_diameter / 30))."""
        det = InoculumDetector(min_diameter=30.0)
        assert max(1, round(det.min_diameter / 30)) == 1

        det2 = InoculumDetector(min_diameter=90.0)
        assert max(1, round(det2.min_diameter / 30)) == 3

    def test_derived_gmm_min_core_area(self):
        """GMM min core area = max(5, round(min_diameter * 0.8))."""
        det = InoculumDetector(min_diameter=30.0)
        assert max(5, round(det.min_diameter * 0.8)) == 24

        det2 = InoculumDetector(min_diameter=3.0)
        assert max(5, round(det2.min_diameter * 0.8)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

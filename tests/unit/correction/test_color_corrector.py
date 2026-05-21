from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import colour
import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import ColorCheckerProfile, ColorCorrector
from phenotypic.correction._color_correction._diagnostic_dashboard import (
    PANEL_AVAILABLE,
)
from phenotypic.correction._color_correction._helpers import (
    compute_core_mask,
    trim_background_edges,
    validate_patch_shape,
)

if PANEL_AVAILABLE:
    import panel as pn

panel_required = pytest.mark.skipif(
    not PANEL_AVAILABLE, reason="Panel/param not installed"
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_synthetic_checker(
    n_patches: int = 24,
    noise_sigma: float = 0.01,
) -> tuple[np.ndarray, list[str]]:
    """Create synthetic checker patch colors based on real ColorChecker24 reference.

    Returns measured sRGB [0, 1] with small additive Gaussian noise and the
    corresponding patch names.
    """
    checker = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    ref_colors: list[np.ndarray] = []
    names: list[str] = []
    for name, xyY in list(checker.data.items())[:n_patches]:
        XYZ = colour.xyY_to_XYZ(xyY)
        rgb = colour.XYZ_to_sRGB(XYZ)
        ref_colors.append(np.clip(rgb, 0, 1))
        names.append(name)

    ref_rgb = np.array(ref_colors)
    rng = np.random.default_rng(42)
    measured_rgb = np.clip(
        ref_rgb + rng.normal(0, noise_sigma, ref_rgb.shape), 0, 1
    )
    return measured_rgb, names


@pytest.fixture()
def fitted_profile() -> ColorCheckerProfile:
    """Return a ColorCheckerProfile that has been fitted on synthetic data."""
    measured_rgb, patch_names = make_synthetic_checker()
    profile = ColorCheckerProfile(degree=2)
    profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
    return profile


@pytest.fixture()
def synthetic_rgb_image() -> Image:
    """Return a small synthetic RGB uint8 Image for correction tests."""
    rng = np.random.default_rng(99)
    arr = rng.integers(50, 200, (32, 32, 3), dtype=np.uint8)
    return Image(arr=arr)


# ===========================================================================
# TestColorCheckerProfileValidation
# ===========================================================================


class TestColorCheckerProfileValidation:
    """Test parameter validation and default values on ColorCheckerProfile."""

    def test_invalid_checker_type_raises(self):
        """Fitting with a nonexistent checker type raises KeyError."""
        profile = ColorCheckerProfile(checker_type="nonexistent_checker")
        measured_rgb, _ = make_synthetic_checker()
        with pytest.raises(KeyError):
            profile._fit_from_patch_colors(measured_rgb)

    def test_invalid_degree_zero_raises(self):
        """Constructing with degree=0 raises ValueError."""
        with pytest.raises(ValueError, match="degree must be"):
            ColorCheckerProfile(degree=0)

    def test_invalid_degree_five_raises(self):
        """Constructing with degree=5 raises ValueError."""
        with pytest.raises(ValueError, match="degree must be"):
            ColorCheckerProfile(degree=5)

    def test_valid_defaults(self):
        """Verify all default parameter values on a fresh profile."""
        profile = ColorCheckerProfile()
        assert profile.checker_type == "ColorChecker24 - After November 2014"
        assert profile.degree == 2
        assert profile.target_illuminant == "D65"
        assert profile.median_filter_size == 10
        assert profile.stddev_mag_threshold == 15.0
        assert profile.border_distance_threshold == 12.0
        assert profile.core_fraction == 0.5
        assert profile.ridge_lambda == 1e-3
        assert profile.outlier_sigma == 2.0
        assert profile.is_fitted is False


# ===========================================================================
# TestColorCheckerProfileFitting
# ===========================================================================


class TestColorCheckerProfileFitting:
    """Test profile fitting from synthetic patch data."""

    def test_fit_from_synthetic_patches(self):
        """Fitting with low-noise synthetic patches succeeds."""
        measured_rgb, patch_names = make_synthetic_checker(noise_sigma=0.01)
        profile = ColorCheckerProfile(degree=2)
        result = profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
        assert result.is_fitted is True

    def test_fit_produces_correction_matrix(self):
        """Correction matrix shape is (3, F) where F depends on degree."""
        for degree, expected_f in [(1, 3), (2, 6), (3, 13)]:
            measured_rgb, patch_names = make_synthetic_checker()
            profile = ColorCheckerProfile(degree=degree)
            profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
            matrix = profile.correction_matrix
            assert matrix.shape == (3, expected_f), (
                f"degree={degree}: expected (3, {expected_f}), got {matrix.shape}"
            )

    def test_diagnostics_populated(self, fitted_profile):
        """Verify all expected keys present in diagnostics dict."""
        diag = fitted_profile.diagnostics
        expected_keys = {
            "checker_type",
            "degree",
            "target_illuminant",
            "n_patches_detected",
            "n_patches_expected",
            "n_patches_rejected",
            "rejected_patches",
            "per_card_summary",
            "patches",
            "mean_deltaE00_before",
            "mean_deltaE00_after",
            "max_deltaE00_after",
            "median_deltaE00_after",
            "correction_matrix_condition_number",
            "warnings",
        }
        assert expected_keys.issubset(set(diag.keys())), (
            f"Missing keys: {expected_keys - set(diag.keys())}"
        )

    def test_unfitted_profile_correction_matrix_is_none(self):
        """An unfitted profile exposes ``correction_matrix`` as ``None``.

        Post-pydantic-migration ``correction_matrix`` is a defaulted model
        field rather than a ``RuntimeError``-raising property, so an
        unfitted profile still constructs and reports ``is_fitted=False``.
        """
        profile = ColorCheckerProfile()
        assert profile.correction_matrix is None
        assert profile.is_fitted is False


# ===========================================================================
# TestEdgeNoiseHandling
# ===========================================================================


class TestEdgeNoiseHandling:
    """Test helper functions for edge trimming, core masks, and validation."""

    def test_background_edge_trimming(self):
        """Uniform background edges are trimmed, reducing image size."""
        rng = np.random.default_rng(42)
        # Create an image with a uniform gray border (20px) and coloured center.
        h, w = 100, 100
        image = np.full((h, w, 3), 128, dtype=np.uint8)
        # Fill center with varied content so it is not trimmed.
        image[20:80, 20:80, :] = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
        trimmed = trim_background_edges(image, n_edge_pixels=10, variance_threshold=5.0)
        # The trimmed image should be smaller than the original.
        assert trimmed.shape[0] <= h
        assert trimmed.shape[1] <= w
        assert trimmed.size < image.size

    def test_core_sampling_excludes_boundaries(self):
        """compute_core_mask with fraction < 1 produces a smaller mask."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        core = compute_core_mask(mask, core_fraction=0.5)
        assert core.sum() < mask.sum()
        assert core.sum() > 0
        # All core pixels should be inside the original mask.
        assert np.all(core[~mask] == False)  # noqa: E712

    def test_patch_shape_validation_flags_irregular(self):
        """An elongated mask (aspect ratio > 3) is flagged invalid."""
        mask = np.zeros((100, 100), dtype=bool)
        # Elongated horizontal bar: 5 pixels tall, 80 pixels wide.
        mask[45:50, 10:90] = True
        is_valid, warnings = validate_patch_shape(mask)
        assert is_valid is False
        assert any("Aspect ratio" in w for w in warnings)

    def test_outlier_rejection(self):
        """A patch with a deliberately wrong color appears in rejected_patches."""
        measured_rgb, patch_names = make_synthetic_checker(noise_sigma=0.005)
        # Corrupt one patch drastically.
        measured_rgb[0] = [1.0, 0.0, 1.0]  # magenta -- far from 'dark skin'
        profile = ColorCheckerProfile(degree=2, outlier_sigma=1.5)
        profile._fit_from_patch_colors(measured_rgb, patch_names=patch_names)
        rejected = profile.diagnostics["rejected_patches"]
        assert len(rejected) > 0
        # The corrupted patch should be among the rejected.
        assert patch_names[0] in rejected


# ===========================================================================
# TestColorCorrectorOperation
# ===========================================================================


class TestColorCorrectorOperation:
    """Test ColorCorrector._operate modifies the image correctly."""

    def test_rgb_modified(self, fitted_profile, synthetic_rgb_image):
        """Applying the corrector changes the RGB data."""
        original_rgb = synthetic_rgb_image.rgb[:].copy()
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.apply(synthetic_rgb_image)
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_gray_recomputed(self, fitted_profile, synthetic_rgb_image):
        """Gray channel is recomputed from corrected RGB and differs from original."""
        original_gray = synthetic_rgb_image.gray[:].copy()
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.apply(synthetic_rgb_image)
        # Gray should have changed.
        assert not np.array_equal(result.gray[:], original_gray)
        # Gray should be in [0, 1] range (float luminance).
        assert result.gray[:].min() >= 0.0
        assert result.gray[:].max() <= 1.0
        # Gray should have the same spatial shape as RGB (sans channel dim).
        assert result.gray[:].shape == result.rgb[:].shape[:2]

    def test_detect_mat_recomputed(self, fitted_profile, synthetic_rgb_image):
        """detect_mat is updated from the corrected RGB."""
        original_dm = synthetic_rgb_image.detect_mat[:].copy()
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.apply(synthetic_rgb_image)
        assert not np.array_equal(result.detect_mat[:], original_dm)

    def test_shape_preserved(self, fitted_profile, synthetic_rgb_image):
        """Output shape matches input shape."""
        original_shape = synthetic_rgb_image.rgb[:].shape
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.apply(synthetic_rgb_image)
        assert result.rgb[:].shape == original_shape

    def test_output_dtype_preserved(self, fitted_profile, synthetic_rgb_image):
        """uint8 input produces uint8 output."""
        corrector = ColorCorrector(profile=fitted_profile)
        result = corrector.apply(synthetic_rgb_image)
        assert result.rgb[:].dtype == np.uint8


# ===========================================================================
# TestColorCorrectorSerialization
# ===========================================================================


class TestColorCorrectorSerialization:
    """Test JSON serialization round-trip for ColorCheckerProfile."""

    def test_correction_matrix_json_serializable(self, fitted_profile):
        """json.dumps on profile.to_dict() succeeds without errors."""
        data = fitted_profile.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_profile_to_dict_round_trip(self, fitted_profile):
        """to_dict -> from_dict recovers the correction matrix exactly."""
        data = fitted_profile.to_dict()
        restored = ColorCheckerProfile.from_dict(data)
        np.testing.assert_array_almost_equal(
            restored.correction_matrix,
            fitted_profile.correction_matrix,
        )
        assert restored.degree == fitted_profile.degree
        assert restored.checker_type == fitted_profile.checker_type
        assert restored.is_fitted is True


# ===========================================================================
# TestColorCorrectionDashboard
# ===========================================================================


@panel_required
class TestColorCorrectionDashboard:
    """Test interactive Panel diagnostic dashboard."""

    def test_profile_dashboard_returns_column(self, fitted_profile):
        """profile.dashboard(show=False) returns a pn.Column."""
        layout = fitted_profile.dashboard(show=False)
        assert isinstance(layout, pn.Column)

    def test_corrector_dashboard_delegates(self, fitted_profile):
        """ColorCorrector.dashboard() delegates to the profile."""
        corrector = ColorCorrector(profile=fitted_profile)
        layout = corrector.dashboard(show=False)
        assert isinstance(layout, pn.Column)

    def test_unfitted_profile_raises(self):
        """Unfitted profile raises RuntimeError."""
        profile = ColorCheckerProfile()
        with pytest.raises(RuntimeError, match="unfitted"):
            profile.dashboard(show=False)

    def test_delta_e_section_uses_diagnostics(self, fitted_profile):
        """Delta E section renders using diagnostics data."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        section = dashboard._delta_e_section()
        assert isinstance(section, pn.Card)

    def test_patches_section_renders(self, fitted_profile):
        """Matched patches section renders without error."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        section = dashboard._patches_section()
        assert isinstance(section, pn.Card)

    def test_pipeline_hidden_without_image(self, fitted_profile):
        """Pipeline section returns empty when no image provided."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        section = dashboard._pipeline_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_segmentation_hidden_without_image(self, fitted_profile):
        """Segmentation section returns empty when no image provided."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        section = dashboard._segmentation_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_show_delta_e_toggle(self, fitted_profile):
        """Toggling show_delta_e hides the section."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        dashboard.show_delta_e = False
        section = dashboard._delta_e_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_show_patches_toggle(self, fitted_profile):
        """Toggling show_patches hides the section."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        dashboard = ColorCorrectionDashboard(profile=fitted_profile)
        dashboard.show_patches = False
        section = dashboard._patches_section()
        assert isinstance(section, pn.Column)
        assert len(section) == 0

    def test_fit_without_rois_raises(self):
        """fit() raises ValueError when no ROIs provided."""
        from phenotypic import Image

        profile = ColorCheckerProfile()
        img = Image(arr=np.zeros((10, 10, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="No ROIs available"):
            profile.fit(img)

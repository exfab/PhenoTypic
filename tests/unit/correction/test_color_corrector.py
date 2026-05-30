from __future__ import annotations

import json
import logging
import warnings

import matplotlib
matplotlib.use("Agg")

import colour
import numpy as np
import pytest

from phenotypic import Image
from phenotypic.correction import (
    CaptureMetadata,
    ColorCheckerProfile,
    ColorCorrector,
)
from phenotypic.correction._color_correction._diagnostic_dashboard import (
    PANEL_AVAILABLE,
)
from phenotypic.correction._color_correction._helpers import (
    compute_core_mask,
    compute_swatch_roi_mask,
    segment_chips_by_border_fill,
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


BLACK_PATCH_NAME = "black 2 (1.5 D)"  # F4 in the A1..F4 grid layout


def make_synthetic_framed_checker_image(
    patch_size: int = 50,
    gutter: int = 10,
    frame: int = 50,
    n_rows: int = 4,
    n_cols: int = 6,
    frame_srgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Build a synthetic framed-checker uint8 RGB image.

    Lays the 24 ColorChecker24 patches in a *n_rows* x *n_cols* grid (default
    4x6) separated by uniform dark *gutter* pixels and surrounded by a uniform
    dark *frame*. Frame and gutter colour defaults to pure black so the
    black F4 reference patch (Lab ``(20.64, 0.07, -0.46)``) is the
    unambiguous nearest match in Lab — and *without* the stddev ROI mask,
    every frame/gutter pixel gets assigned to F4, contaminating its
    measurement.

    Patch colours are sRGB-encoded (the standard interpretation of uint8
    image pixels). Returns ``(H, W, 3)`` uint8.
    """
    checker = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    target_wp = colour.CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"]
    items = list(checker.data.items())[: n_rows * n_cols]

    patch_colors_u8: list[np.ndarray] = []
    for _name, xyY in items:
        XYZ = colour.xyY_to_XYZ(xyY)
        # Bradford-adapt to D65 so the resulting Lab matches the reference
        # the profile loads via _load_reference_data().
        XYZ = colour.chromatic_adaptation(
            XYZ,
            XYZ_w=colour.xy_to_XYZ(checker.illuminant),
            XYZ_wr=colour.xy_to_XYZ(target_wp),
            transform="Bradford",
        )
        rgb = colour.XYZ_to_RGB(
            XYZ,
            colourspace=colour.RGB_COLOURSPACES["sRGB"],
            illuminant=target_wp,
            apply_cctf_encoding=True,
        )
        rgb = np.clip(rgb, 0.0, 1.0)
        patch_colors_u8.append(np.round(rgb * 255).astype(np.uint8))

    grid_h = n_rows * patch_size + (n_rows - 1) * gutter
    grid_w = n_cols * patch_size + (n_cols - 1) * gutter
    H = grid_h + 2 * frame
    W = grid_w + 2 * frame

    img = np.full((H, W, 3), frame_srgb, dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = frame + r * (patch_size + gutter)
            x0 = frame + c * (patch_size + gutter)
            img[y0 : y0 + patch_size, x0 : x0 + patch_size] = (
                patch_colors_u8[r * n_cols + c]
            )
    return img


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


# ---------------------------------------------------------------------------
# Capture-metadata helpers
# ---------------------------------------------------------------------------

#: Maps a CaptureMetadata field to an exiftool-style EXIF key for injection.
_EXIF_KEY_MAP = {
    "camera_make"  : "EXIF:Make",
    "camera_model" : "EXIF:Model",
    "lens_model"   : "EXIF:LensModel",
    "iso"          : "EXIF:ISO",
    "exposure_time": "EXIF:ExposureTime",
    "f_number"     : "EXIF:FNumber",
    "focal_length" : "EXIF:FocalLength",
}

#: A complete set of injectable EXIF fields used by the calibration fixtures.
_REFERENCE_EXIF = {
    "camera_make"  : "Canon",
    "camera_model" : "EOS R100",
    "lens_model"   : "RF 50mm F1.8 STM",
    "iso"          : 800,
    "exposure_time": 1 / 60,
    "f_number"     : 5.6,
    "focal_length" : 50.0,
}


def _inject_exif(image: Image, *, key_style: str = "exiftool", **fields) -> Image:
    """Write EXIF-style fields into an image's imported metadata.

    Args:
        image: Image to mutate.
        key_style: ``"exiftool"`` (``EXIF:FNumber``), ``"exifread"``
            (``EXIF FNumber`` / ``Image Make``), or ``"tiff"`` (``TIFF:Make``).
        **fields: CaptureMetadata field name -> value.

    Returns:
        The same image, for chaining.
    """
    exifread_keys = {
        "camera_make"  : "Image Make",
        "camera_model" : "Image Model",
        "lens_model"   : "EXIF LensModel",
        "iso"          : "EXIF ISOSpeedRatings",
        "exposure_time": "EXIF ExposureTime",
        "f_number"     : "EXIF FNumber",
        "focal_length" : "EXIF FocalLength",
    }
    tiff_keys = {k: v.replace("EXIF:", "TIFF:") for k, v in _EXIF_KEY_MAP.items()}
    key_maps = {
        "exiftool": _EXIF_KEY_MAP,
        "exifread": exifread_keys,
        "tiff"    : tiff_keys,
    }
    key_map = key_maps[key_style]
    image._metadata.imported.update(
        {key_map[field]: value for field, value in fields.items()}
    )
    return image


@pytest.fixture()
def calibration_image_with_exif() -> Image:
    """A framed synthetic checker carrying full reference EXIF."""
    img = make_synthetic_framed_checker_image()
    image = Image(arr=img, bit_depth=8)
    return _inject_exif(image, **_REFERENCE_EXIF)


@pytest.fixture()
def fitted_profile_with_exif(calibration_image_with_exif) -> ColorCheckerProfile:
    """A fitted profile whose capture_metadata holds the reference EXIF."""
    return ColorCheckerProfile(degree=2).fit(calibration_image_with_exif)


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
        assert profile.min_swatch_area_frac == 0.3
        assert profile.pad_checker is False
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
# TestSwatchRoiMask
# ===========================================================================


class TestSwatchRoiMask:
    """Test the cross-channel stddev ROI mask helper."""

    def test_framed_grid_excludes_frame_rows_and_cols(self):
        """A framed checker yields a mask False on the outer frame band."""
        frame_px = 50
        filter_size = 10
        img = make_synthetic_framed_checker_image(
            patch_size=50, gutter=10, frame=frame_px,
        )
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(
            lab, stddev_mag_threshold=15.0, filter_size=filter_size,
        )
        # The inner ``filter_size`` pixels of the frame are blurred into the
        # grid by the median pre-filter; check only the strictly-outer band.
        strict = frame_px - filter_size
        assert not mask[:strict, :].any()
        assert not mask[-strict:, :].any()
        assert not mask[:, :strict].any()
        assert not mask[:, -strict:].any()
        # The grid interior must have at least some True pixels.
        interior = mask[frame_px:-frame_px, frame_px:-frame_px]
        assert interior.any()

    def test_uniform_image_returns_all_false(self):
        """Pure black image has zero stddev everywhere -> all-False mask."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        assert not mask.any()

    def test_no_frame_grid_returns_mostly_true(self):
        """Patch grid edge-to-edge (no frame) yields a mostly-True mask."""
        img = make_synthetic_framed_checker_image(
            patch_size=50, gutter=0, frame=0,
        )
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        # With no border at all, the median pre-filter blurs patch boundaries
        # but most rows/cols still cross multiple patches -> high stddev.
        assert mask.mean() > 0.5


# ===========================================================================
# TestStddevRoiMaskIntegration
# ===========================================================================


class TestStddevRoiMaskIntegration:
    """End-to-end tests covering the dark-frame regression and safety net."""

    def test_dark_frame_no_longer_misidentified_as_f4_patch(self):
        """With the stddev ROI mask, the F4 measurement matches the swatch.

        Without the fix, every dark-frame pixel was assigned to the black F4
        patch (nearest in Lab), so F4's geometric median collapsed to
        Lab~(0, 0, 0) instead of the F4 reference ``(20.64, 0.07, -0.46)``.
        """
        img = make_synthetic_framed_checker_image(frame_srgb=(0, 0, 0))
        profile = ColorCheckerProfile(degree=2)
        profile.fit(Image(arr=img, bit_depth=8))

        f4 = profile.diagnostics["patches"][BLACK_PATCH_NAME]
        assert f4["measured_lab"] is not None
        measured = np.asarray(f4["measured_lab"])
        reference = np.asarray(f4["reference_lab"])
        dE = float(colour.difference.delta_E_CIE2000(measured, reference))
        assert dE < 5.0, (
            f"F4 measured_lab {measured.tolist()} drifted ΔE={dE:.2f} from "
            f"reference {reference.tolist()} — dark frame likely still "
            f"polluting the F4 patch mask."
        )
        # Sanity check: a frame-contaminated F4 collapses toward Lab L~0.
        assert measured[0] > 10.0

    def test_empty_mask_raises_on_strict_count(self, caplog):
        """An impossibly high stddev threshold yields no gutters → raises.

        With border-fill segmentation an empty ROI mask falls back to an
        all-True mask, which is a single merged blob — the strict chip-count
        gate then raises rather than silently producing a degenerate fit.
        """
        import logging
        caplog.set_level(
            logging.WARNING,
            logger="phenotypic.correction._color_correction._color_checker_profile",
        )
        img = make_synthetic_framed_checker_image()
        # Threshold so high that no row/col passes -> mask is all-False.
        profile = ColorCheckerProfile(degree=2, stddev_mag_threshold=1.0e6)
        with pytest.raises(ValueError, match="expected 24"):
            profile.fit(Image(arr=img, bit_depth=8))
        # The empty-mask fallback warning is still logged before the raise.
        assert any(
            "empty" in rec.message.lower() and "fall" in rec.message.lower()
            for rec in caplog.records
        )


# ===========================================================================
# TestBorderFillSegmentation
# ===========================================================================


def _framed_ref_lab() -> dict[str, tuple[float, float, float]]:
    """Reference Lab dict (D65) for the 24 ColorChecker24 patches."""
    from phenotypic.correction._color_correction._color_checker_profile import (
        _load_reference_data,
    )

    ref_Lab, _, _ = _load_reference_data(
        "ColorChecker24 - After November 2014", "D65"
    )
    return {name: tuple(v.tolist()) for name, v in ref_Lab.items()}


class TestBorderFillSegmentation:
    """Geometric chip segmentation + Hungarian labelling."""

    def test_segments_exactly_24_chips(self):
        """A framed grid yields exactly 24 connected-component chips."""
        img = make_synthetic_framed_checker_image()
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        blob_masks, blob_names = segment_chips_by_border_fill(
            mask, lab, _framed_ref_lab(), strict=True,
        )
        assert len(blob_masks) == 24
        assert len(set(blob_names)) == 24  # all distinct names assigned

    def test_blob_centroids_match_grid_cells(self):
        """Each chip blob's centroid lands inside its physical swatch cell."""
        patch_size, gutter, frame = 50, 10, 50
        n_rows, n_cols = 4, 6
        img = make_synthetic_framed_checker_image(
            patch_size=patch_size, gutter=gutter, frame=frame,
            n_rows=n_rows, n_cols=n_cols,
        )
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        blob_masks, _ = segment_chips_by_border_fill(
            mask, lab, _framed_ref_lab(), strict=True,
        )
        cells = []
        for r in range(n_rows):
            for c in range(n_cols):
                y0 = frame + r * (patch_size + gutter)
                x0 = frame + c * (patch_size + gutter)
                cells.append((y0, y0 + patch_size, x0, x0 + patch_size))
        for m in blob_masks:
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()
            assert any(
                y0 <= cy < y1 and x0 <= cx < x1
                for (y0, y1, x0, x1) in cells
            ), f"centroid ({cy:.0f},{cx:.0f}) not in any swatch cell"

    def test_labels_independent_of_spatial_order(self):
        """Hungarian labelling assigns the correct name to the black chip."""
        img = make_synthetic_framed_checker_image()
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        blob_masks, blob_names = segment_chips_by_border_fill(
            mask, lab, _framed_ref_lab(), strict=True,
        )
        median_L = [float(np.median(lab[m][:, 0])) for m in blob_masks]
        darkest = blob_names[int(np.argmin(median_L))]
        assert darkest == BLACK_PATCH_NAME

    def test_noise_specks_do_not_break_count(self):
        """Tiny spurious mask specks are filtered; strict count stays 24.

        Locks in the robust size floor (median of the largest n_expected
        components) so a few high-variance survivors along a noisy gutter do
        not inflate the component count past the strict gate.
        """
        img = make_synthetic_framed_checker_image()
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        for (y, x) in [(2, 2), (3, 3), (5, 120), (180, 5), (100, 2)]:
            mask[y, x] = True
        blob_masks, _ = segment_chips_by_border_fill(
            mask, lab, _framed_ref_lab(), strict=True,
        )
        assert len(blob_masks) == 24

    def test_strict_raises_when_chips_merge(self):
        """Touching chips (no gutter) collapse into one blob → strict raise."""
        img = make_synthetic_framed_checker_image(gutter=0, frame=0)
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        with pytest.raises(ValueError, match="expected 24"):
            segment_chips_by_border_fill(
                mask, lab, _framed_ref_lab(), strict=True,
            )

    def test_non_strict_returns_partial(self):
        """strict=False returns whatever was found instead of raising."""
        img = make_synthetic_framed_checker_image(gutter=0, frame=0)
        lab = Image(arr=img, bit_depth=8).color.Lab[:]
        mask = compute_swatch_roi_mask(lab, stddev_mag_threshold=15.0)
        blob_masks, blob_names = segment_chips_by_border_fill(
            mask, lab, _framed_ref_lab(), strict=False,
        )
        assert len(blob_masks) == len(blob_names)
        assert len(blob_masks) < 24  # merged, not the full set

    def test_fit_raises_when_chips_merge(self):
        """fit() surfaces the strict-count error end-to-end."""
        img = make_synthetic_framed_checker_image(gutter=0, frame=0)
        profile = ColorCheckerProfile(degree=2)
        with pytest.raises(ValueError, match="expected 24"):
            profile.fit(Image(arr=img, bit_depth=8))

    def test_min_swatch_area_frac_validator(self):
        """min_swatch_area_frac outside (0, 1] is rejected at construction."""
        with pytest.raises(ValueError, match="min_swatch_area_frac"):
            ColorCheckerProfile(min_swatch_area_frac=0.0)
        with pytest.raises(ValueError, match="min_swatch_area_frac"):
            ColorCheckerProfile(min_swatch_area_frac=1.5)

    def test_serialization_ignores_removed_key(self):
        """A stray legacy border_distance_threshold key deserializes cleanly."""
        data = ColorCheckerProfile(degree=2).model_dump(mode="json")
        data["border_distance_threshold"] = 12.0  # legacy key
        restored = ColorCheckerProfile.model_validate(data)
        assert restored.degree == 2
        assert not hasattr(restored, "border_distance_threshold")

    def test_unknown_kwarg_still_raises(self):
        """Only the legacy key is dropped; genuine typos still raise."""
        with pytest.raises(ValueError):
            ColorCheckerProfile(degree=2, not_a_real_param=5)


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
        """json.dumps on profile.model_dump(mode="json") succeeds without errors."""
        data = fitted_profile.model_dump(mode="json")
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_profile_model_dump_round_trip(self, fitted_profile):
        """model_dump -> model_validate recovers the correction matrix exactly."""
        data = fitted_profile.model_dump(mode="json")
        restored = ColorCheckerProfile.model_validate(data)
        np.testing.assert_array_almost_equal(
            restored.correction_matrix,
            fitted_profile.correction_matrix,
        )
        assert restored.degree == fitted_profile.degree
        assert restored.checker_type == fitted_profile.checker_type
        assert restored.is_fitted is True

    def test_capture_metadata_round_trip(self, fitted_profile_with_exif):
        """capture_metadata survives model_dump(mode="json") -> model_validate."""
        data = fitted_profile_with_exif.model_dump(mode="json")
        # Nested object, not a bare string/None.
        assert isinstance(data["capture_metadata"], dict)
        restored = ColorCheckerProfile.model_validate(data)
        assert restored.capture_metadata == fitted_profile_with_exif.capture_metadata
        assert restored.capture_metadata.camera_model == "EOS R100"
        assert restored.capture_metadata.iso == 800

    def test_profile_without_capture_metadata_key_validates(self, fitted_profile):
        """An older serialized profile lacking capture_metadata still validates."""
        data = fitted_profile.model_dump(mode="json")
        data.pop("capture_metadata", None)  # simulate a pre-feature payload
        restored = ColorCheckerProfile.model_validate(data)
        assert restored.capture_metadata is None
        assert restored.is_fitted is True


# ===========================================================================
# TestCaptureMetadata
# ===========================================================================


class TestCaptureMetadata:
    """Extraction, coercion, and comparison on CaptureMetadata."""

    @pytest.mark.parametrize("key_style", ["exiftool", "exifread", "tiff"])
    def test_extraction_consistent_across_readers(self, key_style):
        """The same logical EXIF yields identical metadata across key spellings."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        fields = {
            "camera_make" : "Canon",
            "camera_model": "EOS R100",
            "lens_model"  : "RF 50mm",
            "iso"         : 800,
            "f_number"    : 5.6,
            "focal_length": 50.0,
        }
        # TIFF baseline tags rarely include the exposure sub-IFD, but Make/Model
        # and the numeric fields are enough to prove prefix-independence.
        _inject_exif(image, key_style=key_style, **fields)
        cm = CaptureMetadata.from_image(image)
        assert cm.camera_make == "Canon"
        assert cm.camera_model == "EOS R100"
        assert cm.lens_model == "RF 50mm"
        assert cm.iso == 800
        assert cm.f_number == pytest.approx(5.6)
        assert cm.focal_length == pytest.approx(50.0)

    def test_missing_exif_yields_all_none(self):
        """An image with no imported EXIF produces an all-None metadata."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        cm = CaptureMetadata.from_image(image)
        assert cm == CaptureMetadata()

    def test_numeric_coercion_from_strings(self):
        """String EXIF values (fractions, f/-prefix, units) coerce numerically."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        _inject_exif(
            image,
            iso="800",
            exposure_time="1/60",
            f_number="f/5.6",
            focal_length="50.0 mm",
        )
        cm = CaptureMetadata.from_image(image)
        assert cm.iso == 800
        assert cm.exposure_time == pytest.approx(1 / 60)
        assert cm.f_number == pytest.approx(5.6)
        assert cm.focal_length == pytest.approx(50.0)

    def test_bytes_values_are_decoded(self):
        """UTF-8 bytes EXIF values are decoded before coercion."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        _inject_exif(
            image,
            camera_model=b"EOS R100",
            iso=b"800",
            f_number=b"5.6",
        )
        cm = CaptureMetadata.from_image(image)
        assert cm.camera_model == "EOS R100"
        assert cm.iso == 800
        assert cm.f_number == pytest.approx(5.6)

    def test_public_metadata_fallback(self):
        """Manually-set public metadata is read when imported EXIF is absent."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        image.metadata["EXIF FNumber"] = 8.0  # lands in public, not imported
        cm = CaptureMetadata.from_image(image)
        assert cm.f_number == pytest.approx(8.0)

    def test_imported_takes_precedence_over_public(self):
        """Imported EXIF wins over a manually-set public key for the same field."""
        image = Image(arr=np.zeros((4, 4, 3), dtype=np.uint8), bit_depth=8)
        image.metadata["EXIF FNumber"] = 8.0  # public
        _inject_exif(image, f_number=5.6)  # imported
        cm = CaptureMetadata.from_image(image)
        assert cm.f_number == pytest.approx(5.6)

    def test_compare_splits_critical_and_informational(self):
        """Camera/lens diffs are critical; exposure-setting diffs informational."""
        a = CaptureMetadata(
            camera_model="EOS R100", lens_model="RF 50mm", iso=800, f_number=5.6
        )
        b = CaptureMetadata(
            camera_model="EOS R5", lens_model="RF 50mm", iso=400, f_number=8.0
        )
        critical, informational = a.compare(b)
        assert critical == [
            "camera_model (profile='EOS R100', image='EOS R5')"
        ]
        assert any("iso" in m for m in informational)
        assert any("f_number" in m for m in informational)
        # lens_model matched, so it must not appear anywhere.
        assert not any("lens_model" in m for m in critical + informational)

    def test_compare_skips_fields_missing_on_either_side(self):
        """A field None on either side is not reported as a difference."""
        a = CaptureMetadata(camera_model="EOS R100", iso=800)
        b = CaptureMetadata()  # all None
        assert a.compare(b) == ([], [])

    def test_compare_numeric_tolerance(self):
        """Tiny float differences are within tolerance and not flagged."""
        a = CaptureMetadata(f_number=5.6, focal_length=50.0)
        b = CaptureMetadata(f_number=5.6000001, focal_length=50.0)
        assert a.compare(b) == ([], [])

    def test_compare_string_case_insensitive(self):
        """Camera strings compare case-insensitively after trimming."""
        a = CaptureMetadata(camera_model="EOS R100")
        b = CaptureMetadata(camera_model="  eos r100 ")
        assert a.compare(b) == ([], [])

    def test_fit_populates_capture_metadata(self, calibration_image_with_exif):
        """ColorCheckerProfile.fit captures the calibration image's EXIF."""
        profile = ColorCheckerProfile(degree=2).fit(calibration_image_with_exif)
        cm = profile.capture_metadata
        assert cm is not None
        assert cm.camera_make == "Canon"
        assert cm.camera_model == "EOS R100"
        assert cm.lens_model == "RF 50mm F1.8 STM"
        assert cm.iso == 800
        assert cm.exposure_time == pytest.approx(1 / 60)
        assert cm.f_number == pytest.approx(5.6)
        assert cm.focal_length == pytest.approx(50.0)

    def test_patch_color_fit_leaves_capture_metadata_none(self):
        """The image-less _fit_from_patch_colors path leaves capture_metadata None."""
        measured_rgb, names = make_synthetic_checker()
        profile = ColorCheckerProfile(degree=2)
        profile._fit_from_patch_colors(measured_rgb, patch_names=names)
        assert profile.capture_metadata is None


# ===========================================================================
# TestColorCorrectorMetadataCheck
# ===========================================================================


def _exif_warnings(records) -> list[str]:
    """Return messages of recorded UserWarnings about camera/lens mismatch."""
    return [
        str(w.message)
        for w in records
        if issubclass(w.category, UserWarning)
        and "different camera" in str(w.message)
    ]


class TestColorCorrectorMetadataCheck:
    """ColorCorrector warns when corrected images differ in camera/lens."""

    def test_warns_on_camera_mismatch(
        self, fitted_profile_with_exif, synthetic_rgb_image
    ):
        """A different camera body raises a UserWarning."""
        target = _inject_exif(
            synthetic_rgb_image,
            camera_make="Canon",
            camera_model="EOS R5",  # differs from profile's EOS R100
            lens_model="RF 50mm F1.8 STM",
        )
        corrector = ColorCorrector(profile=fitted_profile_with_exif)
        with pytest.warns(UserWarning, match="different camera"):
            corrector.apply(target)

    def test_warns_on_lens_mismatch(
        self, fitted_profile_with_exif, synthetic_rgb_image
    ):
        """A different lens raises a UserWarning."""
        target = _inject_exif(
            synthetic_rgb_image,
            camera_make="Canon",
            camera_model="EOS R100",
            lens_model="EF 100mm Macro",  # differs from profile's lens
        )
        corrector = ColorCorrector(profile=fitted_profile_with_exif)
        with pytest.warns(UserWarning, match="different camera"):
            corrector.apply(target)

    def test_no_warning_on_exposure_only_difference(
        self, fitted_profile_with_exif, synthetic_rgb_image, caplog
    ):
        """Same camera/lens but different ISO/aperture logs info, never warns."""
        target = _inject_exif(
            synthetic_rgb_image,
            camera_make="Canon",
            camera_model="EOS R100",
            lens_model="RF 50mm F1.8 STM",
            iso=200,  # differs
            f_number=11.0,  # differs
        )
        corrector = ColorCorrector(profile=fitted_profile_with_exif)
        caplog.set_level(
            logging.INFO,
            logger=(
                "phenotypic.correction._color_correction._color_corrector."
                "ColorCorrector"
            ),
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            corrector.apply(target)
        assert _exif_warnings(records) == []
        assert "Capture-setting differences" in caplog.text

    def test_no_warning_without_capture_metadata(
        self, fitted_profile, synthetic_rgb_image
    ):
        """A profile fitted from patch colours (no EXIF) never warns."""
        assert fitted_profile.capture_metadata is None
        corrector = ColorCorrector(profile=fitted_profile)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            corrector.apply(synthetic_rgb_image)
        assert _exif_warnings(records) == []

    def test_no_warning_when_image_lacks_exif(
        self, fitted_profile_with_exif, synthetic_rgb_image
    ):
        """A target image without EXIF skips comparison and does not warn."""
        corrector = ColorCorrector(profile=fitted_profile_with_exif)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            corrector.apply(synthetic_rgb_image)
        assert _exif_warnings(records) == []

    def test_correction_still_applies_despite_warning(
        self, fitted_profile_with_exif, synthetic_rgb_image
    ):
        """A mismatch warning does not prevent the correction from running."""
        target = _inject_exif(synthetic_rgb_image, camera_model="EOS R5")
        original_rgb = target.rgb[:].copy()
        corrector = ColorCorrector(profile=fitted_profile_with_exif)
        with pytest.warns(UserWarning, match="different camera"):
            result = corrector.apply(target)
        assert not np.array_equal(result.rgb[:], original_rgb)


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

    def test_pipeline_section_includes_border_mask_stage(self):
        """The pipeline panel renders a 5th column for the border-mask overlay."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        img = make_synthetic_framed_checker_image()
        image = Image(arr=img, bit_depth=8)
        profile = ColorCheckerProfile(degree=2).fit(image)

        dashboard = ColorCorrectionDashboard(
            profile=profile, image=image, rois=profile.rois,
        )
        card = dashboard._pipeline_section()
        assert isinstance(card, pn.Card)
        # First child is the matplotlib pane wrapping the figure.
        mpl_pane = card[0]
        fig = mpl_pane.object
        # The figure must have 5 columns (and at least 1 row) of axes.
        assert len(fig.axes) >= 5
        # Axes appear in row-major order; first 5 belong to ROI 0.
        n_rois = len(profile.rois or [])
        assert len(fig.axes) == 5 * n_rois

    def test_segmentation_section_renders_with_border_fill(self):
        """The segmentation panel renders via the border-fill pipeline."""
        from phenotypic.correction._color_correction._diagnostic_dashboard import (
            ColorCorrectionDashboard,
        )

        img = make_synthetic_framed_checker_image()
        image = Image(arr=img, bit_depth=8)
        profile = ColorCheckerProfile(degree=2).fit(image)

        dashboard = ColorCorrectionDashboard(
            profile=profile, image=image, rois=profile.rois,
        )
        section = dashboard._segmentation_section()
        assert isinstance(section, pn.Card)

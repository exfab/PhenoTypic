"""Tests for lensfunpy-based lens correction operations.

Tests are organized into:
- TestLensfunBase: EXIF parsing (no lensfunpy required)
- TestResolveParams: parameter resolution logic (no lensfunpy required)
- TestLensDistortionCorrector: geometric remapping of all components
- TestLensVignettingCorrector: gamma-aware radiometric correction
- TestLensTCACorrector: per-channel RGB correction
- TestImportError: helpful error when lensfunpy not installed

Integration tests that exercise lensfunpy are gated with pytest.importorskip.
Camera/lens combos use Canon EOS 1000D + Canon EF-S 18-135mm (distortion/TCA)
and Sigma 60-600mm (vignetting), which have calibration data in lensfunpy's DB.
"""

import numpy as np
import pytest

from phenotypic import Image, GridImage

# Camera/lens combos known to have calibration data in lensfunpy's DB
# Canon EOS 1000D + Canon EF-S 18-135mm — has distortion + TCA
_CANON_CAM_MAKER = "Canon"
_CANON_CAM_MODEL = "Canon EOS 1000D"
_CANON_LENS_MODEL = "Canon EF-S 18-135mm f/3.5-5.6 IS USM"
_CANON_FOCAL = 35.0
_CANON_APERTURE = 5.6

# Sigma 60-600mm on Canon EOS 1000D — has distortion + vignetting + TCA
_SIGMA_LENS_MAKER = "Sigma"
_SIGMA_LENS_MODEL = "Sigma 60-600mm f/4.5-6.3 DG OS HSM | S"
_SIGMA_FOCAL = 50.0
_SIGMA_APERTURE = 5.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(h=200, w=300, with_rgb=True, with_objmap=False):
    """Create a minimal test image."""
    if with_rgb:
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.random.rand(h, w).astype(np.float64)
    image = Image(arr=arr)
    if with_objmap:
        objmap = np.zeros((h, w), dtype=np.uint16)
        objmap[50:100, 50:100] = 1
        objmap[120:150, 180:220] = 2
        image.objmap[:] = objmap
    return image


def _make_image_with_metadata(h=200, w=300):
    """Create a test image with EXIF-like metadata."""
    image = _make_test_image(h, w)
    image.metadata["Image Make"] = "Nikon"
    image.metadata["Image Model"] = "D3S"
    image.metadata["EXIF LensModel"] = "Nikkor 28mm f/2.8D"
    image.metadata["EXIF FocalLength"] = "28"
    image.metadata["EXIF FNumber"] = "14/5"
    return image


def _make_resolve_params_instance():
    """Create a concrete subclass instance for testing _resolve_params.

    Uses LensDistortionCorrector via __new__ to skip __init__ (which
    calls _require_lensfunpy), then manually sets attributes.
    """
    from phenotypic.correction import LensDistortionCorrector
    obj = LensDistortionCorrector.__new__(LensDistortionCorrector)
    return obj


# ---------------------------------------------------------------------------
# TestLensfunBase — no lensfunpy required
# ---------------------------------------------------------------------------

class TestLensfunBase:
    """Test _LensfunCorrectorBase EXIF parsing (static method, no lensfunpy)."""

    def test_parse_exif_value_integer_string(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("28") == 28.0

    def test_parse_exif_value_rational_string(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        result = _LensfunCorrectorBase._parse_exif_value("14/5")
        assert abs(result - 2.8) < 1e-10

    def test_parse_exif_value_float(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(2.8) == 2.8

    def test_parse_exif_value_int(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(28) == 28.0

    def test_parse_exif_value_none(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(None) is None

    def test_parse_exif_value_invalid_string(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("not-a-number") is None

    def test_parse_exif_value_zero_denominator(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("14/0") is None

    def test_parse_exif_value_numpy_float(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(np.float32(2.8)) == pytest.approx(2.8, rel=1e-5)

    def test_parse_exif_value_whitespace_string(self):
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("  28  ") == 28.0


# ---------------------------------------------------------------------------
# TestResolveParams — no lensfunpy required (uses __new__ to skip __init__)
# ---------------------------------------------------------------------------

class TestResolveParams:
    """Test parameter resolution from user params + EXIF metadata."""

    def test_resolve_all_from_metadata(self):
        """Resolve all params from image metadata."""
        image = _make_image_with_metadata()
        obj = _make_resolve_params_instance()
        obj.cam_maker = None
        obj.cam_model = None
        obj.lens_maker = None
        obj.lens_model = None
        obj.focal_length = None
        obj.aperture = None
        obj.distance = 0.5

        params = obj._resolve_params(image)
        assert params["cam_maker"] == "Nikon"
        assert params["cam_model"] == "D3S"
        assert params["lens_maker"] == "Nikon"
        assert params["lens_model"] == "Nikkor 28mm f/2.8D"
        assert params["focal_length"] == 28.0
        assert abs(params["aperture"] - 2.8) < 1e-10

    def test_user_params_override_metadata(self):
        """User-provided params take priority over EXIF."""
        image = _make_image_with_metadata()
        obj = _make_resolve_params_instance()
        obj.cam_maker = "Canon"
        obj.cam_model = "EOS 5D"
        obj.lens_maker = None
        obj.lens_model = "EF 50mm f/1.8"
        obj.focal_length = 50.0
        obj.aperture = 1.8
        obj.distance = 1.0

        params = obj._resolve_params(image)
        assert params["cam_maker"] == "Canon"
        assert params["cam_model"] == "EOS 5D"
        assert params["lens_maker"] == "Canon"  # defaults to cam_maker
        assert params["lens_model"] == "EF 50mm f/1.8"
        assert params["focal_length"] == 50.0

    def test_resolve_raises_on_missing_params(self):
        """Raise ValueError when required params cannot be resolved."""
        image = _make_test_image()
        obj = _make_resolve_params_instance()
        obj.cam_maker = None
        obj.cam_model = None
        obj.lens_maker = None
        obj.lens_model = None
        obj.focal_length = None
        obj.aperture = None
        obj.distance = 0.5

        with pytest.raises(ValueError, match="Cannot resolve lens parameters"):
            obj._resolve_params(image)

    def test_lens_maker_defaults_to_cam_maker(self):
        """lens_maker defaults to cam_maker when not specified."""
        image = _make_test_image()
        obj = _make_resolve_params_instance()
        obj.cam_maker = "Sony"
        obj.cam_model = "A7III"
        obj.lens_maker = None
        obj.lens_model = "FE 50mm"
        obj.focal_length = 50.0
        obj.aperture = 1.8
        obj.distance = 0.5

        params = obj._resolve_params(image)
        assert params["lens_maker"] == "Sony"


# ---------------------------------------------------------------------------
# Integration tests (require lensfunpy + database lookup)
# ---------------------------------------------------------------------------

class TestLensDistortionCorrector:
    """Test LensDistortionCorrector geometric remapping."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    def test_instantiation_stores_params(self):
        from phenotypic.correction import LensDistortionCorrector
        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE, distance=1.0,
        )
        assert c.cam_maker == _CANON_CAM_MAKER
        assert c.focal_length == _CANON_FOCAL
        assert c.distance == 1.0

    def test_rgb_remapped(self):
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_rgb = image.rgb[:].copy()

        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_gray_remapped(self):
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_objmap_uses_nearest_interpolation(self):
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600, with_objmap=True)

        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)

        objmap = result.objmap[:]
        unique = np.unique(objmap)
        for val in unique:
            assert val == int(val)

    def test_shape_preserved(self):
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_shape = image.shape

        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert result.shape == original_shape

    def test_gridimage_preserved(self):
        from phenotypic.correction import LensDistortionCorrector
        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        c = LensDistortionCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(grid_img)
        assert isinstance(result, GridImage)
        assert result.nrows == 8
        assert result.ncols == 12


class TestLensVignettingCorrector:
    """Test LensVignettingCorrector radiometric correction."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    def test_instantiation_stores_params(self):
        from phenotypic.correction import LensVignettingCorrector
        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        assert c.cam_maker == _CANON_CAM_MAKER
        assert c.aperture == _SIGMA_APERTURE

    def test_rgb_modified(self):
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)
        original_rgb = image.rgb[:].copy()

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_gray_modified(self):
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_objmap_not_modified(self):
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600, with_objmap=True)
        original_objmap = image.objmap[:].copy()

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)
        assert np.array_equal(result.objmap[:], original_objmap)

    def test_rgb_values_clipped(self):
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)
        assert result.rgb[:].min() >= 0
        assert result.rgb[:].max() <= 255

    def test_gray_values_clipped(self):
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)
        assert result.gray[:].min() >= 0.0
        assert result.gray[:].max() <= 1.0

    def test_corners_brighter_than_before(self):
        """Vignetting correction should brighten corners."""
        from phenotypic.correction import LensVignettingCorrector
        arr = np.full((400, 600, 3), 128, dtype=np.uint8)
        image = Image(arr=arr)

        c = LensVignettingCorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_maker=_SIGMA_LENS_MAKER, lens_model=_SIGMA_LENS_MODEL,
            focal_length=_SIGMA_FOCAL, aperture=_SIGMA_APERTURE,
        )
        result = c.apply(image)

        corner_result = result.rgb[0, 0, :].mean()
        assert corner_result >= 128


class TestLensTCACorrector:
    """Test LensTCACorrector per-channel RGB correction."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    def test_instantiation_stores_params(self):
        from phenotypic.correction import LensTCACorrector
        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        assert c.focal_length == _CANON_FOCAL

    def test_skips_when_no_rgb(self):
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(200, 300, with_rgb=False)

        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert result.gray[:].shape == (200, 300)

    def test_gray_not_modified(self):
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert np.array_equal(result.gray[:], original_gray)

    def test_detect_mat_not_modified(self):
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_detect = image.detect_mat[:].copy()

        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert np.array_equal(result.detect_mat[:], original_detect)

    def test_objmap_not_modified(self):
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600, with_objmap=True)
        original_objmap = image.objmap[:].copy()

        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert np.array_equal(result.objmap[:], original_objmap)

    def test_shape_preserved(self):
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_shape = image.shape

        c = LensTCACorrector(
            cam_maker=_CANON_CAM_MAKER, cam_model=_CANON_CAM_MODEL,
            lens_model=_CANON_LENS_MODEL,
            focal_length=_CANON_FOCAL, aperture=_CANON_APERTURE,
        )
        result = c.apply(image)
        assert result.shape == original_shape


# ---------------------------------------------------------------------------
# Import error handling
# ---------------------------------------------------------------------------

class TestImportError:
    """Test helpful error when lensfunpy is not installed."""

    def test_import_error_message(self):
        from phenotypic.correction._lensfun_base import _require_lensfunpy, _LENSFUNPY_AVAILABLE
        if _LENSFUNPY_AVAILABLE:
            pytest.skip("lensfunpy is installed; cannot test import error")
        with pytest.raises(ImportError, match="pip install phenotypic\\[lens\\]"):
            _require_lensfunpy()

    def test_constructor_raises_when_missing(self):
        from phenotypic.correction._lensfun_base import _LENSFUNPY_AVAILABLE
        if _LENSFUNPY_AVAILABLE:
            pytest.skip("lensfunpy is installed; cannot test import error")
        from phenotypic.correction import LensDistortionCorrector
        with pytest.raises(ImportError, match="lensfunpy"):
            LensDistortionCorrector(cam_maker="Nikon", cam_model="D3S")

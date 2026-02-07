"""Tests for lensfunpy-based lens correction operations.

Tests are organized into:
- TestLensfunBase: parameter storage, EXIF parsing, resolve_params
- TestLensDistortionCorrector: geometric remapping of all components
- TestLensVignettingCorrector: gamma-aware radiometric correction
- TestLensTCACorrector: per-channel RGB correction
- TestImportError: helpful error when lensfunpy not installed

Integration tests that require lensfunpy are gated with pytest.importorskip.
"""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from phenotypic import Image, GridImage


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


# ---------------------------------------------------------------------------
# TestLensfunBase
# ---------------------------------------------------------------------------

class TestLensfunBase:
    """Test _LensfunCorrectorBase parameter handling and EXIF parsing."""

    def test_parse_exif_value_integer_string(self):
        """Parse integer string like '28'."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("28") == 28.0

    def test_parse_exif_value_rational_string(self):
        """Parse rational string like '14/5'."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        result = _LensfunCorrectorBase._parse_exif_value("14/5")
        assert abs(result - 2.8) < 1e-10

    def test_parse_exif_value_float(self):
        """Pass through float values."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(2.8) == 2.8

    def test_parse_exif_value_int(self):
        """Pass through integer values."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(28) == 28.0

    def test_parse_exif_value_none(self):
        """Return None for None input."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(None) is None

    def test_parse_exif_value_invalid_string(self):
        """Return None for unparseable strings."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("not-a-number") is None

    def test_parse_exif_value_zero_denominator(self):
        """Return None for division by zero rational strings."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("14/0") is None

    def test_parse_exif_value_numpy_float(self):
        """Parse numpy float types."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value(np.float32(2.8)) == pytest.approx(2.8, rel=1e-5)

    def test_parse_exif_value_whitespace_string(self):
        """Handle string with whitespace."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        assert _LensfunCorrectorBase._parse_exif_value("  28  ") == 28.0


class TestResolveParams:
    """Test parameter resolution from user params + EXIF metadata."""

    @pytest.fixture
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    @pytest.mark.usefixtures("_skip_if_no_lensfunpy")
    def test_resolve_all_from_metadata(self):
        """Resolve all params from image metadata."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        image = _make_image_with_metadata()
        base = _LensfunCorrectorBase.__new__(_LensfunCorrectorBase)
        base.cam_maker = None
        base.cam_model = None
        base.lens_maker = None
        base.lens_model = None
        base.focal_length = None
        base.aperture = None
        base.distance = 0.5

        params = base._resolve_params(image)
        assert params["cam_maker"] == "Nikon"
        assert params["cam_model"] == "D3S"
        assert params["lens_maker"] == "Nikon"
        assert params["lens_model"] == "Nikkor 28mm f/2.8D"
        assert params["focal_length"] == 28.0
        assert abs(params["aperture"] - 2.8) < 1e-10
        assert params["distance"] == 0.5

    @pytest.mark.usefixtures("_skip_if_no_lensfunpy")
    def test_user_params_override_metadata(self):
        """User-provided params take priority over EXIF."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        image = _make_image_with_metadata()
        base = _LensfunCorrectorBase.__new__(_LensfunCorrectorBase)
        base.cam_maker = "Canon"
        base.cam_model = "EOS 5D"
        base.lens_maker = None
        base.lens_model = "EF 50mm f/1.8"
        base.focal_length = 50.0
        base.aperture = 1.8
        base.distance = 1.0

        params = base._resolve_params(image)
        assert params["cam_maker"] == "Canon"
        assert params["cam_model"] == "EOS 5D"
        assert params["lens_maker"] == "Canon"  # defaults to cam_maker
        assert params["lens_model"] == "EF 50mm f/1.8"
        assert params["focal_length"] == 50.0
        assert params["aperture"] == 1.8

    @pytest.mark.usefixtures("_skip_if_no_lensfunpy")
    def test_resolve_raises_on_missing_params(self):
        """Raise ValueError when required params cannot be resolved."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        image = _make_test_image()  # No EXIF metadata
        base = _LensfunCorrectorBase.__new__(_LensfunCorrectorBase)
        base.cam_maker = None
        base.cam_model = None
        base.lens_maker = None
        base.lens_model = None
        base.focal_length = None
        base.aperture = None
        base.distance = 0.5

        with pytest.raises(ValueError, match="Cannot resolve lens parameters"):
            base._resolve_params(image)

    @pytest.mark.usefixtures("_skip_if_no_lensfunpy")
    def test_lens_maker_defaults_to_cam_maker(self):
        """lens_maker defaults to cam_maker when not specified."""
        from phenotypic.correction._lensfun_base import _LensfunCorrectorBase
        image = _make_test_image()
        base = _LensfunCorrectorBase.__new__(_LensfunCorrectorBase)
        base.cam_maker = "Sony"
        base.cam_model = "A7III"
        base.lens_maker = None
        base.lens_model = "FE 50mm"
        base.focal_length = 50.0
        base.aperture = 1.8
        base.distance = 0.5

        params = base._resolve_params(image)
        assert params["lens_maker"] == "Sony"


# ---------------------------------------------------------------------------
# Integration tests (require lensfunpy)
# ---------------------------------------------------------------------------

class TestLensDistortionCorrector:
    """Test LensDistortionCorrector geometric remapping."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    def test_instantiation_stores_params(self):
        """Test that constructor stores all params."""
        from phenotypic.correction import LensDistortionCorrector
        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8, distance=1.0,
        )
        assert c.cam_maker == "Nikon"
        assert c.cam_model == "D3S"
        assert c.focal_length == 28.0
        assert c.aperture == 2.8
        assert c.distance == 1.0

    def test_rgb_remapped(self):
        """Test that RGB data is modified after distortion correction."""
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_rgb = image.rgb[:].copy()

        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        # Distortion correction should change at least some pixels
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_gray_remapped(self):
        """Test that gray data is modified."""
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_objmap_uses_nearest_interpolation(self):
        """Test that objmap labels remain integers after remapping."""
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600, with_objmap=True)

        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)

        objmap = result.objmap[:]
        # All values should be exact integers (no interpolation artifacts)
        unique = np.unique(objmap)
        for val in unique:
            assert val == int(val)

    def test_shape_preserved(self):
        """Test that image dimensions are preserved after distortion correction."""
        from phenotypic.correction import LensDistortionCorrector
        image = _make_test_image(400, 600)
        original_shape = image.shape

        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert result.shape == original_shape

    def test_gridimage_preserved(self):
        """Test that GridImage type is preserved after correction."""
        from phenotypic.correction import LensDistortionCorrector
        arr = np.random.randint(0, 256, (800, 1000, 3), dtype=np.uint8)
        grid_img = GridImage(arr=arr, nrows=8, ncols=12)

        c = LensDistortionCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
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
        """Test that constructor stores all params."""
        from phenotypic.correction import LensVignettingCorrector
        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        assert c.cam_maker == "Nikon"
        assert c.aperture == 2.8

    def test_rgb_modified(self):
        """Test that RGB data is modified by vignetting correction."""
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)
        original_rgb = image.rgb[:].copy()

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert not np.array_equal(result.rgb[:], original_rgb)

    def test_gray_modified(self):
        """Test that gray data is modified."""
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert not np.array_equal(result.gray[:], original_gray)

    def test_objmap_not_modified(self):
        """Test that objmap is NOT modified by vignetting correction."""
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600, with_objmap=True)
        original_objmap = image.objmap[:].copy()

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert np.array_equal(result.objmap[:], original_objmap)

    def test_rgb_values_clipped(self):
        """Test that RGB output is within valid range [0, 255]."""
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert result.rgb[:].min() >= 0
        assert result.rgb[:].max() <= 255

    def test_gray_values_clipped(self):
        """Test that gray output is within valid range [0, 1]."""
        from phenotypic.correction import LensVignettingCorrector
        image = _make_test_image(400, 600)

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert result.gray[:].min() >= 0.0
        assert result.gray[:].max() <= 1.0

    def test_corners_brighter_than_before(self):
        """Test that corners are brightened (vignetting adds light at edges)."""
        from phenotypic.correction import LensVignettingCorrector
        # Create a uniform image so we can measure the effect
        arr = np.full((400, 600, 3), 128, dtype=np.uint8)
        image = Image(arr=arr)

        c = LensVignettingCorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)

        # Corners should be >= original (vignetting correction brightens edges)
        corner_orig = 128
        corner_result = result.rgb[0, 0, :].mean()
        assert corner_result >= corner_orig


class TestLensTCACorrector:
    """Test LensTCACorrector per-channel RGB correction."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lensfunpy(self):
        pytest.importorskip("lensfunpy")

    def test_instantiation_stores_params(self):
        """Test that constructor stores params."""
        from phenotypic.correction import LensTCACorrector
        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        assert c.focal_length == 28.0

    def test_skips_when_no_rgb(self):
        """Test that TCA corrector returns image unchanged if no RGB data."""
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(200, 300, with_rgb=False)

        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        # Should return without error
        assert result.gray[:].shape == (200, 300)

    def test_gray_not_modified(self):
        """Test that gray data is NOT modified by TCA correction."""
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_gray = image.gray[:].copy()

        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert np.array_equal(result.gray[:], original_gray)

    def test_detect_mat_not_modified(self):
        """Test that detect_mat is NOT modified by TCA correction."""
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_detect = image.detect_mat[:].copy()

        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert np.array_equal(result.detect_mat[:], original_detect)

    def test_objmap_not_modified(self):
        """Test that objmap is NOT modified by TCA correction."""
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600, with_objmap=True)
        original_objmap = image.objmap[:].copy()

        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert np.array_equal(result.objmap[:], original_objmap)

    def test_shape_preserved(self):
        """Test that image shape is preserved after TCA correction."""
        from phenotypic.correction import LensTCACorrector
        image = _make_test_image(400, 600)
        original_shape = image.shape

        c = LensTCACorrector(
            cam_maker="Nikon", cam_model="D3S",
            lens_model="Nikkor 28mm f/2.8D",
            focal_length=28.0, aperture=2.8,
        )
        result = c.apply(image)
        assert result.shape == original_shape


# ---------------------------------------------------------------------------
# Import error handling
# ---------------------------------------------------------------------------

class TestImportError:
    """Test helpful error when lensfunpy is not installed."""

    def test_import_error_message(self):
        """Test that _require_lensfunpy raises helpful ImportError."""
        from phenotypic.correction._lensfun_base import _require_lensfunpy, _LENSFUNPY_AVAILABLE
        if _LENSFUNPY_AVAILABLE:
            pytest.skip("lensfunpy is installed; cannot test import error")

        with pytest.raises(ImportError, match="pip install phenotypic\\[lens\\]"):
            _require_lensfunpy()

    def test_constructor_raises_when_missing(self):
        """Test that constructing any corrector raises ImportError when missing."""
        from phenotypic.correction._lensfun_base import _LENSFUNPY_AVAILABLE
        if _LENSFUNPY_AVAILABLE:
            pytest.skip("lensfunpy is installed; cannot test import error")

        from phenotypic.correction import LensDistortionCorrector
        with pytest.raises(ImportError, match="lensfunpy"):
            LensDistortionCorrector(cam_maker="Nikon", cam_model="D3S")

"""Tests for metadata round-trip I/O functionality.

This module tests the reading and writing of metadata to/from image files
in JPEG, PNG, and TIFF formats, including PhenoTypic-specific metadata.
"""
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PIL_Image

import phenotypic
from phenotypic.tools.constants_ import IO, METADATA

HAS_EXIFTOOL = shutil.which('exiftool') is not None


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    return phenotypic.Image(arr=arr, name="test_rgb")


@pytest.fixture
def sample_gray_image():
    """Create a sample grayscale image for testing."""
    arr = np.random.rand(100, 100).astype(np.float32)
    return phenotypic.Image(arr=arr, name="test_gray", bit_depth=8)


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory for image files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# -----------------------------------------------------------------------------
# Test Metadata Normalization
# -----------------------------------------------------------------------------

class TestMetadataNormalization:
    """Tests for the _normalize_metadata_value helper."""

    def test_normalize_int(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(42) == 42
        assert isinstance(ImageIOHandler._normalize_metadata_value(42), int)

    def test_normalize_float(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(3.14) == 3.14
        assert isinstance(ImageIOHandler._normalize_metadata_value(3.14), float)

    def test_normalize_bool(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(True) is True
        assert ImageIOHandler._normalize_metadata_value(False) is False

    def test_normalize_string(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value("test") == "test"

    def test_normalize_bytes(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(b"test") == "test"

    def test_normalize_none(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(None) is None

    def test_normalize_numpy_int(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value(np.int64(42)) == 42
        assert isinstance(ImageIOHandler._normalize_metadata_value(np.int64(42)), int)

    def test_normalize_numpy_float(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        result = ImageIOHandler._normalize_metadata_value(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_normalize_list_single_element(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        assert ImageIOHandler._normalize_metadata_value([42]) == 42

    def test_normalize_list_multiple_elements(self):
        from phenotypic.core._image_parts._image_io_handler import ImageIOHandler
        result = ImageIOHandler._normalize_metadata_value([1, 2, 3])
        assert result == "[1, 2, 3]"


# -----------------------------------------------------------------------------
# Test PNG Round-Trip
# -----------------------------------------------------------------------------

class TestPNGMetadataRoundTrip:
    """Tests for PNG metadata round-trip."""

    def test_png_roundtrip_gray(self, sample_gray_image, temp_image_dir):
        """Test saving and loading grayscale PNG with metadata."""
        filepath = temp_image_dir / "test_gray.png"

        # Add custom metadata
        sample_gray_image.metadata["test_key"] = "test_value"
        sample_gray_image.metadata["test_int"] = 42

        # Save
        sample_gray_image.gray.imsave(filepath)

        # Verify file exists
        assert filepath.exists()

        # Load and verify metadata
        loaded = phenotypic.Image.imread(filepath)

        # Check PhenoTypic data was restored
        assert loaded._metadata.public.get("test_key") == "test_value"
        assert loaded._metadata.public.get("test_int") == 42

    def test_png_roundtrip_rgb(self, sample_rgb_image, temp_image_dir):
        """Test saving and loading RGB PNG with metadata."""
        filepath = temp_image_dir / "test_rgb.png"

        sample_rgb_image.metadata["experiment"] = "growth_curve"

        # Save RGB
        sample_rgb_image.rgb.imsave(filepath)

        assert filepath.exists()

        # Load and verify
        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("experiment") == "growth_curve"

    def test_png_phenotypic_image_property_gray(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic_image_property is correctly set for gray accessor."""
        filepath = temp_image_dir / "test_property_gray.png"

        sample_gray_image.gray.imsave(filepath)

        # Read the PNG tEXt chunk directly
        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            assert phenotypic_json is not None
            data = json.loads(phenotypic_json)
            assert data["phenotypic_image_property"] == "Image.gray"

    def test_png_phenotypic_image_property_enh_gray(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic_image_property is correctly set for enh_gray accessor."""
        filepath = temp_image_dir / "test_property_enh_gray.png"

        sample_gray_image.enh_gray.imsave(filepath)

        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            assert phenotypic_json is not None
            data = json.loads(phenotypic_json)
            assert data["phenotypic_image_property"] == "Image.enh_gray"

    def test_png_phenotypic_image_property_rgb(self, sample_rgb_image, temp_image_dir):
        """Test that phenotypic_image_property is correctly set for rgb accessor."""
        filepath = temp_image_dir / "test_property_rgb.png"

        sample_rgb_image.rgb.imsave(filepath)

        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            assert phenotypic_json is not None
            data = json.loads(phenotypic_json)
            assert data["phenotypic_image_property"] == "Image.rgb"


# -----------------------------------------------------------------------------
# Test JPEG Round-Trip
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
class TestJPEGMetadataRoundTrip:
    """Tests for JPEG metadata round-trip (requires exiftool)."""

    def test_jpeg_roundtrip_gray(self, sample_gray_image, temp_image_dir):
        """Test saving and loading grayscale JPEG with metadata."""
        filepath = temp_image_dir / "test_gray.jpg"

        sample_gray_image.metadata["test_key"] = "jpeg_test"
        sample_gray_image.gray.imsave(filepath)

        assert filepath.exists()

        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("test_key") == "jpeg_test"

    def test_jpeg_roundtrip_rgb(self, sample_rgb_image, temp_image_dir):
        """Test saving and loading RGB JPEG with metadata."""
        filepath = temp_image_dir / "test_rgb.jpg"

        sample_rgb_image.metadata["experiment"] = "jpeg_growth"
        sample_rgb_image.rgb.imsave(filepath)

        assert filepath.exists()

        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("experiment") == "jpeg_growth"

    def test_jpeg_phenotypic_image_property(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic_image_property is correctly set in JPEG EXIF."""
        filepath = temp_image_dir / "test_property.jpg"

        sample_gray_image.gray.imsave(filepath)

        # Read EXIF UserComment using exiftool
        result = subprocess.run(
            ['exiftool', '-json', '-UserComment', str(filepath)],
            capture_output=True,
            text=True
        )
        exif_data = json.loads(result.stdout)
        user_comment = exif_data[0].get('UserComment')

        assert user_comment is not None
        data = json.loads(user_comment)
        assert data["phenotypic_image_property"] == "Image.gray"


# -----------------------------------------------------------------------------
# Test TIFF Round-Trip
# -----------------------------------------------------------------------------

class TestTIFFMetadataRoundTrip:
    """Tests for TIFF metadata round-trip."""

    def test_tiff_roundtrip_gray(self, sample_gray_image, temp_image_dir):
        """Test saving and loading grayscale TIFF with metadata."""
        filepath = temp_image_dir / "test_gray.tif"

        sample_gray_image.metadata["tiff_test"] = "value"
        sample_gray_image.gray.imsave(filepath)

        assert filepath.exists()

        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("tiff_test") == "value"

    def test_tiff_roundtrip_rgb(self, sample_rgb_image, temp_image_dir):
        """Test saving and loading RGB TIFF with metadata."""
        filepath = temp_image_dir / "test_rgb.tiff"

        sample_rgb_image.metadata["experiment"] = "tiff_growth"
        sample_rgb_image.rgb.imsave(filepath)

        assert filepath.exists()

        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("experiment") == "tiff_growth"

    def test_tiff_phenotypic_image_property(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic_image_property is correctly set in TIFF ImageDescription."""
        filepath = temp_image_dir / "test_property.tif"

        sample_gray_image.enh_gray.imsave(filepath)

        # Read ImageDescription tag directly
        with PIL_Image.open(filepath) as img:
            desc = img.tag_v2.get(270)  # ImageDescription tag
            assert desc is not None
            data = json.loads(desc)
            assert data["phenotypic_image_property"] == "Image.enh_gray"


# -----------------------------------------------------------------------------
# Test Protected Metadata Preservation
# -----------------------------------------------------------------------------

class TestProtectedMetadataPreservation:
    """Tests for protected metadata preservation during round-trip."""

    def test_bit_depth_preserved(self, sample_gray_image, temp_image_dir):
        """Test that bit depth is preserved through round-trip."""
        filepath = temp_image_dir / "test_bitdepth.png"

        original_bit_depth = sample_gray_image.bit_depth
        sample_gray_image.gray.imsave(filepath)

        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.protected.get(METADATA.BIT_DEPTH) == original_bit_depth

    def test_image_name_not_overwritten(self, sample_gray_image, temp_image_dir):
        """Test that image name from filename takes precedence."""
        filepath = temp_image_dir / "new_name.png"

        sample_gray_image.gray.imsave(filepath)
        loaded = phenotypic.Image.imread(filepath)

        # Name should come from filename, not saved metadata
        assert loaded.name == "new_name"

    def test_enh_gray_metadata_not_restored(self, sample_gray_image, temp_image_dir):
        """Test that metadata is NOT restored when image was saved from enh_gray."""
        filepath = temp_image_dir / "test_enh_gray.png"

        # Add custom metadata
        sample_gray_image.metadata["should_not_restore"] = "test_value"

        # Save from enh_gray (not rgb or gray)
        sample_gray_image.enh_gray.imsave(filepath)

        # Load and verify metadata was NOT restored to public
        # (only rgb and gray sources restore metadata on imread)
        loaded = phenotypic.Image.imread(filepath)
        assert loaded._metadata.public.get("should_not_restore") is None


# -----------------------------------------------------------------------------
# Test Version Info
# -----------------------------------------------------------------------------

class TestVersionInfo:
    """Tests for version information in saved metadata."""

    def test_version_saved_png(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic version is saved in PNG metadata."""
        filepath = temp_image_dir / "test_version.png"

        sample_gray_image.gray.imsave(filepath)

        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            data = json.loads(phenotypic_json)
            assert "phenotypic_version" in data
            assert data["phenotypic_version"] == phenotypic.__version__

    def test_version_saved_tiff(self, sample_gray_image, temp_image_dir):
        """Test that phenotypic version is saved in TIFF metadata."""
        filepath = temp_image_dir / "test_version.tif"

        sample_gray_image.gray.imsave(filepath)

        with PIL_Image.open(filepath) as img:
            desc = img.tag_v2.get(270)
            data = json.loads(desc)
            assert "phenotypic_version" in data
            assert data["phenotypic_version"] == phenotypic.__version__


# -----------------------------------------------------------------------------
# Test Accessor Property Names
# -----------------------------------------------------------------------------

class TestAccessorPropertyNames:
    """Tests for accessor property name class attributes."""

    def test_grayscale_accessor_property_name(self):
        """Test Grayscale accessor has correct property name."""
        from phenotypic.core._image_parts.accessors._grayscale_accessor import Grayscale
        assert Grayscale._accessor_property_name == "gray"

    def test_enhanced_grayscale_accessor_property_name(self):
        """Test EnhancedGrayscale accessor has correct property name."""
        from phenotypic.core._image_parts.accessors._enh_grayscale_accessor import EnhancedGrayscale
        assert EnhancedGrayscale._accessor_property_name == "enh_gray"

    def test_rgb_accessor_property_name(self):
        """Test ImageRGB accessor has correct property name."""
        from phenotypic.core._image_parts.accessors._array_accessor import ImageRGB
        assert ImageRGB._accessor_property_name == "rgb"

    def test_base_accessor_property_name_default(self):
        """Test ImageAccessorBase has default property name."""
        from phenotypic.core._image_parts.accessor_abstracts._image_accessor_base import ImageAccessorBase
        assert ImageAccessorBase._accessor_property_name == "unknown"

    def test_xyz_accessor_property_name(self):
        """Test XyzAccessor has correct property name."""
        from phenotypic.core._image_parts.color_space_accessors._xyz_accessor import XyzAccessor
        assert XyzAccessor._accessor_property_name == "color.XYZ"

    def test_xyz_d65_accessor_property_name(self):
        """Test XyzD65Accessor has correct property name."""
        from phenotypic.core._image_parts.color_space_accessors._xyz_d65_accessor import XyzD65Accessor
        assert XyzD65Accessor._accessor_property_name == "color.XYZ_D65"

    def test_cielab_accessor_property_name(self):
        """Test CieLabAccessor has correct property name."""
        from phenotypic.core._image_parts.color_space_accessors._cielab_accessor import CieLabAccessor
        assert CieLabAccessor._accessor_property_name == "color.Lab"

    def test_chromaticity_xy_accessor_property_name(self):
        """Test xyChromaticityAccessor has correct property name."""
        from phenotypic.core._image_parts.color_space_accessors._chromaticity_xy_accessor import xyChromaticityAccessor
        assert xyChromaticityAccessor._accessor_property_name == "color.xy"

    def test_hsv_accessor_property_name(self):
        """Test HsvAccessor has correct property name."""
        from phenotypic.core._image_parts.accessors._hsv_accessor import HsvAccessor
        assert HsvAccessor._accessor_property_name == "color.hsv"

    def test_color_space_accessor_base_property_name(self):
        """Test ColorSpaceAccessor has default property name."""
        from phenotypic.core._image_parts.accessor_abstracts._color_space_accessor import ColorSpaceAccessor
        assert ColorSpaceAccessor._accessor_property_name == "color.unknown"


# -----------------------------------------------------------------------------
# Test Color Space TIFF Round-Trip
# -----------------------------------------------------------------------------

class TestColorSpaceTIFFRoundTrip:
    """Tests for color space accessor TIFF metadata round-trip."""

    def test_color_space_xyz_tiff_metadata(self, sample_rgb_image, temp_image_dir):
        """Test XYZ color space saves with correct metadata in TIFF."""
        import tifffile

        filepath = temp_image_dir / "test_xyz.tif"

        sample_rgb_image.color.XYZ.imsave(filepath)

        assert filepath.exists()

        # Use tifffile to read float TIFF metadata
        with tifffile.TiffFile(filepath) as tif:
            desc = tif.pages[0].description
            assert desc is not None
            data = json.loads(desc)
            assert data["phenotypic_image_property"] == "Image.color.XYZ"
            assert "phenotypic_version" in data

    def test_color_space_lab_tiff_metadata(self, sample_rgb_image, temp_image_dir):
        """Test Lab color space saves with correct metadata in TIFF."""
        import tifffile

        filepath = temp_image_dir / "test_lab.tif"

        sample_rgb_image.color.Lab.imsave(filepath)

        assert filepath.exists()

        with tifffile.TiffFile(filepath) as tif:
            desc = tif.pages[0].description
            assert desc is not None
            data = json.loads(desc)
            assert data["phenotypic_image_property"] == "Image.color.Lab"

    def test_color_space_hsv_tiff_metadata(self, sample_rgb_image, temp_image_dir):
        """Test HSV color space saves with correct metadata in TIFF."""
        import tifffile

        filepath = temp_image_dir / "test_hsv.tif"

        sample_rgb_image.color.hsv.imsave(filepath)

        assert filepath.exists()

        with tifffile.TiffFile(filepath) as tif:
            desc = tif.pages[0].description
            assert desc is not None
            data = json.loads(desc)
            assert data["phenotypic_image_property"] == "Image.color.hsv"

    def test_color_space_rejects_non_tiff(self, sample_rgb_image, temp_image_dir):
        """Test that color space accessor raises error for non-TIFF formats."""
        filepath = temp_image_dir / "test_xyz.png"

        with pytest.raises(ValueError, match="Color space arrays can only be saved in TIFF format"):
            sample_rgb_image.color.XYZ.imsave(filepath)


# -----------------------------------------------------------------------------
# Test Accessor Load Methods
# -----------------------------------------------------------------------------

class TestAccessorLoad:
    """Tests for accessor load class methods."""

    def test_grayscale_load_success(self, sample_gray_image, temp_image_dir):
        """Test Grayscale.load() with matching metadata."""
        from phenotypic.core._image_parts.accessors._grayscale_accessor import Grayscale

        filepath = temp_image_dir / "test_gray.png"
        sample_gray_image.gray.imsave(filepath)

        # Should load without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = Grayscale.load(filepath)
            # Filter for our specific warnings
            phenotypic_warnings = [x for x in w if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)

    def test_grayscale_load_mismatch_warning(self, sample_rgb_image, temp_image_dir):
        """Test Grayscale.load() warns when metadata doesn't match."""
        from phenotypic.core._image_parts.accessors._grayscale_accessor import Grayscale

        filepath = temp_image_dir / "test_rgb.png"
        # Save from RGB accessor
        sample_rgb_image.rgb.imsave(filepath)

        # Load with Grayscale.load() should warn about mismatch
        with pytest.warns(UserWarning, match="Metadata mismatch"):
            arr = Grayscale.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_rgb_load_success(self, sample_rgb_image, temp_image_dir):
        """Test ImageRGB.load() with matching metadata."""
        from phenotypic.core._image_parts.accessors._array_accessor import ImageRGB

        filepath = temp_image_dir / "test_rgb.png"
        sample_rgb_image.rgb.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = ImageRGB.load(filepath)
            phenotypic_warnings = [x for x in w if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3

    def test_load_missing_metadata_warning(self, temp_image_dir):
        """Test load warns when no PhenoTypic metadata exists."""
        from phenotypic.core._image_parts.accessors._grayscale_accessor import Grayscale

        # Create a plain image without PhenoTypic metadata
        filepath = temp_image_dir / "plain_image.png"
        plain_arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        PIL_Image.fromarray(plain_arr).save(filepath)

        with pytest.warns(UserWarning, match="No PhenoTypic metadata found"):
            arr = Grayscale.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_color_space_load_success(self, sample_rgb_image, temp_image_dir):
        """Test ColorSpaceAccessor.load() with matching metadata."""
        import tifffile
        from phenotypic.core._image_parts.color_space_accessors._cielab_accessor import CieLabAccessor

        filepath = temp_image_dir / "test_lab.tif"
        sample_rgb_image.color.Lab.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = CieLabAccessor.load(filepath)
            phenotypic_warnings = [x for x in w if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_color_space_load_mismatch_warning(self, sample_rgb_image, temp_image_dir):
        """Test ColorSpaceAccessor.load() warns when metadata doesn't match."""
        from phenotypic.core._image_parts.color_space_accessors._xyz_accessor import XyzAccessor
        from phenotypic.core._image_parts.color_space_accessors._cielab_accessor import CieLabAccessor

        filepath = temp_image_dir / "test_xyz.tif"
        # Save from XYZ accessor
        sample_rgb_image.color.XYZ.imsave(filepath)

        # Load with Lab accessor should warn
        with pytest.warns(UserWarning, match="Metadata mismatch"):
            arr = CieLabAccessor.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_color_space_load_rejects_non_tiff(self, temp_image_dir):
        """Test ColorSpaceAccessor.load() raises error for non-TIFF."""
        from phenotypic.core._image_parts.color_space_accessors._cielab_accessor import CieLabAccessor

        filepath = temp_image_dir / "test.png"
        with pytest.raises(ValueError, match="can only be loaded from TIFF format"):
            CieLabAccessor.load(filepath)

    def test_hsv_load_success(self, sample_rgb_image, temp_image_dir):
        """Test HsvAccessor.load() with matching metadata."""
        from phenotypic.core._image_parts.accessors._hsv_accessor import HsvAccessor

        filepath = temp_image_dir / "test_hsv.tif"
        sample_rgb_image.color.hsv.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = HsvAccessor.load(filepath)
            phenotypic_warnings = [x for x in w if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape[2] == 3  # HSV has 3 channels


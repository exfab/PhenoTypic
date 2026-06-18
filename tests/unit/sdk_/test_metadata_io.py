"""Tests for metadata round-trip I/O functionality.

This module tests the reading and writing of metadata to/from image files
in JPEG, PNG, and TIFF formats, including PhenoTypic-specific metadata.
"""

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image as PIL_Image

import phenotypic
from phenotypic.schema import METADATA
from phenotypic.sdk_.constants_ import IO

HAS_EXIFTOOL = shutil.which("exiftool") is not None


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
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(42) == 42
        assert isinstance(ImageIOHandler._normalize_metadata_value(42), int)

    def test_normalize_float(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(3.14) == 3.14
        assert isinstance(ImageIOHandler._normalize_metadata_value(3.14), float)

    def test_normalize_bool(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(True) is True
        assert ImageIOHandler._normalize_metadata_value(False) is False

    def test_normalize_string(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value("test") == "test"

    def test_normalize_bytes(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(b"test") == "test"

    def test_normalize_none(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(None) is None

    def test_normalize_numpy_int(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value(np.int64(42)) == 42
        assert isinstance(ImageIOHandler._normalize_metadata_value(np.int64(42)), int)

    def test_normalize_numpy_float(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        result = ImageIOHandler._normalize_metadata_value(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_normalize_list_single_element(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

        assert ImageIOHandler._normalize_metadata_value([42]) == 42

    def test_normalize_list_multiple_elements(self):
        from phenotypic._core._image_parts._image_io_handler import ImageIOHandler

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

    def test_png_phenotypic_image_property_gray(
            self, sample_gray_image, temp_image_dir
    ):
        """Test that phenotypic_image_property is correctly set for gray accessor."""
        filepath = temp_image_dir / "test_property_gray.png"

        sample_gray_image.gray.imsave(filepath)

        # Read the PNG tEXt chunk directly
        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            assert phenotypic_json is not None
            data = json.loads(phenotypic_json)
            assert data["phenotypic_image_property"] == "Image.gray"

    def test_png_phenotypic_image_property_detect_mat(
            self, sample_gray_image, temp_image_dir
    ):
        """Test that phenotypic_image_property is correctly set for detect_mat accessor."""
        filepath = temp_image_dir / "test_property_detect_mat.png"

        sample_gray_image.detect_mat.imsave(filepath)

        with PIL_Image.open(filepath) as img:
            phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
            assert phenotypic_json is not None
            data = json.loads(phenotypic_json)
            assert data["phenotypic_image_property"] == "Image.detect_mat"

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
                ["exiftool", "-json", "-UserComment", str(filepath)],
                capture_output=True,
                text=True,
        )
        exif_data = json.loads(result.stdout)
        user_comment = exif_data[0].get("UserComment")

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

        sample_gray_image.detect_mat.imsave(filepath)

        # Read ImageDescription tag directly
        with PIL_Image.open(filepath) as img:
            desc = img.tag_v2.get(270)  # ImageDescription tag
            assert desc is not None
            data = json.loads(desc)
            assert data["phenotypic_image_property"] == "Image.detect_mat"


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

    def test_detect_mat_metadata_not_restored(self, sample_gray_image, temp_image_dir):
        """Test that metadata is NOT restored when image was saved from detect_mat."""
        filepath = temp_image_dir / "test_detect_mat.png"

        # Add custom metadata
        sample_gray_image.metadata["should_not_restore"] = "test_value"

        # Save from detect_mat (not rgb or gray)
        sample_gray_image.detect_mat.imsave(filepath)

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
        from phenotypic._core._image_parts.accessors._grayscale_accessor import \
            Grayscale

        assert Grayscale._accessor_property_name_value() == "gray"

    def test_enhanced_grayscale_accessor_property_name(self):
        """Test DetectMatAccessor accessor has correct property name."""
        from phenotypic._core._image_parts.accessors._detect_mat_accessor import (
            DetectMatAccessor,
        )

        assert DetectMatAccessor._accessor_property_name_value() == "detect_mat"

    def test_rgb_accessor_property_name(self):
        """Test ImageRGB accessor has correct property name."""
        from phenotypic._core._image_parts.accessors._rgb_accessor import ImageRGB

        assert ImageRGB._accessor_property_name_value() == "rgb"

    def test_xyz_accessor_property_name(self):
        """Test XyzAccessor has correct property name."""
        from phenotypic._core._image_parts.color_space_accessors._xyz_accessor import (
            XyzAccessor,
        )

        assert XyzAccessor._accessor_property_name_value() == "color.XYZ"

    def test_xyz_d65_accessor_property_name(self):
        """Test XyzD65Accessor has correct property name."""
        from phenotypic._core._image_parts.color_space_accessors._xyz_d65_accessor import (
            XyzD65Accessor,
        )

        assert XyzD65Accessor._accessor_property_name_value() == "color.XYZ_D65"

    def test_cielab_accessor_property_name(self):
        """Test CieLabAccessor has correct property name."""
        from phenotypic._core._image_parts.color_space_accessors._cielab_accessor import (
            CieLabAccessor,
        )

        assert CieLabAccessor._accessor_property_name_value() == "color.Lab"

    def test_chromaticity_xy_accessor_property_name(self):
        """Test xyChromaticityAccessor has correct property name."""
        from phenotypic._core._image_parts.color_space_accessors._chromaticity_xy_accessor import (
            xyChromaticityAccessor,
        )

        assert xyChromaticityAccessor._accessor_property_name_value() == "color.xy"

    def test_hsv_accessor_property_name(self):
        """Test HsvAccessor has correct property name."""
        from phenotypic._core._image_parts.color_space_accessors._hsv_accessor import \
            HsvAccessor

        assert HsvAccessor._accessor_property_name_value() == "color.hsv"

    def test_color_space_accessor_base_property_name(self):
        """Test ColorSpaceAccessor has default property name."""
        from phenotypic._core._image_parts.accessor_abstracts._color_space_accessor import (
            ColorSpaceAccessor,
        )

        assert ColorSpaceAccessor._accessor_property_name_value() == "color.unknown"


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

        with pytest.raises(
                ValueError, match="Color space arrays can only be saved in TIFF format"
        ):
            sample_rgb_image.color.XYZ.imsave(filepath)


# -----------------------------------------------------------------------------
# Test Accessor Load Methods
# -----------------------------------------------------------------------------


class TestAccessorLoad:
    """Tests for accessor load class methods."""

    def test_grayscale_load_success(self, sample_gray_image, temp_image_dir):
        """Test Grayscale.load() with matching metadata."""
        from phenotypic._core._image_parts.accessors._grayscale_accessor import \
            Grayscale

        filepath = temp_image_dir / "test_gray.png"
        sample_gray_image.gray.imsave(filepath)

        # Should load without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = Grayscale.load(filepath)
            # Filter for our specific warnings
            phenotypic_warnings = [
                x
                for x in w
                if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)
            ]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)

    def test_grayscale_load_mismatch_warning(self, sample_rgb_image, temp_image_dir):
        """Test Grayscale.load() warns when metadata doesn't match."""
        from phenotypic._core._image_parts.accessors._grayscale_accessor import \
            Grayscale

        filepath = temp_image_dir / "test_rgb.png"
        # Save from RGB accessor
        sample_rgb_image.rgb.imsave(filepath)

        # Load with Grayscale.load() should warn about mismatch
        with pytest.warns(UserWarning, match="Metadata mismatch"):
            arr = Grayscale.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_rgb_load_success(self, sample_rgb_image, temp_image_dir):
        """Test ImageRGB.load() with matching metadata."""
        from phenotypic._core._image_parts.accessors._rgb_accessor import ImageRGB

        filepath = temp_image_dir / "test_rgb.png"
        sample_rgb_image.rgb.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = ImageRGB.load(filepath)
            phenotypic_warnings = [
                x
                for x in w
                if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)
            ]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3

    def test_load_missing_metadata_warning(self, temp_image_dir):
        """Test load warns when no PhenoTypic metadata exists."""
        from phenotypic._core._image_parts.accessors._grayscale_accessor import \
            Grayscale

        # Create a plain image without PhenoTypic metadata
        filepath = temp_image_dir / "plain_image.png"
        plain_arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        PIL_Image.fromarray(plain_arr).save(filepath)

        with pytest.warns(UserWarning, match="No PhenoTypic metadata found"):
            arr = Grayscale.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_color_space_load_success(self, sample_rgb_image, temp_image_dir):
        """Test ColorSpaceAccessor.load() with matching metadata."""
        from phenotypic._core._image_parts.color_space_accessors._cielab_accessor import (
            CieLabAccessor,
        )

        filepath = temp_image_dir / "test_lab.tif"
        sample_rgb_image.color.Lab.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = CieLabAccessor.load(filepath)
            phenotypic_warnings = [
                x
                for x in w
                if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)
            ]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_color_space_load_mismatch_warning(self, sample_rgb_image, temp_image_dir):
        """Test ColorSpaceAccessor.load() warns when metadata doesn't match."""
        from phenotypic._core._image_parts.color_space_accessors._cielab_accessor import (
            CieLabAccessor,
        )

        filepath = temp_image_dir / "test_xyz.tif"
        # Save from XYZ accessor
        sample_rgb_image.color.XYZ.imsave(filepath)

        # Load with Lab accessor should warn
        with pytest.warns(UserWarning, match="Metadata mismatch"):
            arr = CieLabAccessor.load(filepath)

        assert isinstance(arr, np.ndarray)

    def test_color_space_load_rejects_non_tiff(self, temp_image_dir):
        """Test ColorSpaceAccessor.load() raises error for non-TIFF."""
        from phenotypic._core._image_parts.color_space_accessors._cielab_accessor import (
            CieLabAccessor,
        )

        filepath = temp_image_dir / "test.png"
        with pytest.raises(ValueError, match="can only be loaded from TIFF format"):
            CieLabAccessor.load(filepath)

    def test_hsv_load_success(self, sample_rgb_image, temp_image_dir):
        """Test HsvAccessor.load() with matching metadata."""
        from phenotypic._core._image_parts.color_space_accessors._hsv_accessor import \
            HsvAccessor

        filepath = temp_image_dir / "test_hsv.tif"
        sample_rgb_image.color.hsv.imsave(filepath)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arr = HsvAccessor.load(filepath)
            phenotypic_warnings = [
                x
                for x in w
                if "PhenoTypic" in str(x.message) or "mismatch" in str(x.message)
            ]
            assert len(phenotypic_warnings) == 0

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape[2] == 3  # HSV has 3 channels


# -----------------------------------------------------------------------------
# Legacy metadata-key shim (loading HDF5 written before the Metadata_ prefix)
# -----------------------------------------------------------------------------


def _write_legacy_flat_hdf5(path, *, protected: dict, public: dict) -> None:
    """Write a minimal schema_version=1 flat-layout HDF5 with *bare* metadata keys.

    Mirrors how PhenoTypic persisted images before the ``METADATA`` enum gained
    its ``Metadata_`` category prefix: framework keys stored bare (``ImageName``,
    ``BitDepth``, …) in the ``protected_metadata`` / ``public_metadata`` attribute
    subgroups, and no ``schema_version`` root attr (so the loader dispatches to the
    legacy flat path).
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=np.zeros((8, 8, 3), dtype=np.uint8))
        f.create_dataset("gray", data=np.zeros((8, 8), dtype=np.uint8))
        f.create_dataset("detect_mat", data=np.zeros((8, 8), dtype=np.uint8))
        f.create_dataset("objmap", data=np.zeros((8, 8), dtype=np.int32))
        prot = f.require_group("protected_metadata")
        for key, val in protected.items():
            prot.attrs[key] = val
        pub = f.require_group("public_metadata")
        for key, val in public.items():
            pub.attrs[key] = val


class TestLegacyMetadataKeyShim:
    """Old HDF5 files with bare metadata keys load under the new Metadata_* keys."""

    def test_remap_helper_maps_bare_to_prefixed(self):
        """The helper maps bare framework labels to their prefixed value."""
        from phenotypic._core._image_parts._image_io_handler import (
            _remap_legacy_metadata_key,
        )

        assert _remap_legacy_metadata_key("ImageName") == METADATA.IMAGE_NAME.value
        assert _remap_legacy_metadata_key("BitDepth") == METADATA.BIT_DEPTH.value
        assert _remap_legacy_metadata_key("FileSuffix") == METADATA.SUFFIX.value

    def test_remap_helper_is_idempotent_and_passes_through_user_keys(self):
        """Already-prefixed keys and arbitrary user keys pass through unchanged."""
        from phenotypic._core._image_parts._image_io_handler import (
            _remap_legacy_metadata_key,
        )

        # Already prefixed (a new file's key) -> unchanged.
        assert _remap_legacy_metadata_key("Metadata_ImageName") == "Metadata_ImageName"
        # Arbitrary biological tag a user supplied -> unchanged.
        assert _remap_legacy_metadata_key("Strain") == "Strain"

    def test_legacy_flat_hdf5_bare_keys_remapped_on_load(self, temp_image_dir):
        """Legacy flat HDF5 with bare keys loads under prefixed METADATA keys.

        Framework keys are remapped (with the legacy digit-string -> int
        coercion preserved), no stale bare keys survive, and arbitrary
        user-supplied public keys pass through untouched.
        """
        path = temp_image_dir / "legacy_meta.h5"
        _write_legacy_flat_hdf5(
            path,
            protected={
                "ImageName": "legacy_name",
                "ImageType": "Image",
                "BitDepth": "8",            # legacy digit-string -> int coercion
                "ParentUUID": "parent-123",  # field with no constructor default
            },
            public={
                "FileSuffix": ".png",
                "Strain": "BY4741",          # arbitrary user key -> untouched
            },
        )

        loaded = phenotypic.Image.load_hdf5(path)
        prot = loaded._metadata.protected
        pub = loaded._metadata.public

        # Framework keys remapped to the prefixed METADATA members ...
        assert prot[METADATA.IMAGE_NAME] == "legacy_name"
        assert prot[METADATA.BIT_DEPTH] == 8            # int-coerced
        assert prot[METADATA.PARENT_UUID] == "parent-123"
        assert pub[METADATA.SUFFIX] == ".png"
        # ... with no stale bare keys left behind.
        for bare in ("ImageName", "ImageType", "BitDepth", "ParentUUID", "FileSuffix"):
            assert bare not in prot
            assert bare not in pub
        # Arbitrary, non-framework keys pass through unchanged.
        assert pub["Strain"] == "BY4741"

    def test_legacy_v2_hdf5_has_no_duplicate_bare_keys(self, temp_image_dir):
        """A v2 file whose protected keys were stored bare loads without duplicates.

        Simulates an old schema_version=2 file by rewriting the on-disk protected
        metadata attribute names back to their bare labels, then asserts the shim
        folds them onto the prefixed keys instead of adding stale bare duplicates.
        """
        img = phenotypic.Image(
            arr=np.zeros((16, 24, 3), dtype=np.uint8), name="v2_legacy", bit_depth=8
        )
        path = temp_image_dir / "v2_legacy.h5"
        img.save2hdf5(path)

        # Rewrite prefixed protected attr names -> bare labels, in place.
        with h5py.File(path, "r+") as f:
            prot_attrs = f["metadata"]["protected"].attrs
            for member in METADATA:
                if member.value in prot_attrs:
                    val = prot_attrs[member.value]
                    del prot_attrs[member.value]
                    prot_attrs[member.label] = val

        loaded = phenotypic.Image.load_hdf5(path)
        prot = loaded._metadata.protected

        # No bare framework keys survive; every protected key is Metadata_*-prefixed.
        assert all(str(k).startswith("Metadata_") for k in prot)
        assert "ImageName" not in prot and "BitDepth" not in prot
        assert loaded.bit_depth == 8

    def test_legacy_png_bare_metadata_keys_remapped_on_imread(self, temp_image_dir):
        """Old PNG/JPEG ``_phenotypic_data`` with bare keys remaps on imread.

        PNG embeds the round-trip payload as JSON in a tEXt chunk and JPEG in an
        EXIF UserComment; both feed the *same* restore block, so this PNG case
        also covers JPEG. Bare framework keys are remapped, the critical-field
        skip (UUID / ImageName) still fires for old files, and arbitrary user
        keys pass through untouched.
        """
        from PIL.PngImagePlugin import PngInfo

        path = temp_image_dir / "legacy.png"
        payload = {
            "phenotypic_version": "0.14.0",
            "phenotypic_image_property": "Image.rgb",
            "protected": {
                "ImageName": "saved_name",   # critical -> remapped then skipped
                "UUID": "saved-uuid",        # critical -> remapped then skipped
                "ParentUUID": "parent-9",    # non-critical -> remapped + restored
            },
            "public": {
                "FileSuffix": ".png",
                "Strain": "BY4741",          # arbitrary user key -> untouched
            },
        }
        info = PngInfo()
        info.add_text(IO.PHENOTYPIC_METADATA_KEY, json.dumps(payload))
        PIL_Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            path, pnginfo=info
        )

        loaded = phenotypic.Image.imread(path)
        prot = loaded._metadata.protected
        pub = loaded._metadata.public

        # Critical fields come from the import flow, not the saved payload; the
        # saved bare "UUID"/"ImageName" must not leak in under any spelling.
        assert "UUID" not in prot and "ImageName" not in prot
        assert loaded.name == "legacy"               # from filename, not payload
        # Non-critical framework key remapped + restored under its prefixed key.
        assert prot[METADATA.PARENT_UUID] == "parent-9"
        # Public framework key remapped; arbitrary user key untouched.
        assert pub[METADATA.SUFFIX] == ".png"
        assert "FileSuffix" not in pub
        assert pub["Strain"] == "BY4741"

    def test_backcompat_unpickler_remaps_moved_metadata_class(self):
        """The back-compat unpickler maps the old METADATA import path to schema."""
        import io as _io

        from phenotypic._core._image_parts._image_io_handler import (
            _BackCompatUnpickler,
        )

        unpickler = _BackCompatUnpickler(_io.BytesIO(b""))
        # The moved symbol resolves to the current schema.METADATA. The legacy
        # path is the genuinely-old ``tools_`` location pre-existing pickles were
        # written with (the package has since been renamed to ``sdk_``); the
        # shim must keep matching it verbatim.
        assert (
            unpickler.find_class("phenotypic.tools_.constants_", "METADATA")
            is METADATA
        )
        # ... and unmoved classes pass through unchanged.
        assert unpickler.find_class("numpy", "ndarray") is np.ndarray

    def test_current_pickle_roundtrip_unaffected(self, temp_image_dir):
        """A current pickle still round-trips cleanly through the back-compat path."""
        img = phenotypic.Image(
            arr=np.zeros((12, 16, 3), dtype=np.uint8), name="pk", bit_depth=8
        )
        img.metadata["Strain"] = "BY4741"
        path = temp_image_dir / "cur.pkl"
        img.save2pickle(path)

        loaded = phenotypic.Image.load_pickle(path)
        assert loaded.bit_depth == 8
        assert loaded._metadata.public["Strain"] == "BY4741"
        # Protected keys are the prefixed METADATA members; no bare keys.
        assert all(str(k).startswith("Metadata_") for k in loaded._metadata.protected)

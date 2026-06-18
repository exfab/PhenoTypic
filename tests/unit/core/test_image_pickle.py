"""Tests for Image and GridImage pickle save/load functionality.

This module tests the pickle serialization and deserialization behavior for both
Image and GridImage classes, ensuring that:
- GridFinder objects are properly saved and restored
- Image.load_pickle automatically returns the correct type (Image vs GridImage)
- Backward compatibility with old pickles is maintained
- All image data, metadata, and grid configuration are preserved
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image, GridImage
from phenotypic.grid import AutoGridFinder, CenteredAutoGridFinder


class TestImagePickle:
    """Test suite for Image pickle save/load functionality."""

    def test_save_and_load_image_rgb(self):
        """Test saving and loading a regular RGB Image."""
        # Create RGB image
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(rgb_array, name="test_rgb")

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify it's an Image (not GridImage)
            assert isinstance(loaded, Image)
            assert not isinstance(loaded, GridImage)

            # Verify data is preserved
            assert np.array_equal(loaded.rgb[:], img.rgb[:])
            assert loaded.shape == img.shape

    def test_save_and_load_image_grayscale(self):
        """Test saving and loading a grayscale Image."""
        # Create grayscale image
        gray_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image(gray_array, name="test_gray")

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify it's an Image (not GridImage)
            assert isinstance(loaded, Image)
            assert not isinstance(loaded, GridImage)

            # Verify data is preserved
            assert np.array_equal(loaded.gray[:], img.gray[:])
            assert loaded.shape == img.shape

    def test_image_pickle_preserves_metadata(self):
        """Test that metadata is preserved during pickle save/load."""
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(rgb_array)

        # Add some metadata
        img.metadata["test_key"] = "test_value"
        img.metadata["numeric_value"] = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify metadata is preserved
            assert loaded.metadata["test_key"] == "test_value"
            assert loaded.metadata["numeric_value"] == 42

    def test_image_pickle_preserves_enhanced_gray(self):
        """Test that enhanced grayscale is preserved during pickle save/load."""
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(rgb_array)

        # Modify enhanced gray
        modified_enh = img.detect_mat[:] + 10
        img.detect_mat[:] = modified_enh

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify enhanced gray is preserved
            assert np.array_equal(loaded.detect_mat[:], modified_enh)

    def test_image_pickle_preserves_objmap(self):
        """Test that object map is preserved during pickle save/load."""
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(rgb_array)

        # Create a simple object map
        objmap_data = np.zeros((100, 100), dtype=np.uint16)
        objmap_data[10:20, 10:20] = 1
        objmap_data[30:40, 30:40] = 2
        img.objmap[:] = objmap_data

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify object map is preserved
            assert np.array_equal(loaded.objmap[:], objmap_data)


class TestGridImagePickle:
    """Test suite for GridImage pickle save/load functionality."""

    def test_save_and_load_grid_image_with_default_finder(self):
        """Test saving and loading a GridImage with default CenteredAutoGridFinder."""
        rgb_array = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        grid_img = GridImage(rgb_array, nrows=8, ncols=12)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_grid.pkl"

            # Save and load
            grid_img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))  # Note: calling Image.load_pickle

            # Verify it's a GridImage (automatic type detection)
            assert isinstance(loaded, GridImage)

            # Verify grid configuration is preserved
            assert loaded.nrows == 8
            assert loaded.ncols == 12
            assert loaded.grid_finder is not None
            assert isinstance(loaded.grid_finder, CenteredAutoGridFinder)

            # Verify image data is preserved
            assert np.array_equal(loaded.rgb[:], grid_img.rgb[:])

    def test_save_and_load_grid_image_with_custom_finder(self):
        """Test saving and loading a GridImage with custom grid configuration."""
        rgb_array = np.random.randint(0, 256, (640, 960, 3), dtype=np.uint8)
        custom_finder = AutoGridFinder(nrows=16, ncols=24)
        grid_img = GridImage(rgb_array, grid_finder=custom_finder)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_grid_custom.pkl"

            # Save and load
            grid_img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify it's a GridImage
            assert isinstance(loaded, GridImage)

            # Verify custom grid configuration is preserved
            assert loaded.nrows == 16
            assert loaded.ncols == 24
            assert loaded.grid_finder is not None
            assert isinstance(loaded.grid_finder, AutoGridFinder)
            assert loaded.grid_finder.nrows == 16
            assert loaded.grid_finder.ncols == 24

    def test_grid_image_pickle_preserves_all_data(self):
        """Test that all data components are preserved in GridImage pickle."""
        rgb_array = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        grid_img = GridImage(rgb_array, nrows=8, ncols=12)

        # Add metadata
        grid_img.metadata["experiment_id"] = "plate_001"

        # Modify enhanced gray
        modified_enh = grid_img.detect_mat[:] + 5
        grid_img.detect_mat[:] = modified_enh

        # Create object map
        objmap_data = np.zeros((400, 600), dtype=np.uint16)
        objmap_data[50:100, 50:100] = 1
        grid_img.objmap[:] = objmap_data

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_grid_full.pkl"

            # Save and load
            grid_img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify type
            assert isinstance(loaded, GridImage)

            # Verify grid configuration
            assert loaded.nrows == 8
            assert loaded.ncols == 12

            # Verify all data components
            assert np.array_equal(loaded.rgb[:], grid_img.rgb[:])
            assert np.array_equal(loaded.detect_mat[:], modified_enh)
            assert np.array_equal(loaded.objmap[:], objmap_data)
            assert loaded.metadata["experiment_id"] == "plate_001"

    def test_grid_image_384_well_configuration(self):
        """Test GridImage pickle with 384-well plate configuration."""
        rgb_array = np.random.randint(0, 256, (800, 1200, 3), dtype=np.uint8)
        finder_384 = AutoGridFinder(nrows=16, ncols=24)
        grid_img = GridImage(rgb_array, grid_finder=finder_384)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "test_384well.pkl"

            # Save and load
            grid_img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify 384-well configuration
            assert isinstance(loaded, GridImage)
            assert loaded.nrows == 16
            assert loaded.ncols == 24
            assert loaded.grid_finder.nrows == 16
            assert loaded.grid_finder.ncols == 24


class TestPickleBackwardCompatibility:
    """Test backward compatibility with old pickle files."""

    def test_load_image_pickle_without_grid_finder(self):
        """Test loading an old-style pickle without grid_finder (should return Image)."""
        # Simulate an old pickle by manually creating one without grid_finder
        import pickle

        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image(rgb_array)

        # Manually create old-style pickle data (without grid_finder)
        old_style_data = {
            "_data.rgb"         : img.rgb[:],
            "_data.gray"        : img.gray[:],
            "_data.detect_mat"    : img.detect_mat[:],
            "objmap"            : img.objmap[:],
            "protected_metadata": img._metadata.protected,
            "public_metadata"   : img._metadata.public,
            # Deliberately omit grid_finder
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "old_style.pkl"

            # Save old-style pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(old_style_data, f)

            # Load should return Image (not GridImage)
            loaded = Image.load_pickle(str(pkl_path))

            # Verify it's an Image, not GridImage
            assert isinstance(loaded, Image)
            assert not isinstance(loaded, GridImage)

            # Verify data is correct
            assert np.array_equal(loaded.rgb[:], img.rgb[:])


class TestPickleEdgeCases:
    """Test edge cases and error conditions for pickle functionality."""

    def test_roundtrip_empty_rgb_array(self):
        """Test pickle with empty RGB array (grayscale-only image)."""
        from phenotypic.sdk_.exceptions_ import NoArrayError

        gray_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image(gray_array)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "grayscale_only.pkl"

            # Save and load
            img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify grayscale data is preserved
            assert isinstance(loaded, Image)
            # Accessing RGB on grayscale-only image should raise NoArrayError
            with pytest.raises(NoArrayError):
                _ = loaded.rgb[:]
            assert np.array_equal(loaded.gray[:], img.gray[:])

    def test_grid_image_from_grayscale(self):
        """Test GridImage pickle with grayscale input."""
        gray_array = np.random.randint(0, 256, (400, 600), dtype=np.uint8)
        grid_img = GridImage(gray_array, nrows=8, ncols=12)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "grid_grayscale.pkl"

            # Save and load
            grid_img.save2pickle(str(pkl_path))
            loaded = Image.load_pickle(str(pkl_path))

            # Verify it's a GridImage
            assert isinstance(loaded, GridImage)
            assert loaded.nrows == 8
            assert loaded.ncols == 12

            # Verify grayscale data
            assert np.array_equal(loaded.gray[:], grid_img.gray[:])

    def test_pickle_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent pickle file."""
        with pytest.raises(FileNotFoundError):
            Image.load_pickle("/nonexistent/path/to/file.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

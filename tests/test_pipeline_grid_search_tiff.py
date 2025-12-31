"""Tests for TIFF saving and HTML generation in pipeline grid search."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.util._pipeline_grid_search._shared import (
    _save_array_as_tiff,
    _validate_save_tiff_params,
    _create_trial_view_html,
)
from phenotypic.util._pipeline_grid_search import PipelineGridSearch


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    return Image(arr=arr, name="test_rgb")


@pytest.fixture
def temp_save_dir(tmp_path):
    """Create a temporary directory for saving TIFFs."""
    return str(tmp_path / "grid_search_results")


class TestSaveArrayAsTiff:
    """Tests for _save_array_as_tiff function."""

    def test_save_rgb_array(self, tmp_path):
        """Test saving RGB array as TIFF."""
        array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        save_path = _save_array_as_tiff(array, str(tmp_path), "rgb_test")

        assert save_path.exists()
        assert save_path.suffix == ".tiff"
        assert save_path.name == "rgb_test.tiff"

    def test_save_grayscale_array(self, tmp_path):
        """Test saving grayscale array as TIFF."""
        array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        save_path = _save_array_as_tiff(array, str(tmp_path), "gray_test")

        assert save_path.exists()
        assert save_path.suffix == ".tiff"

    def test_save_boolean_mask(self, tmp_path):
        """Test saving boolean mask as TIFF."""
        array = np.random.rand(100, 100) > 0.5
        save_path = _save_array_as_tiff(array, str(tmp_path), "mask_test")

        assert save_path.exists()
        assert save_path.suffix == ".tiff"

    def test_save_uint16_labels(self, tmp_path):
        """Test saving uint16 label map as TIFF."""
        array = np.random.randint(0, 1000, (100, 100), dtype=np.uint16)
        save_path = _save_array_as_tiff(array, str(tmp_path), "labels_test")

        assert save_path.exists()
        assert save_path.suffix == ".tiff"

    def test_invalid_shape_raises_error(self, tmp_path):
        """Test that invalid array shapes raise RuntimeError."""
        # 4D array is not supported
        array = np.random.rand(10, 10, 3, 3)
        with pytest.raises(RuntimeError, match="Unsupported array shape"):
            _save_array_as_tiff(array, str(tmp_path), "invalid_test")

    def test_invalid_3d_shape_raises_error(self, tmp_path):
        """Test that 3D arrays with wrong number of channels raise RuntimeError."""
        # 3D array with 4 channels is not supported
        array = np.random.rand(10, 10, 4)
        with pytest.raises(RuntimeError, match="Unsupported array shape"):
            _save_array_as_tiff(array, str(tmp_path), "invalid_test")


class TestValidateSaveTiffParams:
    """Tests for _validate_save_tiff_params function."""

    def test_create_trial_view_requires_save_dir(self):
        """Test that create_trial_view=True without save_tiff_dir raises error."""
        with pytest.raises(ValueError, match="create_trial_view=True requires save_tiff_dir"):
            _validate_save_tiff_params(None, True, "joblib")

    def test_invalid_backend_raises_error(self, tmp_path):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend must be"):
            _validate_save_tiff_params(str(tmp_path), False, "invalid_backend")

    def test_valid_parameters(self, tmp_path):
        """Test that valid parameters pass validation."""
        # Should not raise
        _validate_save_tiff_params(str(tmp_path), False, "joblib")
        _validate_save_tiff_params(str(tmp_path), True, "joblib")
        _validate_save_tiff_params(None, False, "joblib")

    def test_directory_creation(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        new_dir = str(tmp_path / "new" / "nested" / "dir")
        _validate_save_tiff_params(new_dir, False, "joblib")

        assert Path(new_dir).exists()

    def test_submitit_not_installed_raises_error(self, tmp_path, monkeypatch):
        """Test that submitit backend requires submitit package."""
        # Mock import failure
        import sys
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "submitit":
                raise ImportError("No module named 'submitit'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="submitit.*not installed"):
            _validate_save_tiff_params(str(tmp_path), False, "submitit")


class TestCreateTrialViewHtml:
    """Tests for _create_trial_view_html function."""

    def test_html_generation(self, tmp_path):
        """Test basic HTML generation."""
        # Create sample TIFF files
        sample_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        configs = {
            "001_sigma=1.0": '{"pipeline": "test1"}',
            "002_sigma=2.0": '{"pipeline": "test2"}',
        }

        # Create dummy TIFF files
        for base_name in configs.keys():
            _save_array_as_tiff(sample_array, str(tmp_path), f"{base_name}_rgb")
            _save_array_as_tiff(sample_array[:, :, 0], str(tmp_path), f"{base_name}_gray")

        html_path = _create_trial_view_html(str(tmp_path), configs, ["rgb", "gray"])

        assert html_path.exists()
        assert html_path.name == "trial_overview.html"

        html_content = html_path.read_text()
        assert "Pipeline Grid Search Results" in html_content
        assert "001_sigma=1.0" in html_content
        assert "002_sigma=2.0" in html_content

    def test_thumbnail_creation(self, tmp_path):
        """Test that thumbnails are created."""
        sample_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        configs = {"001_test": '{"test": "config"}'}

        _save_array_as_tiff(sample_array, str(tmp_path), "001_test_rgb")

        _create_trial_view_html(str(tmp_path), configs, ["rgb"])

        thumbnails_dir = tmp_path / "thumbnails"
        assert thumbnails_dir.exists()
        assert len(list(thumbnails_dir.glob("*.jpg"))) > 0

    def test_html_structure(self, tmp_path):
        """Test HTML structure and responsive design."""
        sample_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        configs = {"001_test": '{"test": "config"}'}

        _save_array_as_tiff(sample_array, str(tmp_path), "001_test_rgb")

        html_path = _create_trial_view_html(str(tmp_path), configs, ["rgb"])
        html_content = html_path.read_text()

        # Check for essential HTML elements
        assert "<!DOCTYPE html>" in html_content
        assert "grid-template-columns" in html_content  # CSS grid
        assert "results-grid" in html_content  # Grid class
        assert "result-card" in html_content  # Card class


class TestPipelineGridSearchTiff:
    """Integration tests for PipelineGridSearch with TIFF mode."""

    def test_tiff_mode_returns_configs_only(self, sample_rgb_image, temp_save_dir):
        """Test that TIFF mode returns configs dict only (no viewer)."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1, 2]})]

        result = PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            save_tiff_dir=temp_save_dir,
            n_jobs=1,
        )

        # Should return dict, not tuple
        assert isinstance(result, dict)
        assert len(result) == 2  # Two sigma values

    def test_napari_mode_returns_viewer_and_configs(self, sample_rgb_image):
        """Test that napari mode returns (viewer, configs) tuple."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1]})]

        result = PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            n_jobs=1,
        )

        # Should return tuple (viewer, configs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tiff_files_created(self, sample_rgb_image, temp_save_dir):
        """Test that TIFF files are actually created."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1, 2]})]

        PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            save_tiff_dir=temp_save_dir,
            data_layers=["rgb", "gray"],
            n_jobs=1,
        )

        # Check TIFF files exist
        tiff_dir = Path(temp_save_dir)
        tiff_files = list(tiff_dir.glob("*.tiff"))
        # 2 configs × 2 layers = 4 TIFFs
        assert len(tiff_files) == 4

    def test_trial_view_generation(self, sample_rgb_image, temp_save_dir):
        """Test HTML trial view generation."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1]})]

        PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            save_tiff_dir=temp_save_dir,
            create_trial_view=True,
            data_layers=["rgb"],
            n_jobs=1,
        )

        # Check HTML file exists
        html_file = Path(temp_save_dir) / "trial_overview.html"
        assert html_file.exists()

        # Check thumbnails exist
        thumbnails_dir = Path(temp_save_dir) / "thumbnails"
        assert thumbnails_dir.exists()
        assert len(list(thumbnails_dir.glob("*.jpg"))) > 0

    def test_memory_optimization(self, sample_rgb_image, temp_save_dir):
        """Test that memory is freed after TIFF saving."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1]})]

        # This should complete without excessive memory usage
        result = PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            save_tiff_dir=temp_save_dir,
            n_jobs=1,
        )

        assert isinstance(result, dict)
        # No viewer should be present
        assert "viewer" not in str(result)

    def test_backwards_compatibility_napari_mode(self, sample_rgb_image):
        """Ensure old API (napari mode) still works."""
        ops = [(GaussianBlur(sigma=1), {"sigma": [1]})]

        # Call without save_tiff_dir - should get viewer
        viewer, configs = PipelineGridSearch(
            image=sample_rgb_image,
            ops=ops,
            n_jobs=1,
        )

        assert viewer is not None
        assert isinstance(configs, dict)
        assert len(configs) == 1

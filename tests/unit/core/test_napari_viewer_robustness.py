"""Tests for napari viewer robustness when window is closed externally.

Verifies that both the base accessor and NapariLabelsMixin recover gracefully
when a viewer's Qt window has been closed or destroyed, and that the
DetectMatAccessor.preview_modes() method works correctly with its own
independent viewer.
"""

import pytest
import numpy as np
import napari

from phenotypic import Image
from phenotypic.detect import OtsuDetector
from phenotypic._core._image_parts.accessor_abstracts import _image_accessor_base


@pytest.fixture(autouse=True)
def cleanup_all_napari_viewers():
    """Clean up all napari viewers between tests."""
    yield
    # Clean global viewer
    viewer = _image_accessor_base._global_napari_viewer
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
    _image_accessor_base._global_napari_viewer = None
    # Clean detect modes viewer
    from phenotypic._core._image_parts.accessors import _detect_mat_accessor
    viewer = getattr(_detect_mat_accessor, "_detect_modes_viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
        _detect_mat_accessor._detect_modes_viewer = None


@pytest.fixture
def sample_image():
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image(arr)


class TestBaseAccessorViewerRobustness:
    """Verify base accessor napari() recovers from closed viewer."""

    def test_napari_reopens_after_close(self, sample_image):
        """Calling napari() after viewer.close() should create a new viewer."""
        viewer1 = sample_image.gray.napari()
        viewer1.close()
        # This should NOT raise - it should create a new viewer
        viewer2 = sample_image.gray.napari()
        assert viewer2 is not viewer1
        assert viewer2.window is not None

    def test_napari_reopens_after_window_destruction(self, sample_image):
        """Calling napari() after Qt window is destroyed should recover."""
        viewer1 = sample_image.gray.napari()
        # Simulate user closing the window (Qt object deletion)
        viewer1.close()
        _image_accessor_base._global_napari_viewer = viewer1  # stale ref
        # This should NOT raise RuntimeError
        viewer2 = sample_image.gray.napari()
        assert viewer2.window is not None


class TestLabelsMixinViewerRobustness:
    """Verify labels mixin napari() recovers from closed viewer."""

    @pytest.fixture
    def image_with_objects(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:40, 20:40] = 255
        arr[60:80, 60:80] = 255
        img = Image(arr)
        return OtsuDetector().apply(img)

    def test_labels_napari_reopens_after_close(self, image_with_objects):
        """Labels napari() should recover from closed viewer."""
        viewer1 = image_with_objects.objmap.napari()
        viewer1.close()
        viewer2 = image_with_objects.objmap.napari()
        assert viewer2 is not viewer1
        assert viewer2.window is not None


class TestPreviewModes:
    """Tests for DetectMatAccessor.preview_modes()."""

    @pytest.fixture
    def rgb_image(self):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image(arr)

    def test_returns_napari_viewer(self, rgb_image):
        viewer = rgb_image.detect_mat.preview_modes()
        assert isinstance(viewer, napari.Viewer)

    def test_has_layer_per_available_mode_plus_references(self, rgb_image):
        """Should have one layer per mode + current + rgb + gray references."""
        from phenotypic._core._image_parts.detection_modes import available_modes
        viewer = rgb_image.detect_mat.preview_modes()
        # modes + current detect_mat + rgb + gray = modes + 3
        expected_count = len(available_modes()) + 3
        assert len(viewer.layers) == expected_count

    def test_uses_separate_viewer_from_global(self, rgb_image):
        """preview_modes() must NOT use the global napari viewer."""
        global_viewer = rgb_image.gray.napari()
        modes_viewer = rgb_image.detect_mat.preview_modes()
        assert modes_viewer is not global_viewer

    def test_reopens_after_close(self, rgb_image):
        """Calling preview_modes() after closing should create new viewer."""
        viewer1 = rgb_image.detect_mat.preview_modes()
        viewer1.close()
        viewer2 = rgb_image.detect_mat.preview_modes()
        assert viewer2 is not viewer1

    def test_reset_creates_fresh_viewer(self, rgb_image):
        viewer1 = rgb_image.detect_mat.preview_modes()
        viewer2 = rgb_image.detect_mat.preview_modes(reset=True)
        assert viewer2 is not viewer1

    def test_current_detect_mat_layer_exists(self, rgb_image):
        """Should include a layer for the current (possibly enhanced) detect_mat."""
        viewer = rgb_image.detect_mat.preview_modes()
        layer_names = [layer.name for layer in viewer.layers]
        assert any("current" in name.lower() for name in layer_names)

    def test_rgb_reference_layer_exists(self, rgb_image):
        """Should include RGB reference layer."""
        viewer = rgb_image.detect_mat.preview_modes()
        layer_names = [layer.name for layer in viewer.layers]
        assert "rgb" in layer_names

    def test_gray_reference_layer_exists(self, rgb_image):
        """Should include grayscale reference layer."""
        viewer = rgb_image.detect_mat.preview_modes()
        layer_names = [layer.name for layer in viewer.layers]
        assert "[ref] gray" in layer_names

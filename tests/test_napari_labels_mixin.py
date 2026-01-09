"""Tests for NapariLabelsMixin functionality.

Tests verify that ObjectMap and ObjectMask use napari's labels layer API
instead of image layers, with custom parameters for visualization control.
"""

import pytest
import numpy as np
import napari

from phenotypic import Image
from phenotypic.detect import OtsuDetector
from phenotypic._core._image_parts.accessor_abstracts import _image_accessor_base


@pytest.fixture(autouse=True)
def cleanup_napari_viewer():
    """Clean up global napari viewer between tests."""
    yield
    # After each test, close viewer and reset global reference
    viewer = _image_accessor_base._global_napari_viewer
    if viewer is not None:
        try:
            if hasattr(viewer, "window") and viewer.window is not None:
                viewer.close()
        except Exception:
            pass  # Viewer already closed or deletion failed
    _image_accessor_base._global_napari_viewer = None


@pytest.fixture
def sample_image_with_objects():
    """Create sample image with detected objects for testing."""
    # Create synthetic image with two distinct objects
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    # Object 1 (top-left)
    arr[20:40, 20:40] = 255
    # Object 2 (bottom-right)
    arr[60:80, 60:80] = 255

    img = Image(arr)
    detector = OtsuDetector()
    img = detector.apply(img)
    return img


@pytest.fixture
def empty_image():
    """Create image with no detected objects."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image(arr)
    return img


class TestNapariLabelsLayerType:
    """Test that labels layers (not image layers) are created."""

    def test_objmap_creates_labels_layer(self, sample_image_with_objects):
        """Verify objmap.napari() creates a labels layer, not image layer."""
        img = sample_image_with_objects
        viewer = img.objmap.napari()

        # Check that a labels layer was created
        layer_name = f"objmap_{img.name}"
        assert layer_name in viewer.layers

        # Verify it's a Labels layer, not Image layer
        layer = viewer.layers[layer_name]
        assert isinstance(layer, napari.layers.Labels)
        assert not isinstance(layer, napari.layers.Image)

    def test_objmask_creates_labels_layer(self, sample_image_with_objects):
        """Verify objmask.napari() creates a labels layer."""
        img = sample_image_with_objects
        viewer = img.objmask.napari()

        layer_name = f"objmask_{img.name}"
        assert layer_name in viewer.layers

        layer = viewer.layers[layer_name]
        assert isinstance(layer, napari.layers.Labels)
        assert not isinstance(layer, napari.layers.Image)

        viewer.close()


class TestNapariLabelsParameters:
    """Test new labels-specific parameters."""

    def test_custom_colormap(self, sample_image_with_objects):
        """Test custom colormap parameter."""
        img = sample_image_with_objects
        cmap = {1: [1.0, 0, 0], 2: [0, 1.0, 0]}

        viewer = img.objmap.napari(colormap=cmap)
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        # Verify colormap was set
        assert layer.colormap is not None

        viewer.close()

    def test_opacity_parameter(self, sample_image_with_objects):
        """Test opacity parameter."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(opacity=0.5)
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        assert layer.opacity == 0.5

        viewer.close()

    def test_opacity_default(self, sample_image_with_objects):
        """Test default opacity value is 0.7."""
        img = sample_image_with_objects

        viewer = img.objmap.napari()
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        assert layer.opacity == 0.7

        viewer.close()

    def test_contour_parameter(self, sample_image_with_objects):
        """Test contour parameter."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(contour=2)
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        assert layer.contour == 2

        viewer.close()

    def test_contour_default(self, sample_image_with_objects):
        """Test default contour value is 0 (filled)."""
        img = sample_image_with_objects

        viewer = img.objmap.napari()
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        assert layer.contour == 0

        viewer.close()

    def test_colormap_opacity_contour_combined(self, sample_image_with_objects):
        """Test all three parameters together."""
        img = sample_image_with_objects
        cmap = {1: [1.0, 0, 0], 2: [0, 1.0, 0]}

        viewer = img.objmap.napari(colormap=cmap, opacity=0.6, contour=1)
        layer_name = f"objmap_{img.name}"
        layer = viewer.layers[layer_name]

        assert layer.colormap is not None
        assert layer.opacity == 0.6
        assert layer.contour == 1

        viewer.close()


class TestNapariLabelsParameterValidation:
    """Test parameter validation."""

    def test_invalid_opacity_too_high(self, sample_image_with_objects):
        """Test that opacity > 1.0 raises ValueError."""
        img = sample_image_with_objects

        with pytest.raises(ValueError, match="opacity must be in range"):
            img.objmap.napari(opacity=1.5)

    def test_invalid_opacity_negative(self, sample_image_with_objects):
        """Test that negative opacity raises ValueError."""
        img = sample_image_with_objects

        with pytest.raises(ValueError, match="opacity must be in range"):
            img.objmap.napari(opacity=-0.1)

    def test_valid_opacity_boundaries(self, sample_image_with_objects):
        """Test that opacity boundaries (0.0, 1.0) are valid."""
        img = sample_image_with_objects

        # Should not raise with opacity 0.0
        viewer = img.objmap.napari(opacity=0.0)
        assert viewer is not None

        # Reset and test with opacity 1.0
        viewer = img.objmap.napari(opacity=1.0, reset=True)
        assert viewer is not None

    def test_invalid_contour_negative(self, sample_image_with_objects):
        """Test that negative contour raises ValueError."""
        img = sample_image_with_objects

        with pytest.raises(ValueError, match="contour must be >= 0"):
            img.objmap.napari(contour=-1)

    def test_valid_contour_zero(self, sample_image_with_objects):
        """Test that contour=0 is valid."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(contour=0)
        layer_name = f"objmap_{img.name}"
        assert layer_name in viewer.layers
        viewer.close()


class TestNapariLabelsLayerManagement:
    """Test layer creation, update, and naming."""

    def test_layer_update(self, sample_image_with_objects):
        """Test that calling napari() twice updates the existing layer."""
        img = sample_image_with_objects

        # First call creates layer
        viewer = img.objmap.napari(opacity=0.5)
        layer_name = f"objmap_{img.name}"
        initial_opacity = viewer.layers[layer_name].opacity

        # Second call updates layer
        viewer = img.objmap.napari(opacity=0.8, contour=1)

        # Should still be only one layer with this name
        layer_names = [layer.name for layer in viewer.layers]
        assert layer_names.count(layer_name) == 1

        # Properties should be updated
        layer = viewer.layers[layer_name]
        assert layer.opacity == 0.8
        assert layer.contour == 1
        assert layer.opacity != initial_opacity

        viewer.close()

    def test_custom_name_parameter(self, sample_image_with_objects):
        """Test custom name parameter."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(name="custom_name")
        layer_name = "objmap_custom_name"

        assert layer_name in viewer.layers

        viewer.close()

    def test_custom_name_with_parameters(self, sample_image_with_objects):
        """Test custom name combined with other parameters."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(name="v2", opacity=0.4)
        layer_name = "objmap_v2"

        assert layer_name in viewer.layers
        assert viewer.layers[layer_name].opacity == 0.4

        viewer.close()

    def test_reset_parameter(self, sample_image_with_objects):
        """Test reset parameter creates fresh viewer."""
        img = sample_image_with_objects

        # Create viewer with objmap layer
        viewer1 = img.objmap.napari()
        layer_name1 = f"objmap_{img.name}"
        assert layer_name1 in viewer1.layers

        # Reset and create fresh viewer
        viewer2 = img.objmap.napari(reset=True)

        # Should be functional
        assert viewer2 is not None

        viewer2.close()


class TestNapariLabelsMultipleAccessors:
    """Test mixing different accessor types in same viewer."""

    def test_mixed_layer_types(self, sample_image_with_objects):
        """Test adding both image and labels layers to same viewer."""
        img = sample_image_with_objects

        # Add image layer (grayscale)
        viewer = img.gray.napari()
        gray_layer_name = f"gray_{img.name}"
        assert gray_layer_name in viewer.layers
        assert isinstance(viewer.layers[gray_layer_name], napari.layers.Image)

        # Add labels layer (objmap)
        viewer = img.objmap.napari()
        objmap_layer_name = f"objmap_{img.name}"
        assert objmap_layer_name in viewer.layers
        assert isinstance(viewer.layers[objmap_layer_name], napari.layers.Labels)

        # Both layers should coexist
        assert len(viewer.layers) >= 2

        viewer.close()

    def test_objmap_and_objmask_together(self, sample_image_with_objects):
        """Test visualizing both object map and mask in same viewer."""
        img = sample_image_with_objects

        # Add object map
        viewer = img.objmap.napari(name="map")
        map_layer_name = f"objmap_map"
        assert map_layer_name in viewer.layers

        # Add object mask
        viewer = img.objmask.napari(name="mask")
        mask_layer_name = f"objmask_mask"
        assert mask_layer_name in viewer.layers

        # Both should be labels layers
        assert isinstance(viewer.layers[map_layer_name], napari.layers.Labels)
        assert isinstance(viewer.layers[mask_layer_name], napari.layers.Labels)

        viewer.close()

    def test_multiple_masks_different_images(self):
        """Test visualizing masks from different images."""
        # Create two different images
        arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
        arr1[20:40, 20:40] = 255
        img1 = Image(arr1, name="image1")
        detector = OtsuDetector()
        img1 = detector.apply(img1)

        arr2 = np.zeros((100, 100, 3), dtype=np.uint8)
        arr2[60:80, 60:80] = 255
        img2 = Image(arr2, name="image2")
        img2 = detector.apply(img2)

        # Visualize both
        viewer = img1.objmap.napari()
        viewer = img2.objmap.napari()

        # Both should be in viewer
        layer_names = [layer.name for layer in viewer.layers]
        assert "objmap_image1" in layer_names
        assert "objmap_image2" in layer_names

        viewer.close()


class TestNapariLabelsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_object_map(self, empty_image):
        """Test napari() works with empty object map (all zeros)."""
        img = empty_image

        viewer = img.objmap.napari()
        layer_name = f"objmap_{img.name}"

        assert layer_name in viewer.layers
        layer = viewer.layers[layer_name]
        assert isinstance(layer, napari.layers.Labels)

        # Data should be all zeros
        assert np.all(layer.data == 0)

        viewer.close()

    def test_empty_object_mask(self, empty_image):
        """Test napari() works with empty object mask."""
        img = empty_image

        viewer = img.objmask.napari()
        layer_name = f"objmask_{img.name}"

        assert layer_name in viewer.layers
        layer = viewer.layers[layer_name]
        assert isinstance(layer, napari.layers.Labels)

        viewer.close()

    def test_colormap_none(self, sample_image_with_objects):
        """Test that colormap=None uses napari defaults."""
        img = sample_image_with_objects

        viewer = img.objmap.napari(colormap=None)
        layer_name = f"objmap_{img.name}"

        # Should work without errors
        assert layer_name in viewer.layers

        viewer.close()


class TestNapariLabelsBackwardCompatibility:
    """Test that existing code patterns still work."""

    def test_no_parameters(self, sample_image_with_objects):
        """Test that existing code with no parameters still works."""
        img = sample_image_with_objects

        # Old pattern: no parameters
        viewer = img.objmap.napari()
        assert viewer is not None

        viewer.close()

    def test_name_only(self, sample_image_with_objects):
        """Test that existing name-only pattern still works."""
        img = sample_image_with_objects

        # Old pattern: custom name only
        viewer = img.objmask.napari(name="mask_v2")
        assert "objmask_mask_v2" in viewer.layers

        viewer.close()

    def test_reset_only(self, sample_image_with_objects):
        """Test that existing reset-only pattern still works."""
        img = sample_image_with_objects

        # Old pattern: reset only
        viewer = img.objmap.napari(reset=True)
        assert viewer is not None

        viewer.close()

    def test_name_and_reset(self, sample_image_with_objects):
        """Test that existing name + reset pattern still works."""
        img = sample_image_with_objects

        # Old pattern: name and reset
        viewer = img.objmap.napari(name="v1", reset=True)
        assert "objmap_v1" in viewer.layers

        viewer.close()


class TestNapariLabelsMROVerification:
    """Test that Method Resolution Order is correct."""

    def test_objmap_mro(self):
        """Verify ObjectMap MRO includes NapariLabelsMixin first."""
        from phenotypic._core._image_parts.accessors._objmap_accessor import ObjectMap
        from phenotypic._core._image_parts.accessor_abstracts._napari_labels_mixin import (
            NapariLabelsMixin,
        )

        # Mixin should be in MRO
        assert NapariLabelsMixin in ObjectMap.__mro__

        # Mixin should be near the beginning (before base class methods)
        mixin_index = ObjectMap.__mro__.index(NapariLabelsMixin)
        assert mixin_index < 3  # Should be in first few positions

    def test_objmask_mro(self):
        """Verify ObjectMask MRO includes NapariLabelsMixin first."""
        from phenotypic._core._image_parts.accessors._objmask_accessor import ObjectMask
        from phenotypic._core._image_parts.accessor_abstracts._napari_labels_mixin import (
            NapariLabelsMixin,
        )

        # Mixin should be in MRO
        assert NapariLabelsMixin in ObjectMask.__mro__

        # Mixin should be near the beginning
        mixin_index = ObjectMask.__mro__.index(NapariLabelsMixin)
        assert mixin_index < 3

    def test_napari_method_override(self, sample_image_with_objects):
        """Verify that napari() method from mixin is being used."""
        img = sample_image_with_objects

        # Call napari with labels-specific parameter
        # This should work because mixin's napari() accepts 'contour'
        viewer = img.objmap.napari(contour=2)

        # Verify it's a labels layer (only possible with mixin)
        layer_name = f"objmap_{img.name}"
        assert isinstance(viewer.layers[layer_name], napari.layers.Labels)

        viewer.close()

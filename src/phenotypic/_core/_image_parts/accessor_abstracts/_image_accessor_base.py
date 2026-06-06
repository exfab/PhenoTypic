from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

from phenotypic.tools_.funcs_ import normalize_rgb_bitdepth

from ._image_accessor_base_parents import AccessorDashHandler

if TYPE_CHECKING:
    import napari

_HAS_NAPARI = importlib.util.find_spec("napari") is not None

# Global napari viewer instance for persistent Jupyter notebook workflows
_global_napari_viewer: napari.Viewer | None = None


def _viewer_is_alive(viewer: napari.Viewer | None) -> bool:
    """Return True if *viewer* is open and its Qt window still exists."""
    if viewer is None:
        return False
    try:
        window = getattr(viewer, "window", None)
        if window is None:
            return False
        # Access the underlying Qt widget to verify it hasn't been deleted.
        # After viewer.close(), the Qt C++ object may be garbage-collected,
        # causing RuntimeError or AttributeError on attribute access.
        qt_window = window._qt_window
        return qt_window is not None and qt_window.isVisible()
    except (RuntimeError, AttributeError):
        # Qt C++ object has been deleted (user closed window)
        return False


class ImageAccessorBase(AccessorDashHandler):
    """
    Provides an abstract base class for image accessor operations.

    The `ImageAccessorBase` class serves as a foundational abstract base class
    to standardize the handling and manipulation of image data. It provides
    attributes and methods for image loading, property enforcement, and shape
    management for image data, supporting both access and processing functionalities.

    This class is particularly useful for processing and analyzing images of
    microbe colonies grown on solid media agar. By ensuring consistent handling
    of image formats, metadata validation, and structured array management, this
    class helps streamline image analysis workflows, maintain metadata integrity,
    and enable reproducible results.

    Attributes:
        _root_image (Image): The root image object from which operations derive.
            Modifying this can change the basis of calculations or operations
            performed within accessor methods. For example, a grayscale `_root_image`
            might yield entirely different results when compared to an RGB image
            for colony segmentation or measurement.
    """

    def napari(
        self,
        name: str | None = None,
        reset: bool = False,
        *,
        viewer: napari.Viewer | None = None,
        layer_name: str | None = None,
    ) -> napari.Viewer:
        """Add image to a persistent global napari viewer for Jupyter workflows.

        Creates or reuses a single napari viewer instance that persists across
        multiple method calls. This is particularly useful in Jupyter notebooks
        where multiple accessors can contribute layers to the same viewer,
        enabling interactive comparison of different image transformations
        (e.g., grayscale, RGB, binary masks) on the same data.

        The viewer is automatically displayed in Jupyter environments and
        recreated if it has been closed externally.

        Args:
            name: Optional custom name for the image layer. If provided, the layer
                will be named ``{accessor}_{name}``. If not provided, defaults to
                using the image's name attribute.
            reset: If True, closes the current napari viewer and creates a fresh
                one. This is useful for starting a new visualization session
                without lingering layers from previous calls. Defaults to False.
            viewer: Optional external napari viewer instance to use instead of the
                global viewer. When provided, global viewer management (creation,
                reset, smart-grid installation) is bypassed entirely. Defaults to
                None.
            layer_name: Optional full layer name to use instead of the auto-generated
                ``{accessor}_{image_name}`` pattern. Defaults to None.

        Returns:
            napari.Viewer: The global napari viewer instance with the current
                image added as a new layer.

        Raises:
            ImportError: If napari is not installed. Install with
                ``pip install phenotypic[napari]``.

        Examples:
            View multiple image transformations in one viewer:

            >>> from phenotypic import Image
            >>> img = Image(arr)
            >>> # Add grayscale version to viewer
            >>> viewer = img.gray.napari()
            >>> # Add RGB version to same viewer
            >>> viewer = img.rgb.napari()
            >>> # Add binary segmentation with custom name
            >>> viewer = img.objmask.napari(name="segmentation_v2")

            Using custom names for comparison:

            >>> viewer = img.gray.napari(name="raw_grayscale")
            >>> viewer = img.objmask.napari(name="segmentation_v2")

            Resetting the viewer for a fresh session:

            >>> viewer = img.gray.napari()
            >>> viewer = img.rgb.napari()  # Same viewer, added layer
            >>> viewer = img.gray.napari(reset=True)  # Fresh viewer, old layers gone

        Note:
            Layers are named using the pattern ``{accessor}_{image_name}`` to
            ensure descriptive identification. If a layer with the same name
            already exists, it is replaced with the new image data. This allows
            for easy updates and comparison of different processing stages.
        """
        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[napari]"
            )
        import napari as _napari

        # Determine active viewer
        if viewer is not None:
            active_viewer = viewer
        else:
            global _global_napari_viewer

            # Reset viewer if requested
            if reset and _viewer_is_alive(_global_napari_viewer):
                _global_napari_viewer.close()
                _global_napari_viewer = None

            # Create new viewer if needed
            if not _viewer_is_alive(_global_napari_viewer):
                _global_napari_viewer = _napari.Viewer()
                from phenotypic.gui._smart_grid import install_smart_grid
                install_smart_grid(_global_napari_viewer)

            active_viewer = _global_napari_viewer

        # Generate descriptive layer name
        if layer_name is not None:
            resolved_layer_name = layer_name
        elif name is not None:
            image_name = name
            resolved_layer_name = f"{self._accessor_property_name}_{image_name}"
        else:
            image_name = getattr(self._root_image, "name", "image")
            resolved_layer_name = f"{self._accessor_property_name}_{image_name}"

        # Replace layer if it exists, otherwise add new layer

        imdata = self._subject_arr
        if imdata.ndim == 3:
            imdata = normalize_rgb_bitdepth(imdata)
        try:
            existing_layer = active_viewer.layers[resolved_layer_name]
            existing_layer.data = imdata
        except KeyError:
            active_viewer.add_image(
                imdata, name=resolved_layer_name,
                contrast_limits=(0, int(np.iinfo(imdata.dtype).max))
                    if np.issubdtype(imdata.dtype, np.integer)
                    else (float(imdata.min()), float(imdata.max())),
                gamma=1.0,
            )

        return active_viewer

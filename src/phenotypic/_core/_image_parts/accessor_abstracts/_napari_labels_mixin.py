"""Mixin class for napari labels layer visualization.

This module provides NapariLabelsMixin, a mixin class that overrides the napari()
method from ImageAccessorBase to use napari's labels layer API instead of image
layers. This is specifically designed for ObjectMap and ObjectMask accessors where
data represents discrete labeled regions rather than continuous intensity values.

Labels vs Image Layers
----------------------
Napari distinguishes between two primary layer types for 2D data:

- **Image layers**: For continuous/intensity data (grayscale, RGB, fluorescence)
  - Rendered with smooth intensity gradients
  - Colormap maps intensity values to colors
  - Best for: grayscale, RGB, detection matrix

- **Labels layers**: For discrete/categorical data (segmentation masks, object IDs)
  - Each unique label gets a distinct color
  - Supports contour visualization (outline mode)
  - Optimized for object identification and counting
  - Best for: object maps, binary masks, segmentation results

Usage
-----
This mixin is applied via multiple inheritance to ObjectMap and ObjectMask classes.
The mixin MUST be placed FIRST in the inheritance list to ensure its napari()
method overrides the base class implementation through Python's Method Resolution Order (MRO):

    class ObjectMap(NapariLabelsMixin, SingleChannelAccessor):
        ...

    class ObjectMask(NapariLabelsMixin, SingleChannelAccessor):
        ...

Incorrect usage (mixin second) will NOT override the base class method:

    class ObjectMap(SingleChannelAccessor, NapariLabelsMixin):  # Wrong!
        ...

Examples
--------
Basic visualization of detected colonies:

    >>> from phenotypic import Image
    >>> from phenotypic.detect import OtsuDetector
    >>> img = Image.imread("colonies.jpg")
    >>> detector = OtsuDetector()
    >>> img = detector.apply(img)
    >>>
    >>> # View object map as labels layer (each colony gets distinct color)
    >>> viewer = img.objmap.napari()

Customizing labels appearance with opacity and contours:

    >>> # Semi-transparent labels layer with contours only
    >>> viewer = img.objmap.napari(opacity=0.5, contour=2)
    >>>
    >>> # Custom colormap for specific colony classification
    >>> cmap = {1: [1.0, 0, 0], 2: [0, 1.0, 0], 3: [0, 0, 1.0]}
    >>> viewer = img.objmap.napari(colormap=cmap)

Comparing masks and maps in a single viewer:

    >>> # Add grayscale base image
    >>> viewer = img.gray.napari()
    >>>
    >>> # Overlay binary mask with transparency
    >>> viewer = img.objmask.napari(opacity=0.4)
    >>>
    >>> # Add full object map with contours only
    >>> viewer = img.objmap.napari(name="boundaries", contour=1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import napari

from phenotypic._core._image_parts.accessor_abstracts import _image_accessor_base

if TYPE_CHECKING:
    pass


class NapariLabelsMixin:
    """Mixin to override napari() method for labels-based visualization.

    This mixin replaces the ImageAccessorBase.napari() implementation to use
    napari's add_labels() API instead of add_image(). This is appropriate for
    accessors representing discrete labeled regions (object maps, binary masks)
    rather than continuous intensity data.

    The mixin must be placed FIRST in the inheritance list to ensure proper
    method resolution order (MRO):

        class ObjectMap(NapariLabelsMixin, SingleChannelAccessor):  # Correct
        class ObjectMap(SingleChannelAccessor, NapariLabelsMixin):  # Wrong!

    Attributes:
        None. This mixin relies on attributes provided by the accessor classes:
        - _accessor_property_name: Layer name prefix (from child class)
        - _root_image: Parent Image object (from ImageAccessorBase)
        - _subject_arr: Array data to visualize (from accessor implementation)

    Notes:
        Labels layers are specifically designed for discrete/categorical data where
        each unique integer value represents a distinct object or region. This is
        fundamentally different from image layers which handle continuous intensity
        data. Labels layers provide:

        - Automatic distinct coloring per label
        - Contour visualization mode
        - Optimized rendering for segmentation data
        - Better interactivity for object selection

        For colony analysis, this means each detected colony gets a unique color,
        making visual QC and counting much easier than grayscale intensity display.
    """

    def napari(
        self,
        name: str | None = None,
        reset: bool = False,
        colormap: dict | None = None,
        opacity: float = 0.7,
        contour: int = 0,
    ) -> napari.Viewer:
        """Add labeled regions to a persistent global napari viewer.

        Creates or reuses a single napari viewer instance with a labels layer for
        visualizing discrete labeled regions. Unlike the base class image layer
        implementation, this uses napari's labels API which assigns distinct colors
        to each unique label value, making it ideal for visualizing segmented
        colonies, object maps, and binary masks.

        The viewer persists across multiple method calls, allowing comparison of
        different processing stages and data types in the same visualization window.

        Args:
            name: Optional custom name for the labels layer. If provided, the layer
                will be named ``{accessor}_{name}``. If not provided, defaults to
                using the image's name attribute. Defaults to None.
            reset: If True, closes the current napari viewer and creates a fresh
                one. This is useful for starting a new visualization session
                without lingering layers from previous calls. Defaults to False.
            colormap: Optional dictionary mapping label values to RGB colors.
                Keys are integer label IDs, values are RGB tuples/lists with
                values in [0, 1] range. If None, napari uses its default label
                colormap which assigns distinct colors automatically.
                Example: ``{1: [1.0, 0, 0], 2: [0, 1.0, 0]}`` for red/green colonies.
                Defaults to None.
            opacity: Layer opacity from 0.0 (fully transparent) to 1.0 (fully opaque).
                This controls the visibility of the labels layer. Values outside this
                range will raise ValueError. Defaults to 0.7.
            contour: Contour thickness in pixels. When > 0, renders only the outline
                of each labeled region rather than filled regions. Useful for overlaying
                object boundaries on other layers. Must be >= 0. Defaults to 0 (filled).

        Returns:
            napari.Viewer: The global napari viewer instance with the labeled regions
                added as a labels layer.

        Raises:
            ValueError: If opacity is not in [0.0, 1.0] range or contour is negative.

        Examples:
            Basic labels visualization:

            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> img = Image.imread("colonies.jpg")
            >>> detector = OtsuDetector()
            >>> img = detector.apply(img)
            >>>
            >>> # View detected colonies as labels
            >>> viewer = img.objmap.napari()

            Customizing labels appearance:

            >>> # Semi-transparent labels with contours
            >>> viewer = img.objmap.napari(opacity=0.5, contour=2)
            >>>
            >>> # Custom color mapping for specific colonies
            >>> cmap = {
            ...     1: [1.0, 0, 0],      # Red for colony 1
            ...     2: [0, 1.0, 0],      # Green for colony 2
            ...     3: [0, 0, 1.0],      # Blue for colony 3
            ... }
            >>> viewer = img.objmap.napari(colormap=cmap)

            Comparing masks and maps in same viewer:

            >>> # Add grayscale base image
            >>> viewer = img.gray.napari()
            >>>
            >>> # Overlay binary mask with transparency
            >>> viewer = img.objmask.napari(opacity=0.4)
            >>>
            >>> # Add full object map with contours only
            >>> viewer = img.objmap.napari(name="boundaries", contour=1)

            Using reset for fresh visualization sessions:

            >>> viewer = img.objmap.napari()
            >>> # ... do some analysis ...
            >>> # Start fresh without old layers
            >>> viewer = img.objmap.napari(reset=True)

        Note:
            Labels layers are specifically designed for discrete/categorical data where
            each unique integer value represents a distinct object or region. This is
            fundamentally different from image layers which handle continuous intensity
            data. Labels layers provide:

            - Automatic distinct coloring per label
            - Contour visualization mode
            - Optimized rendering for segmentation data
            - Better interactivity for object selection

            For colony analysis, this means each detected colony gets a unique color,
            making visual QC and counting much easier than grayscale intensity display.
        """
        # Validate parameters early for clearer error messages
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be in range [0.0, 1.0], got {opacity}")
        if contour < 0:
            raise ValueError(f"contour must be >= 0, got {contour}")

        # Access the global viewer through module reference
        # This ensures we share the same viewer instance across all accessors
        viewer = _image_accessor_base._global_napari_viewer

        # Reset viewer if requested
        if reset and viewer is not None:
            if hasattr(viewer, "window") and viewer.window is not None:
                viewer.close()
            _image_accessor_base._global_napari_viewer = None
            viewer = None

        # Check if viewer exists and is still valid (window open)
        if (
            viewer is None
            or not hasattr(viewer, "window")
            or viewer.window is None
        ):
            viewer = napari.Viewer()
            _image_accessor_base._global_napari_viewer = viewer

        # Generate descriptive layer name (same pattern as base class)
        if name is not None:
            image_name = name
        else:
            image_name = getattr(self._root_image, "name", "image")
        layer_name = f"{self._accessor_property_name}_{image_name}"

        # Get label data - no RGB normalization needed for integer labels
        label_data = self._subject_arr

        # Ensure data is integer type for labels layer
        if not np.issubdtype(label_data.dtype, np.integer):
            label_data = label_data.astype(np.uint16)

        # Replace layer if it exists, otherwise add new labels layer
        try:
            existing_layer = viewer.layers[layer_name]
            # Update existing labels layer data
            existing_layer.data = label_data
            # Update visual properties
            if colormap is not None:
                existing_layer.colormap = colormap
            existing_layer.opacity = opacity
            existing_layer.contour = contour
        except KeyError:
            # Add new labels layer with specified properties
            viewer.add_labels(
                label_data,
                name=layer_name,
                colormap=colormap,
                opacity=opacity,
            )
            # Set contour property after layer creation
            viewer.layers[layer_name].contour = contour

        return viewer

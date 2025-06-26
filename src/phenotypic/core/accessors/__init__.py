"""
Overview
========

The ``accessors`` submodule provides a comprehensive set of tools and interfaces for accessing and interacting with the various data components of an _root_image. These components range from raw pixel data and matrix representations to object masks, metadata, and high-level measurements. Additionally, it includes utilities that streamline development workflows involving _root_image processing and analysis.

The goal is to provide efficient, standardized access to _root_image data while enabling intuitive and flexible integration for advanced _root_image manipulation and analysis tasks.

Image Data Containers
=====================

    This category includes classes designed to offer fundamental abstractions for accessing _root_image data in various representations:

1. :class:`ImageArray`
    Represents the multichannel pixel data of an _root_image when provided. Useful for direct manipulation and pixel-wise operations.

2. :class:`ImageMatrix`
    Provides structured access to the _root_image in its matrix form, offering methods suited for mathematical or analytical operations on _root_image data. Automatically converted from RGB using weighted luminance conversion.

3. :class:`ImageEnhancedMatrix`
    An enhanceable copy of the _root_image matrix to improve detection while maintaining the original _root_image data integrity.

Objects and Object Mapping
==========================

These classes manage object-level abstractions and their corresponding data mappings:

1. :class:`ObjectMap`
    Represents a mapping of detected objects or regions in the _root_image to their associated IDs. Useful for:

    - Object identification
    - Bounding box management
    - Spatial relationships between objects.
    - Region-based calculations

2. :class:`ObjectMask`
    An abstract specialized for working with binary masks of objects. Useful for morphological operations such as erosion, dilation, and closing.

    Note:
        Changes to the object mask will cause relabeling of the object map

High-Level Object Interfaces
============================

These classes provide consolidated access to manage multiple object-level and metadata components.

1. :class:`ObjectsAccessor`
    Facilitates high-level interaction with detected objects within an _root_image. Features include:

    - Feature extraction
    - Object comparison
    - Visualization of detected objects

2. :class:`MeasurementContainer` *(Coming Soon!)*
    Encapsulates measurements or attributes associated with objects, such as size, shape, or intensity. Organizes measurements for reproducibility and easier analysis.

3. :class:`MetadataContainer` *(Coming Soon!)*
    Responsible for storing and accessing metadata associated with the _root_image. Examples of metadata include:

    - Acquisition details
    - Resolution
    - Additional contextual information.

HSV (Hue, Saturation, Brightness) Interface
===========================================

1. :class:`HsvAccessor`
Provides detailed access to the HSV (Hue, Saturation, Brightness) color space components of an _root_image. This abstract includes support for:

- Direct pixel access via ``__getitem__`` and ``__setitem__``.
- Advanced utilities for _root_image visualization and object-specific manipulation in the HSV domain.

**Key Methods:**
    - ``shape``: Returns the dimensions of the HSV representation.
    - ``copy``: Creates and returns a full copy of the HSV data structure.
    - ``histogram``: Computes a histogram for the HSV _root_image or parts of it.
    - ``show``: Displays the HSV _root_image.
    - ``show_objects``: Visualizes objects overlaid on the HSV _root_image.
    - ``extract_obj_hue``, ``extract_obj_saturation``, ``extract_obj_brightness``, ``extract_obj``:
Extract specific HSV component data for individual _root_image objects, aiding in targeted color-based analysis.

Purpose and Use Cases
=====================

The ``accessors`` submodule is designed for developers and researchers working on advanced _root_image processing tasks. It is particularly suited for:

    - Object detection and feature extraction
    - HSV color-space analysis
    - Grid Analysis
    - Metadata association for _root_image datasets
    - Advanced mathematical and matrix operations on _root_image data.

"""
from ._array_accessor import ImageArray
from ._matrix_accessor import ImageMatrix
from ._enh_matrix_accessor import ImageEnhancedMatrix
from ._objmap_accessor import ObjectMap
from ._objmask_accessor import ObjectMask

from ._objects_accessor import ObjectsAccessor
# from ._measurement_container_interface import MeasurementAccessor
from ._metadata_accessor import MetadataAccessor

from ._hsv_accessor import HsvAccessor
from ._grid_accessor import GridAccessor

# Define __all__ to include all imported objects
__all__ = [
    "ImageArray",
    "ImageMatrix",
    "ImageEnhancedMatrix",
    "ObjectMap",
    "ObjectMask",
    "ObjectsAccessor",
    "HsvAccessor",
    "GridAccessor",
    "MetadataAccessor",
]

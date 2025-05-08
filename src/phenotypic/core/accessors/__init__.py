"""
Overview
========

The ``accessors`` submodule provides a comprehensive set of tools and interfaces for accessing and interacting with the various data components of an image. These components range from raw pixel data and matrix representations to object masks, metadata, and high-level measurements. Additionally, it includes utilities that streamline development workflows involving image processing and analysis.

The goal is to provide efficient, standardized access to image data while enabling intuitive and flexible integration for advanced image manipulation and analysis tasks.

Image Data Containers
=====================

    This category includes classes designed to offer fundamental abstractions for accessing image data in various representations:

1. :class:`ImageArray`
    Represents the multichannel pixel data of an image when provided. Useful for direct manipulation and pixel-wise operations.

2. :class:`ImageMatrix`
    Provides structured access to the image in its matrix form, offering methods suited for mathematical or analytical operations on image data. Automatically converted from RGB using weighted luminance conversion.

3. :class:`ImageEnhancedMatrix`
    An enhanceable copy of the image matrix to improve detection while maintaining the original image data integrity.

Objects and Object Mapping
==========================

These classes manage object-level abstractions and their corresponding data mappings:

1. :class:`ObjectMap`
    Represents a mapping of detected objects or regions in the image to their associated IDs. Useful for:

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
    Facilitates high-level interaction with detected objects within an image. Features include:

    - Feature extraction
    - Object comparison
    - Visualization of detected objects

2. :class:`MeasurementContainer` *(Coming Soon!)*
    Encapsulates measurements or attributes associated with objects, such as size, shape, or intensity. Organizes measurements for reproducibility and easier analysis.

3. :class:`MetadataContainer` *(Coming Soon!)*
    Responsible for storing and accessing metadata associated with the image. Examples of metadata include:

    - Acquisition details
    - Resolution
    - Additional contextual information.

HSV (Hue, Saturation, Brightness) Interface
===========================================

1. :class:`HsvAccessor`
Provides detailed access to the HSV (Hue, Saturation, Brightness) color space components of an image. This abstract includes support for:

- Direct pixel access via ``__getitem__`` and ``__setitem__``.
- Advanced utilities for image visualization and object-specific manipulation in the HSV domain.

**Key Methods:**
    - ``shape``: Returns the dimensions of the HSV representation.
    - ``copy``: Creates and returns a full copy of the HSV data structure.
    - ``histogram``: Computes a histogram for the HSV image or parts of it.
    - ``show``: Displays the HSV image.
    - ``show_objects``: Visualizes objects overlaid on the HSV image.
    - ``extract_obj_hue``, ``extract_obj_saturation``, ``extract_obj_brightness``, ``extract_obj``:
Extract specific HSV component data for individual image objects, aiding in targeted color-based analysis.

Purpose and Use Cases
=====================

The ``accessors`` submodule is designed for developers and researchers working on advanced image processing tasks. It is particularly suited for:

    - Object detection and feature extraction
    - HSV color-space analysis
    - Grid Analysis
    - Metadata association for image datasets
    - Advanced mathematical and matrix operations on image data.

"""
from ._image_accessor import ImageAccessor
from ._image_data_accessor import ImageDataAccessor
from ._image_array_accessor import ImageArray
from ._image_matrix_accessor import ImageMatrix
from ._image_enh_matrix_accessor import ImageEnhancedMatrix
from ._objmap_accessor import ObjectMap
from ._objmask_accessor import ObjectMask

from ._image_objects_accessor import ObjectsAccessor
# from ._measurement_container_interface import MeasurementAccessor
from ._metadata_accessor import MetadataAccessor

from ._hsv_accessor import HsvAccessor
from ._grid_accessor import GridAccessor

# Define __all__ to include all imported objects
__all__ = [
    "ImageAccessor",
    "ImageDataAccessor",
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

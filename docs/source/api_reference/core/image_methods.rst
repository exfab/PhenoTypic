Image Class Methods
===================

.. toctree::
   :hidden:
   :maxdepth: 2

   self

.. currentmodule:: phenotypic

Overview
--------

The :class:`Image` class is the primary interface for image processing, analysis, and manipulation
within the PhenoTypic framework. It combines data management, color space handling, object detection,
file I/O, and visualization capabilities into a unified interface.

Image data can be provided as NumPy arrays (2-D grayscale or 3-D RGB/RGBA), another Image instance,
or loaded from file. The class automatically manages format conversions and maintains internal
consistency across multiple data representations.

Key capabilities:

- **Data Management**: Grayscale, RGB, enhanced versions, object maps
- **Color Spaces**: RGB, grayscale, HSV, CIE XYZ, CIE L*a*b*, xy chromaticity
- **Object Analysis**: Detection results, masks, labels, measurements
- **File I/O**: Load/save images with metadata management
- **Visualization**: Display, overlays, plotting for pipeline development

Initialization & Creation
--------------------------

.. automethod:: Image.__init__

.. automethod:: Image.copy

Data Access - Grayscale
------------------------

.. autoproperty:: Image.gray

The ``gray`` property provides access to the grayscale representation of the image.
For RGB images, the grayscale version is automatically computed on first access.
For images created from grayscale arrays, this returns the original data.

**Grayscale API:**

.. autoclass:: phenotypic._core._image_parts.accessors._grayscale_accessor.Grayscale
   :members:
   :undoc-members:
   :show-inheritance:

Data Access - RGB
-----------------

.. autoproperty:: Image.rgb

The ``rgb`` property provides access to the RGB/RGBA image data if available.
For grayscale-only images, this property may be empty. The RGB accessor supports
numpy-like indexing and slicing operations.

**ImageRGB API:**

.. autoclass:: phenotypic._core._image_parts.accessors._rgb_accessor.ImageRGB
   :members:
   :undoc-members:
   :show-inheritance:

Data Access - Enhanced
----------------------

.. autoproperty:: Image.detect_mat

The ``detect_mat`` property provides access to an enhanceable copy of the grayscale image.
This is designed for processing pipelines where enhancement operations should be applied
without modifying the original grayscale data. Changes to detection matrix are tracked
separately from the base image.

**DetectMatAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._detect_mat_accessor.DetectMatAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Data Access - Objects (Detection Results)
------------------------------------------

.. autoproperty:: Image.objects

The ``objects`` property provides a high-level interface to detected objects, including
object properties, statistics, and analysis capabilities. This requires that object
detection has been performed on the image (e.g., via an ObjectDetector).

**ObjectsAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._objects_accessor.ObjectsAccessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoproperty:: Image.objmap

The ``objmap`` property provides access to the labeled object map, where each pixel
contains an integer label corresponding to the detected object it belongs to (0 for
background). This is the result of instance segmentation operations.

**ObjectMap API:**

.. autoclass:: phenotypic._core._image_parts.accessors._objmap_accessor.ObjectMap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoproperty:: Image.objmask

The ``objmask`` property provides access to the binary mask of all detected objects,
where pixels belonging to any object are marked as True/1 and background pixels are
marked as False/0.

**ObjectMask API:**

.. autoclass:: phenotypic._core._image_parts.accessors._objmask_accessor.ObjectMask
   :members:
   :undoc-members:
   :show-inheritance:

Color Space Operations
-----------------------

.. autoproperty:: Image.color

The ``color`` property provides unified access to multiple color space representations
through the ColorAccessor interface. This groups together device-dependent (HSV) and
CIE standard color spaces (XYZ, XYZ-D65, L*a*b*, xy chromaticity).

All color space conversions are computed on-demand and cached to avoid redundant
computations. The parent Image configuration (illuminant, observer, gamma) is used
consistently across all transformations.

**ColorAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._color_accessor.ColorAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Detailed documentation for individual color space operations is available in a dedicated section.

.. toctree::
   :maxdepth: 2

   color_space_operations

Visualization & Plotting
------------------------

.. autoproperty:: Image.plot

The ``plot`` property provides quality-of-life visualizations for pipeline development,
specifically designed for colony detection parameter tuning on arrayed microbe cultures
on agar plates. These methods help understand how morphological operations, size filtering,
and spatial patterns affect detection results.

All methods support flexible data requirements, automatically detecting whether labeled
objects (objmap) or binary masks (objmask) are available.

**PlotAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._plot_accessor.PlotAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Detailed documentation for individual plotting methods is available in a dedicated section.

.. toctree::
   :maxdepth: 2

   visualization_plotting

File I/O
--------

.. automethod:: Image.imread

Load an image from various file formats including JPEG, PNG, TIFF, and RAW camera files.
Automatically extracts metadata when available.

.. automethod:: Image.save2pickle

Save the complete Image state (data, metadata, detection results) to a pickle file
for later restoration.

.. automethod:: Image.save2hdf5

Save the image to HDF5 format with metadata. HDF5 provides efficient storage and
supports compression.

.. automethod:: Image.load_hdf5

Load an Image from HDF5 file created with ``save2hdf5``.

Image Manipulation
------------------

.. automethod:: Image.rotate

Rotate the image by a specified angle with optional centering and resampling control.

.. automethod:: Image.__getitem__

Support for numpy-like slicing to extract spatial regions or specific channels.
Enables intuitive image manipulation with bracket notation.

**Example:**

.. code-block:: python

    img = Image.imread('sample.jpg')
    
    # Extract region of interest
    roi = img[100:300, 100:300]
    
    # Extract specific channel (RGB)
    red_channel = img[..., 0]

Visualization (Basic Display)
------------------------------

.. automethod:: Image.show

Display the image using matplotlib. Supports grayscale and RGB images with automatic
colormap selection.

.. automethod:: Image.show_overlay

Display the image with detection results overlaid. Shows object boundaries, labels,
or other annotations on top of the original image.

.. automethod:: Image.show_objects

Display detected objects with object-specific visualization, including individual
object highlighting and labeling.

Metadata Management
-------------------

.. autoproperty:: Image.metadata

Access the metadata accessor for reading and writing image metadata. Metadata is
categorized into private, protected, public, and imported categories for organized
management.

**MetadataAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._metadata_accessor.MetadataAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Comparison & Utility
--------------------

.. automethod:: Image.__eq__

Compare two Image instances for equality based on their data content.

.. automethod:: Image.__hash__

Compute a hash value for the Image based on its unique identifier.

.. automethod:: Image.__repr__

Return a string representation of the Image with key information (shape, type, name).

.. automethod:: Image.__str__

Return a human-readable string description of the Image.


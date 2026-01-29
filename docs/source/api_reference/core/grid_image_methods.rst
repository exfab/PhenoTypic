GridImage Class Methods
========================

.. toctree::
   :hidden:
   :maxdepth: 2
   :local:

   self

.. currentmodule:: phenotypic

Overview
--------

The :class:`GridImage` class extends :class:`Image` to support grid-based processing
and overlay visualization for arrayed samples on agar plates or other gridded microbe
cultures. It combines complete image processing capabilities with grid-aware operations,
enabling well-level detection, measurement, and visualization.

This class is designed for high-throughput phenotyping workflows where colonies are
arranged in regular arrays such as:

- **96-well plates:** 8 rows × 12 columns
- **384-well plates:** 16 rows × 24 columns  
- **1536-well plates:** 32 rows × 48 columns
- **Custom arrays:** Any regular grid pattern

The class automatically manages grid detection and alignment, supports grid-based
slicing to extract individual well images, and provides overlay visualizations with
gridlines, well labels, and measurements aligned to the detected grid structure.

Initialization & Creation
--------------------------

.. automethod:: GridImage.__init__

Create a GridImage with grid-based processing capabilities. The grid structure can
be specified explicitly (nrows, ncols) or detected automatically using a custom
GridFinder algorithm.

**Key Parameters:**

- **arr:** Image data (NumPy array, Image instance, or file path)
- **nrows, ncols:** Grid dimensions (e.g., 8×12 for 96-well plates)
- **grid_finder:** Optional custom grid detection algorithm (defaults to AutoGridFinder)
- **illuminant, gamma:** Color space configuration (inherited from Image)

**Example:**

.. code-block:: python

    from phenotypic import GridImage
    
    # Load 96-well plate image
    grid_img = GridImage('plate_96well.jpg', nrows=8, ncols=12)
    
    # Load 384-well plate image
    grid_img_384 = GridImage('plate_384well.jpg', nrows=16, ncols=24)
    
    # Show with gridlines overlay
    grid_img.plot.overlay(show_gridlines=True)

.. automethod:: GridImage.copy

Create a complete independent copy of the GridImage, including all grid structure,
detection results, and metadata.

Grid Operations
---------------

.. autoproperty:: GridImage.grid

The ``grid`` property provides access to grid detection and optimization functionality.
Through this accessor, you can:

- Access detected grid structure (row/column edges)
- Retrieve grid dimensions (nrows, ncols)
- Access the GridFinder instance used for detection
- Perform grid refinement and optimization

The grid is automatically detected on first access using the configured GridFinder
algorithm. Detection results are cached for efficiency.

**Example:**

.. code-block:: python

    from phenotypic import GridImage
    from phenotypic.detect import OtsuDetector
    
    # Create and detect colonies
    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    detector = OtsuDetector()
    detected = detector.apply(grid_img)
    
    # Access grid structure
    nrows = detected.grid.nrows  # 8
    ncols = detected.grid.ncols  # 12
    
    # Get grid edges
    row_edges = detected.grid.get_row_edges()
    col_edges = detected.grid.get_col_edges()
    
    # Access grid finder
    finder = detected.grid.grid_finder

**GridAccessor API:**

.. autoclass:: phenotypic._core._image_parts.accessors._grid_accessor.GridAccessor
   :members:
   :undoc-members:
   :show-inheritance:

Grid-Based Access
-----------------

GridImage supports grid-aware slicing to extract individual well images or sub-regions
aligned to the detected grid structure. This enables well-level analysis and processing.

**Grid Coordinate Transformations:**

The class provides methods to convert between:

- Pixel coordinates (x, y in image space)
- Grid coordinates (row, column in well space)
- Well indices (linear index from 0 to nrows×ncols-1)

**Extracting Individual Wells:**

.. code-block:: python

    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    
    # Extract specific well by row/column
    well_A1 = grid_img.grid.get_well(row=0, col=0)
    well_H12 = grid_img.grid.get_well(row=7, col=11)
    
    # Iterate over all wells
    for row in range(grid_img.grid.nrows):
        for col in range(grid_img.grid.ncols):
            well = grid_img.grid.get_well(row, col)
            # Process individual well image

Data Access (Inherited from Image)
-----------------------------------

GridImage inherits all data access methods and properties from the :class:`Image` class.
All the following are available:

**Data Accessors:**

- :attr:`Image.gray` - Grayscale representation
- :attr:`Image.rgb` - RGB/RGBA image data
- :attr:`Image.enh_gray` - Enhanceable grayscale copy
- :attr:`Image.objects` - High-level interface to detected objects
- :attr:`Image.objmap` - Labeled object map
- :attr:`Image.objmask` - Binary mask of detected objects

**Color Space Operations:**

- :attr:`Image.color` - Unified ColorAccessor interface
  
  - ``Image.color.XYZ`` - CIE XYZ color space
  - ``Image.color.XYZ_D65`` - XYZ under D65 illuminant
  - ``Image.color.Lab`` - Perceptually uniform L*a*b* color space
  - ``Image.color.xy`` - CIE xy chromaticity coordinates
  - ``Image.color.hsv`` - HSV (Hue-Saturation-Value) color space

**Visualization & Plotting:**

- :attr:`Image.plot` - Unified PlotAccessor interface

  - ``Image.plot.morph_progression()`` - Visualize morphological operation effects
  - ``Image.plot.structural_response_curve()`` - Quantify detection sensitivity
  - ``Image.plot.boundary_displacement()`` - Spatial sensitivity heatmap
  - ``Image.plot.size_distribution()`` - Comprehensive size analysis
  - ``Image.plot.spatial_size_map()`` - Pseudo-color size distribution map
  - ``Image.plot.size_scatter()`` - Size-intensity correlation scatter plot

For detailed documentation of these inherited methods, see :doc:`image_methods`.

Grid Visualization
------------------

.. automethod:: GridImage.show_overlay

Display the grid image with optional overlays including:

- **Gridlines:** Show detected row and column boundaries
- **Well labels:** Annotate wells with row/column identifiers (A1, B2, etc.)
- **Detection results:** Overlay detected objects with boundaries
- **Measurements:** Display well-level measurements on the grid

This method extends the base :meth:`Image.show_overlay` with grid-specific visualization
capabilities.

**Example:**

.. code-block:: python

    from phenotypic import GridImage
    from phenotypic.detect import OtsuDetector
    
    # Load and detect
    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    detector = OtsuDetector()
    detected = detector.apply(grid_img)
    
    # Show with gridlines and labels
    detected.plot.overlay(
        show_gridlines=True,
        show_labels=True,
    )

    # Show with gridlines and section boxes
    detected.plot.overlay(
        show_gridlines=True,
        show_section_boxes=True
    )

**Grid-Specific Visualization Methods:**

The GridImage class provides additional grid-aware visualization methods:

- **Well highlighting:** Highlight specific wells or well groups
- **Grid-aligned annotations:** Add text or markers at well centers
- **Well-level heatmaps:** Visualize measurements as colors on grid structure
- **Row/column summaries:** Display aggregate statistics per row or column

**Example - Well Highlighting:**

.. code-block:: python

    # Highlight control wells - show gridlines and section boxes
    detected.plot.overlay(
        show_gridlines=True,
        show_section_boxes=True
    )

**Example - Well-Level Heatmap:**

.. code-block:: python

    # Visualize colony count per well
    detected.show_grid_heatmap(
        metric='colony_count',
        cmap='viridis',
        show_values=True
    )

File I/O (Inherited)
--------------------

GridImage inherits all file I/O methods from :class:`Image`:

- :meth:`Image.imread` - Load image from file
- :meth:`Image.save2pickle` - Save to pickle format
- :meth:`Image.save2hdf5` - Save to HDF5 format with metadata
- :meth:`Image.load_hdf5` - Load from HDF5

The grid structure (nrows, ncols, grid_finder configuration) is preserved during
save/load operations, allowing complete restoration of the GridImage state.

**Example:**

.. code-block:: python

    # Create and configure grid image
    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    
    # Save with grid structure preserved
    grid_img.save2hdf5('plate_with_grid.h5')
    
    # Load later - grid structure restored
    restored = GridImage.load_hdf5('plate_with_grid.h5')
    print(restored.grid.nrows, restored.grid.ncols)  # 8, 12

Image Manipulation (Inherited)
-------------------------------

GridImage inherits image manipulation methods from :class:`Image`:

- :meth:`Image.rotate` - Rotate image with optional centering
- :meth:`Image.__getitem__` - Numpy-like slicing

.. note::
   When using image manipulation operations on GridImage, be aware that geometric
   transformations (rotation, cropping) may invalidate the detected grid structure.
   You may need to re-detect the grid after transformations.

**Example:**

.. code-block:: python

    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    
    # Rotate (grid structure may need re-detection)
    rotated = grid_img.rotate(angle=2.5)
    
    # Grid detection will be re-run on first access
    rotated.plot.overlay(show_gridlines=True)

Metadata Management (Inherited)
--------------------------------

GridImage inherits metadata management from :class:`Image`:

- :attr:`Image.metadata` - Access metadata accessor

Grid-specific metadata (nrows, ncols, grid_finder type) is automatically stored
in the metadata and preserved during save/load operations.

Comparison & Utility (Inherited)
---------------------------------

GridImage inherits comparison and utility methods from :class:`Image`:

- :meth:`Image.__eq__` - Equality comparison
- :meth:`Image.__hash__` - Hash value computation
- :meth:`Image.__repr__` - String representation
- :meth:`Image.__str__` - Human-readable description

The string representation includes grid dimensions for easy identification:

.. code-block:: python

    grid_img = GridImage('plate.jpg', nrows=8, ncols=12)
    print(grid_img)
    # Output: GridImage(shape=(1200, 1600), grid=8×12, name='plate.jpg')


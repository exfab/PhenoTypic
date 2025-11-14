Interactive Object Measurement
==============================

The ``interactive_measure()`` method provides an interactive Dash-based tool for visualizing and measuring detected objects in images. This feature is particularly useful for validating detection results, performing quality control, and manually selecting objects for analysis.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The interactive measurement tool displays your image with detected objects overlaid and provides:

* **Visual overlay** of detected objects with adjustable transparency
* **Click-to-select** functionality for individual objects
* **Real-time area measurements** for selected objects
* **Measurement export** to CSV files
* **Interactive parameter adjustment** for visualization

Installation
------------

The interactive features require optional dependencies. Install them with:

.. code-block:: bash

   pip install phenotypic[interactive]

This installs:

* ``dash>=2.0.0`` - Web application framework
* ``jupyter-dash>=0.4.0`` - Jupyter notebook integration

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   import phenotypic as pht
   from phenotypic.detect import OtsuDetector
   
   # Load and detect objects
   image = pht.Image.imread('colony_plate.jpg')
   detector = OtsuDetector()
   detector.apply(image)
   
   # Launch interactive tool
   image.interactive_measure()

This will open the interactive tool in your default web browser.

Jupyter Notebook
~~~~~~~~~~~~~~~~

For use in Jupyter notebooks, use inline mode:

.. code-block:: python

   # Launch inline in notebook
   image.interactive_measure(mode='inline', height=600)

Method Parameters
-----------------

.. py:method:: Image.interactive_measure(port=8050, height=800, mode='external', detector_type='otsu')

   Launch an interactive Dash application for measuring object areas.
   
   :param int port: Port number for the Dash server. Default: 8050.
   :param int height: Height of the image display in pixels. Default: 800.
   :param str mode: Display mode - 'inline', 'external', or 'jupyterlab'. Default: 'external'.
   :param str detector_type: Type of detector for reference (currently informational). Default: 'otsu'.
   
   :raises ImportError: If dash or jupyter-dash are not installed.
   :raises ValueError: If the image has no grayscale data.

Display Modes
~~~~~~~~~~~~~

**External Mode** (Default)
   Opens the tool in a new browser tab/window. Best for full-screen analysis.
   
   .. code-block:: python
   
      image.interactive_measure(mode='external')

**Inline Mode**
   Embeds the tool directly in Jupyter notebook output. Best for notebook workflows.
   
   .. code-block:: python
   
      image.interactive_measure(mode='inline', height=700)

**JupyterLab Mode**
   Optimized display for JupyterLab environment.
   
   .. code-block:: python
   
      image.interactive_measure(mode='jupyterlab')

Interface Features
------------------

Image Display
~~~~~~~~~~~~~

The main panel displays your image with:

* **Base image**: Grayscale representation of your image
* **Object overlay**: Colored regions showing detected objects
* **Interactive clicks**: Click objects to select/deselect them
* **Hover information**: View pixel coordinates and intensity values

Control Panel
~~~~~~~~~~~~~

The right sidebar provides:

**Show Overlay Toggle**
   Enable/disable the object overlay visualization.

**Overlay Transparency Slider**
   Adjust the transparency (alpha) of the overlay from 0.0 (transparent) to 1.0 (opaque).

**Measurement Display**
   Real-time table showing:
   
   * Object label
   * Area in pixels
   * Centroid coordinates (row, column)
   * Total selected objects
   * Total area sum

**Clear Selection Button**
   Remove all selected objects at once.

**Export Measurements Button**
   Save current measurements to a CSV file named ``measurements_{image_name}.csv``.

Workflow Examples
-----------------

Quality Control Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load and process image
   image = pht.Image.imread('plate_scan.jpg')
   
   # 2. Apply preprocessing
   from phenotypic.enhance import GaussianBlur
   blur = GaussianBlur(sigma=2)
   blur.apply(image)
   
   # 3. Detect objects
   from phenotypic.detect import WatershedDetector
   detector = WatershedDetector(min_size=50)
   detector.apply(image)
   
   # 4. Interactively validate results
   image.interactive_measure()
   
   # Click on objects to verify they are real colonies
   # Export measurements for validated objects

Selective Measurement
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect all objects
   detector = OtsuDetector()
   detector.apply(image)
   
   print(f"Total objects detected: {image.num_objects}")
   
   # Use interactive tool to select only objects of interest
   image.interactive_measure()
   
   # Export measurements for selected objects only
   # (Click "Export Measurements" button in the interface)

Comparison with Non-Interactive Methods
----------------------------------------

The interactive tool complements programmatic measurement approaches:

**Interactive Tool** - Best for:
   * Visual validation
   * Quality control
   * Selective measurement
   * Manual curation
   * Educational demonstrations

**Programmatic Measurement** - Best for:
   * Batch processing
   * Automated workflows
   * High-throughput analysis
   * Reproducible pipelines

Example comparison:

.. code-block:: python

   from phenotypic.measure import MeasureSize
   
   # Programmatic: Measure all objects at once
   measurer = MeasureSize()
   all_measurements = measurer.measure(image, include_meta=True)
   print(f"All objects: {len(all_measurements)} measured")
   
   # Interactive: Select specific objects visually
   image.interactive_measure()
   # User selects 10 colonies of interest via clicks
   # Exports measurements for those 10 only

Architecture
------------

The interactive measurement system is built on three components:

1. **InteractiveImageAnalyzer**
   Abstract base class providing common infrastructure:
   
   * Dash application lifecycle management
   * Jupyter integration
   * Image format conversion
   * Overlay generation

2. **InteractiveMeasurementAnalyzer**
   Concrete implementation for area measurement:
   
   * Click-to-select object functionality
   * Real-time area calculation
   * Measurement display and export
   * Parameter controls

3. **Image.interactive_measure()**
   Convenience method on the Image class:
   
   * Simple API for end users
   * Instantiates appropriate analyzer
   * Manages application lifecycle

Extensibility
-------------

The base ``InteractiveImageAnalyzer`` class can be subclassed to create custom interactive tools:

.. code-block:: python

   from phenotypic.tools import InteractiveImageAnalyzer
   
   class CustomAnalyzer(InteractiveImageAnalyzer):
       def setup_layout(self):
           # Define custom Dash layout
           pass
       
       def create_callbacks(self):
           # Define custom interactivity
           pass
       
       def update_image(self, *args, **kwargs):
           # Custom image update logic
           pass

Technical Details
-----------------

Dependencies
~~~~~~~~~~~~

* **dash**: Web application framework for building the interface
* **jupyter-dash**: Integration layer for Jupyter notebooks
* **plotly**: Graphing library for image display
* **skimage**: Image processing for overlays (already a core dependency)

Performance
~~~~~~~~~~~

* Images are converted to 8-bit format for display (original data unchanged)
* Object maps are stored as sparse matrices to minimize memory usage
* Measurements are calculated on-demand when objects are selected
* The interface runs as a separate web server process

Cross-Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interactive tool works on:

* **macOS**: Tested on macOS 10.13+
* **Linux**: Tested on Ubuntu 20.04+
* **Windows**: Tested on Windows 10+

Browser compatibility:

* Chrome (recommended)
* Firefox
* Safari
* Edge

Troubleshooting
---------------

Port Already in Use
~~~~~~~~~~~~~~~~~~~

If you see an error about the port being in use:

.. code-block:: python

   # Try a different port
   image.interactive_measure(port=8051)

Import Errors
~~~~~~~~~~~~~

If you get ``ImportError`` about missing dependencies:

.. code-block:: bash

   # Install interactive dependencies
   pip install phenotypic[interactive]
   
   # Or install individually
   pip install dash>=2.0.0 jupyter-dash>=0.4.0

Display Issues in Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~

If the tool doesn't display in Jupyter:

1. Ensure jupyter-dash is installed
2. Try restarting the kernel
3. Use external mode as fallback:

.. code-block:: python

   image.interactive_measure(mode='external')

See Also
--------

* :class:`phenotypic.measure.MeasureSize` - Non-interactive area measurement
* :class:`phenotypic.tools.InteractiveImageAnalyzer` - Base analyzer class
* :class:`phenotypic.tools.InteractiveMeasurementAnalyzer` - Measurement analyzer implementation
* :doc:`../api_reference/phenotypic.detect` - Object detection methods
* :doc:`../api_reference/phenotypic.measure` - Measurement operations


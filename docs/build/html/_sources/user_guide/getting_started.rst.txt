.. _getting_started:

Getting Started
===============

Welcome to PhenoScope! This guide will help you get up and running quickly with this powerful image analysis and phenotyping library.

Installation
-----------

PhenoScope can be installed using pip:

.. code-block:: bash

   pip install phenoscope

For more detailed installation instructions, see the :doc:`installation` page.

Quick Start Example
-------------------

Here's a simple example to get you started with PhenoScope:

.. code-block:: python

   import phenoscope as ps
   
   # Load an image
   image = ps.Image.from_file("path/to/your/image.jpg")
   
   # Display the image
   image.show()
   

Core Concepts
-------------

PhenoScope is built around a few key concepts:

1. **Image Objects**: The fundamental unit in PhenoScope is the `Image` class, which provides methods for loading, processing, and analyzing images.

2. **GridImage Objects**: For working with multi-well plates or grid-based images, the `GridImage` class provides specialized functionality.

3. **Processing Pipelines**: Chain multiple operations together to create reproducible image processing workflows.

4. **Feature Extraction**: Extract quantitative features from images for downstream analysis.

Next Steps
----------

- Explore the :doc:`user_guide/index` for more detailed information
- Check out the :doc:`tutorial/index` for step-by-step guides
- Browse the :doc:`examples/index` for practical applications
- Refer to the :doc:`api_reference/index` for detailed API documentation

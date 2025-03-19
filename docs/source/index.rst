.. PhenoScope documentation master file, created by
   sphinx-quickstart on Sat Mar  8 14:29:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. raw:: html

..    <div class="logo-container">
..        <img src="/assets/PhenoScopeLogo.svg" class="logo-light" alt="PhenoScope Logo Light Mode">
..        <img src="/assets/PhenoScopeLogo-DarkMode.svg" class="logo-dark" alt="PhenoScope Logo Dark Mode">
..    </div>

PhenoScope Documentation
========================

Welcome to PhenoScope's documentation. Here you'll find comprehensive guides and examples to help you get the most out of PhenoScope.

Quick Start
-----------

PhenoScope is a Python library for image analysis and phenotyping. It provides tools for processing, analyzing, and extracting features from images.

**Installation**

.. code-block:: bash

   pip install phenoscope

**Basic Usage**

.. code-block:: python

   import phenoscope as ps
   
   # Load an image
   img = ps.imread('path/to/image.jpg')
   
   # Process the image
   # ... your code here ...

For more detailed installation instructions, see the :doc:`installation` page.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   installation

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   user_guide/index
   examples/index
   api_reference/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

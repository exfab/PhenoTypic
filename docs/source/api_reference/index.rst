API Reference
=============

This section provides detailed API documentation for all PhenoScope modules and functions.

.. currentmodule:: phenoscope
.. autosummary::
    :toctree: Image
    :recursive:
    :template: image_accessor.rst
    :caption: Image

    Image 
    Image.array
    Image.matrix
    Image.enh_matrix
    Image.omask
    Image.omap
    Image.obj
    Image.hsv

.. currentmodule:: phenoscope
.. autosummary::
    :toctree: GridImage
    :recursive:
    :template: image_accessor.rst
    :caption: GridImage

    GridImage
    GridImage.grid

.. toctree::
   :maxdepth: 2
   :caption: Modules

   phenoscope.abstract
   phenoscope.core
   phenoscope.data
   phenoscope.detection
   phenoscope.grid
   phenoscope.measure
   phenoscope.morphology
   phenoscope.objects
   phenoscope.pipeline
   phenoscope.preprocessing
   phenoscope.transform
   phenoscope.util

API Reference
=============

This section provides detailed API documentation for all PhenoScope modules and functions.

Image
-----
.. currentmodule:: phenoscope

.. autosummary::
    :toctree: image_class
    :caption: Image
    :template: image_class.rst
    
    Image

.. autosummary::
    :toctree: image_class
    :template: image_accessor.rst

    Image.array
    Image.matrix
    Image.enh_matrix
    Image.objmask
    Image.objmap
    Image.objects
    Image.hsv



GridImage
---------
.. currentmodule:: phenoscope
.. autosummary::
    :toctree: grid_image
    :caption: GridImage
    
    GridImage

GridImage Accessors
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: phenoscope
.. autosummary::
    :toctree: grid_image_accessors
    :recursive:
    :template: image_accessor.rst
    :caption: GridImage Accessors

    GridImage.grid

.. toctree::
   :maxdepth: 1
   :caption: Modules
   
   Image
   phenoscope.GridImage
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

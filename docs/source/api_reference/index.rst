API Reference
=============

This section provides detailed API documentation for all Phenotypic modules and functions.

Image
-----
.. currentmodule:: phenotypic

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
.. currentmodule:: phenotypic
.. autosummary::
    :toctree: grid_image
    :caption: GridImage
    
    GridImage

GridImage Accessors
^^^^^^^^^^^^^^^^^^^
.. currentmodule:: phenotypic
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
   phenotypic.GridImage
   phenotypic.abstract
   phenotypic.core
   phenotypic.data
   phenotypic.detection
   phenotypic.grid
   phenotypic.measure
   phenotypic.morphology
   phenotypic.objects
   phenotypic.pipeline
   phenotypic.preprocessing
   phenotypic.transform
   phenotypic.util

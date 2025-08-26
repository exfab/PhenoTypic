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
    Image.hsb



GridImage
---------
.. currentmodule:: phenotypic
.. autosummary::
    :toctree: grid_image
    :caption: GridImage
    
    GridImage

.. autosummary::
    :toctree: image_class
    :template: image_accessor.rst

    GridImage.grid

Modules
-------
.. toctree::
   :maxdepth: 1
   :caption: Modules
   
   image_class.phenotypic.Image
   grid_image.phenotypic.GridImage
   phenotypic.ImageSet
   phenotypic.ImagePipeline
   phenotypic.abstract
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

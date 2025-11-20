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

    Image.rgb
    Image.gray
    Image.enh_gray
    Image.objmask
    Image.objmap
    Image.objects
    Image.color



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
   
   phenotypic.abc_
   phenotypic.analysis
   phenotypic.core
   phenotypic.correction
   phenotypic.data
   phenotypic.detect
   phenotypic.enhance
   phenotypic.grid
   phenotypic.measure
   phenotypic.prefab
   phenotypic.refine
   phenotypic.tools

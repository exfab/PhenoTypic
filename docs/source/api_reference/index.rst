API Reference
=============

This section provides detailed API documentation for all Phenotypic modules and functions.

Image
-----
.. currentmodule:: phenotypic

.. autosummary::
    :toctree: api/
    :caption: Image
    :template: image_class.rst

    Image

.. autosummary::
    :toctree: api/
    :template: image_accessor_alt.rst

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
    :toctree: api/
    :caption: GridImage

    GridImage

.. autosummary::
    :toctree: api.
    :template: image_accessor_alt.rst

    GridImage.grid

Modules
-------
.. currentmodule:: phenotypic
.. autosummary::
   :toctree: api/
   :caption: Modules
   :template: module.rst
   :recursive:

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
   phenotypic.util

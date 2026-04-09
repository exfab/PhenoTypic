Installation
============

Prerequisites
-------------

Before installing Phenotypic, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)
* pixi (recommended)

Installation Methods
------------------

From PyPi
+++++++++

Using pixi (recommended)
+++++++++
.. code-block:: bash

   pixi add --pypi phenotypic

Using pip
+++++++++

.. code-block:: bash

    pip install phenotypic

From Source
-----------

To install from source:


.. code-block:: bash

  git clone https://github.com/exfab/PhenoTypic.git
  cd PhenoTypic && pixi install


Optional Extras
---------------

PhenoTypic provides optional extras for different use cases:

- ``[jupyter]`` — Lightweight Jupyter notebook support (ipykernel, ipywidgets, jupyter).
- ``[gui]`` — Full interactive environment including napari viewer, Panel dashboards,
  and Jupyter integration. Required for ``image.rgb.napari()`` and related viewer methods.

.. code-block:: bash

   # Jupyter notebooks only
   pixi add --pypi "phenotypic[jupyter]"

   # Full interactive / GUI environment (napari, Panel, Jupyter)
   pixi add --pypi "phenotypic[gui]"


Development Installation
========================

For development of new modules, install additional dependencies:

.. code-block:: bash

    git clone https://github.com/exfab/PhenoTypic.git

    # If on windows, docs may fail to install
    cd PhenoTypic && pixi install -e full


Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)

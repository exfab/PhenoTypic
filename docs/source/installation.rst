Installation
============

Prerequisites
-------------

Before installing Phenotypic, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)
* uv (optional, but recommended)

Installation Methods
------------------

From PyPi
+++++++++

Using uv (recommended)
++++++++
.. code-block:: bash

   uv add phenotypic

Using pip
+++++++++

.. code-block:: bash

    pip install phenotypic

From Source
-----------

To install from source:


.. code-block:: bash

  git clone https://github.com/exfab/PhenoTypic.git
  uv pip install -e ./PhenoTypic # Replace with the path to the module


Optional Extras
---------------

PhenoTypic provides optional extras for different use cases:

- ``[jupyter]`` — Lightweight Jupyter notebook support (ipykernel, ipywidgets, jupyter).
- ``[gui]`` — Full interactive environment including napari viewer, Panel dashboards,
  and Jupyter integration. Required for ``image.rgb.napari()`` and related viewer methods.

.. code-block:: bash

   # Jupyter notebooks only
   uv add "phenotypic[jupyter]"

   # Full interactive / GUI environment (napari, Panel, Jupyter)
   uv add "phenotypic[gui]"


Development Installation
========================

For development of new modules, install additional dependencies:

.. code-block:: bash

    git clone https://github.com/exfab/PhenoTypic.git

    # If on windows, docs may fail to install
    cd PhenoTypic && uv sync --group dev --group docs --extras gui


Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)

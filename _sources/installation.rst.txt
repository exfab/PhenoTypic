Installation
============

Prerequisites
-------------

Before installing Phenotypic, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)
* uv (recommended; see https://docs.astral.sh/uv/)

Installation Methods
------------------

From PyPi
+++++++++

Using uv (recommended)
+++++++++
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
  cd PhenoTypic && uv sync


Optional Extras
---------------

PhenoTypic provides optional extras for different use cases:

- ``[gui]`` — Full interactive environment including napari viewer, Panel dashboards,
  and Jupyter integration. Required for ``image.rgb.napari()`` and related viewer methods.
- ``[torch]`` — PyTorch + SAM2 for ``Sam2Detector`` (Linux/macOS only).

.. code-block:: bash

   # Full interactive / GUI environment (napari, Panel, Jupyter)
   uv add "phenotypic[gui]"

   # SAM2 GPU detector (Linux/macOS)
   uv add "phenotypic[torch]"

``micro_sam`` (used by ``MicroSamDetector``) is only published on
conda-forge and is **not** included in any PhenoTypic extra. See the
`GPU Detection Setup
<how_to/pages/gpu_detection_setup.html>`_ guide for a self-service
recipe that combines PhenoTypic and ``micro_sam`` in a single ``pixi``
environment.


Development Installation
========================

For development of new modules, sync with the ``dev`` (and optionally
``docs``) dependency groups:

.. code-block:: bash

    git clone https://github.com/exfab/PhenoTypic.git
    cd PhenoTypic
    uv sync --group dev --group docs


Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)

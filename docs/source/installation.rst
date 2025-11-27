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


Development Installation
========================

For development of new modules, install additional dependencies:

.. code-block:: bash

    git clone https://github.com/exfab/PhenoTypic.git

    # If on windows, docs may fail to install
    cd PhenoTypic && uv sync --group dev --group docs


Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)

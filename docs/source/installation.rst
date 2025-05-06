Installation
============

Prerequisites
------------

Before installing Phenotypic, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)

Installation Methods
------------------

From PyPI
~~~~~~~~~

The simplest way to install Phenotypic is via pip:

.. code-block:: bash

   pip install phenotypic

From Source
~~~~~~~~~~~

To install from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Xander-git/Phenotypic.git
      cd phenotypic

2. Install the package:

   .. code-block:: bash

      pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development purposes, install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verification
------------

To verify the installation, run:

.. code-block:: python

   import phenotypic
   print(phenotypic.__version__)

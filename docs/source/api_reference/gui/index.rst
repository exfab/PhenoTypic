GUI internals
=============

API reference for the Dash-based pipeline builder's pure-Python
internals. These modules live under ``phenotypic.gui.builder`` and are
public only to the extent that tests and external tooling import them;
all wiring, layout, and rendering primitives remain internal to the
GUI itself.

.. toctree::
   :maxdepth: 1

   builder
   builder_dispatch
   builder_validation

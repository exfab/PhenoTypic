.. PhenoScope documentation master file, created by
   sphinx-quickstart on Sat Mar  8 14:29:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: phenoscope

PhenoScope Documentation
========================

Welcome to PhenoScope's documentation. Here you'll find comprehensive guides and examples to help you get the most out of PhenoScope.



.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: sd-text-center



   .. grid-item-card::  User guide
      :img-top: ./_static/assets/500x300/no_background_svg/user_guide_book.svg
      :class-card: intro-card
      :shadow: md

      The user guide explains how to get started as well as an in-depth overview of the
      key concepts of PhenoScope with useful background information and explanation.

      +++

      .. button-ref:: user_guide/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         To the user guide

   .. grid-item-card:: Examples
      :img-top: ./_static/assets/500x300/no_background_svg/getting_started_rocket.svg
      :class-card: intro-card
      :shadow: md

      The Examples provide a hands-on introduction to *PhenoScope*.

      +++

      .. button-ref:: examples/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         To the examples

   .. grid-item-card::  API reference
      :img-top: ./_static/assets/500x300/no_background_svg/api_ref_sign.svg
      :class-card: intro-card
      :shadow: md

      The reference guide contains the detailed description of
      the PhenoScope API. The reference describes in detail how the methods work and which parameters can
      be used. It assumes that you have an understanding of the key concepts.

      +++

      .. button-ref:: api_reference/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         To the reference guide

   .. grid-item-card::  Developer guide
      :img-top: ./_static/assets/500x300/no_background_svg/dev_guide.svg
      :class-card: intro-card
      :shadow: md

      PhenoScope's strength is in its ability to integrate new modules and workflows. Learn how to make your own module and workflow here. 
      Learn how to contribute your new module to the codebase here. Found a typo in the documentation? The contributing guidelines will guide
      you through the process of making your own module and improving PhenoScope.

      +++

      .. button-ref:: dev_guide/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         To the development guide


Quick Start
-----------

PhenoScope is a Python library for image analysis and phenotyping. It provides tools for processing, analyzing, and extracting features from images.

**Installation**

.. code-block:: bash

   pip install phenoscope



For more detailed installation instructions, see the :doc:`installation` page.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   installation

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   user_guide/index
   examples/index
   api_reference/index
   dev_guide/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

:notoc:

.. module:: phenotypic

PhenoTypic
==========

.. container:: subtitle

   A modular framework for bioimage analysis and visualization


Welcome to PhenoTypic's documentation. Whether you're detecting colonies for the
first time or building custom analysis pipelines, these guides will help you get
started.


.. grid:: 1 2 3 3
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card::  Tutorials
      :img-top: ./_static/assets/200x150/user_guide_book.svg
      :class-card: intro-card
      :shadow: md

      Step-by-step lessons that take you from loading your first plate image
      to building complete analysis pipelines. Start here if you're new.

      +++

      .. button-ref:: tutorials/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Start learning

   .. grid-item-card:: Guides
      :img-top: ./_static/assets/200x150/examples.svg
      :class-card: intro-card
      :shadow: md

      How-to recipes for specific problems, conceptual explanations of how
      and why things work, and guides for extending PhenoTypic with your
      own operations, detectors, and dashboards.

      +++

      .. button-ref:: how_to/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Browse the guides

   .. grid-item-card:: Measurements Reference
      :img-top: ./_static/assets/200x150/dev_guide.svg
      :class-card: intro-card
      :shadow: md

      A quick reference for every column produced by PhenoTypic's per-object
      measurement operators — useful if you've received processed data and
      need to know what a column means.

      +++

      .. button-ref:: measurements_ref/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Look up a measurement

   .. grid-item-card::  API Reference
      :img-top: ./_static/assets/200x150/api_ref_sign.svg
      :class-card: intro-card
      :shadow: md

      Detailed reference for every public class, function, and parameter.
      Includes CLI reference, configuration, and glossary.

      +++

      .. button-ref:: api_reference/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         To the API reference

   .. grid-item-card::  Report a Problem
      :img-top: ./_static/assets/200x150/contact_us.svg
      :class-card: intro-card
      :shadow: md
      :link: https://github.com/exfab/PhenoTypic/issues
      :link-type: url

      Notice a problem or need help? Open an issue on GitHub and add a label
      so we can help you faster.

      +++

      .. button-link:: https://github.com/exfab/PhenoTypic/issues
         :click-parent:
         :color: secondary
         :expand:

         Report an issue


.. toctree::
   :maxdepth: 3
   :caption: Documentation
   :hidden:
   :titlesonly:

   tutorials/index
   how_to/index
   explanation/index
   extending/index
   measurements_ref/index
   api_reference/index
   contrib_guide/index
   downloads

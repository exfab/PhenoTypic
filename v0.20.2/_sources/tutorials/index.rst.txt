Tutorials
=========

Whether you're installing PhenoTypic for the first time, working through the
notebook learning path, batch-processing a directory of plates from the shell,
or taking a screenshot tour of the GUI hub, the tutorials are organized into
four tracks. Pick the one that matches how you plan to use PhenoTypic.


.. grid:: 1 2 2 2
   :gutter: 4
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card:: Getting Started
      :class-card: intro-card
      :shadow: md

      Install PhenoTypic, set up extras (``[gui]``, ``[torch]``), verify your
      install, and launch the GUI hub for the first time.

      +++

      .. button-ref:: getting_started
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Install & launch

   .. grid-item-card:: Python API
      :class-card: intro-card
      :shadow: md

      Ten guided notebooks that take you from loading your first plate image
      to building grid-aware pipelines and detecting filamentous fungi.

      +++

      .. button-ref:: python_api
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Start the learning path

   .. grid-item-card:: Command Line
      :class-card: intro-card
      :shadow: md

      The four ``--mode`` execution modes, the parameters that matter, and how
      to batch-process a directory of plates locally or on SLURM.

      +++

      .. button-ref:: cli
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Run from the shell

   .. grid-item-card:: GUI
      :class-card: intro-card
      :shadow: md

      A screenshot-driven walkthrough of the PhenoTypic hub — sandbox setup,
      pipeline builder, run console, results viewer, and analysis sub-app.

      +++

      .. button-ref:: gui/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Tour the GUI


.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   python_api
   cli
   gui/index

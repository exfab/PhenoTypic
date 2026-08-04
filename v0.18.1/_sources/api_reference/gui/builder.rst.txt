phenotypic.gui.builder
======================

The pipeline builder DAG model — block + edge state, conversion to and
from :class:`~phenotypic.ImagePipeline`, and the pure-function
validation layer that drives the canvas's red/yellow border decoration.

The redesigned DAG-shaped state replaces the popover-anchored linear
list of nodes from earlier builder versions. See
:doc:`builder_dispatch` for the full dispatch-kind reference and
:doc:`builder_validation` for the seven validation rules.

State & schema
--------------

.. autosummary::
   :toctree: generated/

   phenotypic.gui.builder._state.BlockNode
   phenotypic.gui.builder._state.Edge
   phenotypic.gui.builder._state._DagBuilderScope
   phenotypic.gui.builder._state._DagBuilderState
   phenotypic.gui.builder._state.INPUT_IMAGE_CLASS_NAME
   phenotypic.gui.builder._state.PIPELINE_CLASS_NAME
   phenotypic.gui.builder._state._seed_input_image

Conversion (DAG ↔ ImagePipeline)
---------------------------------

.. autosummary::
   :toctree: generated/

   phenotypic.gui.builder._conversion_dag.to_pipeline_dag
   phenotypic.gui.builder._conversion_dag.from_pipeline_dag

Validation
----------

.. autosummary::
   :toctree: generated/

   phenotypic.gui.builder._validation.validate
   phenotypic.gui.builder._validation.Issue

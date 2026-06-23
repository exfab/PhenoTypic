Contributing Guidelines
=======================

.. note::
   This page is under construction. Check back soon for detailed contributing guidelines.

Getting Started
---------------

This section will provide information on how to get started with contributing to Phenotypic.

Code Standards
--------------

This section will describe the coding standards and style guidelines for Phenotypic.

Adding a tuning objective (Scorer)
----------------------------------

A tuning *objective* is a :class:`~phenotypic.tune.score.Scorer`. Every value the
tuner optimizes is a bounded **cost** in ``[0, 1]`` (``0`` = perfect, ``1`` =
worst) and the optimizer **minimizes** it. You emit your metric's *natural*
value and declare its *sense*; the framework orients, aggregates, and combines.

The contract (canonical in the ``Scorer`` base-class docstring and
``src/phenotypic/tune/CLAUDE.md`` — keep all three in sync):

#. **Subclass** ``Scorer`` and implement
   ``_score_terms(image, measurements) -> dict[str, float]`` returning your
   **natural** per-term values. Do **not** flip or normalize by hand — a
   divergence stays a divergence, Dice stays Dice.
#. **Declare the sense** with the ``_TERM_SENSE`` class variable:
   ``Sense.LOWER_BETTER`` (the default) if a larger value is worse;
   ``Sense.HIGHER_BETTER`` if a larger value is better (Dice, IoU, ICC,
   solidity).
#. **Supply an anchor only for an unbounded term** by overriding
   ``_term_anchor`` to return the value at which cost should reach ``0.5`` (for
   a QC-backed term, its check's ``fail_threshold``). Bounded ``[0, 1]`` terms
   need nothing.
#. **Do not add scalarization parameters.** The utopia shift ``ε``, the
   augmentation coefficient ``ρ``, per-axis normalization, and default weights
   are all framework-derived — a scorer never exposes them.
#. **Register** the class: re-export it from ``phenotypic.tune`` (its
   ``__init__``) and the class registry, or the GUI and ``from_json``
   deserialization cannot discover it.

Minimal example — a reference-free scorer that rewards round colonies (Solidity
is already ``[0, 1]`` and higher-is-better, so it only declares the sense):

.. code-block:: python

   from typing import Any, ClassVar

   import pandas as pd

   from phenotypic.tune.score import Scorer
   from phenotypic.tune.score._orient import Sense


   class SolidityScorer(Scorer):
       """Reward compact, non-jagged colonies (mean Solidity, higher = better)."""

       _TERM_SENSE: ClassVar[Sense] = Sense.HIGHER_BETTER

       def _score_terms(
           self, image: Any, measurements: pd.DataFrame
       ) -> dict[str, float]:
           # Solidity is bounded [0, 1]; emit it raw. The base score_image
           # complements it into cost (1 - value) because _TERM_SENSE is
           # HIGHER_BETTER — you write no flip.
           return {"Solidity": float(measurements["Shape_Solidity"].mean())}

The base :meth:`~phenotypic.tune.score.Scorer.score_image` template method then turns
each natural term into cost via ``to_cost``; the ``Evaluator`` robust-aggregates
(``median + λ·IQR``, clamped) and the optimizer minimizes. A
:class:`~phenotypic.tune.score.CompositeScorer` combines several scorers' per-child
cost with an **augmented Tchebycheff** scalarization (worst-axis-dominant by
default) — see the explainer
``docs/superpowers/explain/tune-with-optuna.md`` for the math.

Pull Request Workflow
---------------------

This section will explain the process for submitting pull requests to the Phenotypic repository.

Issue Reporting
---------------

This section will provide guidelines for reporting issues and bugs in Phenotypic.

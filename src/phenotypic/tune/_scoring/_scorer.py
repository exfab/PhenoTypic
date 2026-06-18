"""The pluggable scoring objective — a pydantic ABC.

A ``Scorer`` emits one image's **natural per-term values** (``_score_terms``);
the base-class template method ``score_image`` orients each term into bounded
**cost** ``∈ [0, 1]`` (``0`` = perfect, ``1`` = worst) via the shared
:func:`~phenotypic.tune._scoring._orient.to_cost`, reading the scorer's
``_TERM_SENSE`` and optional ``_term_anchor``. The ``Evaluator`` collects the
per-image cost terms across a calibration set, robust-aggregates each term
(``median + λ·IQR``, clamped to ``[0, 1]``), then asks the scorer to
``finalize`` the aggregated terms into the single scalar **the optimizer
minimizes**. ``availability`` lets a scorer report that it cannot run (e.g.
missing metadata) so the engine can degrade gracefully.

Authoring a new ``Scorer`` (the canonical contract — kept in sync with
``tune/CLAUDE.md`` and ``docs/source/contrib_guide/contributing.rst``):

  1. Implement ``_score_terms(image, measurements) -> dict[str, float]``
     returning your **natural** per-term values — do **not** flip or normalize
     by hand.
  2. Declare ``_TERM_SENSE`` (``Sense.LOWER_BETTER`` if larger = worse — the
     default; ``Sense.HIGHER_BETTER`` if larger = better, e.g. Dice/ICC).
  3. Override ``_term_anchor`` **only** if a term is unbounded, returning the
     half-cost scale (for a QC-backed term, its check's ``fail_threshold``);
     bounded ``[0, 1]`` terms need nothing.
  4. Do **not** add scalarization parameters (``ε``, ``ρ``, normalization,
     default weights are framework-derived).
  5. Register: re-export from ``tune/__init__.py`` and the class registry, or
     the GUI and ``from_json`` cannot see it.

The framework then orients (``to_cost``), robust-aggregates, reduces per child,
and combines (augmented Tchebycheff) — the author writes none of that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict

from phenotypic.sdk_.typing_ import polymorphic_field

from ._orient import Sense, to_cost


def project_objectives_to_scalar(mapping: Mapping[str, float]) -> float:
    """Project a named-objectives mapping to its mean scalar (``0.0`` if empty).

    The single canonical "mean of a dict's values, ``0.0`` for an empty dict"
    reduction shared by every multi-objective → scalar projection: the
    :meth:`Scorer.finalize` default, ``CompositeScorer._as_scalar``, and the
    ``Evaluator``'s ``_project_finalize`` sidecar all collapse a dict of named
    objectives the same way (lower = better cost, ``0.0`` is the best and the
    empty-mapping default).

    Args:
        mapping: Objective name → value (the robust-aggregated terms, or a
            scorer's named-objectives sidecar).

    Returns:
        ``mean(mapping.values())`` as a ``float``; ``0.0`` for an empty mapping
        (a perfect/empty cost).
    """
    values = list(mapping.values())
    if not values:
        return 0.0
    return float(sum(values) / len(values))


class Scorer(BaseModel, ABC):
    """Base class for tuning objectives (no-GT, supervised, reference-free, …).

    **Orientation is a base-class template method (the one cost boundary).** A
    scorer emits its *natural* per-term values in :meth:`_score_terms` (a
    divergence stays a divergence; Dice stays Dice), declares the *sense* of
    those values once via :attr:`_TERM_SENSE`, and the base :meth:`score_image`
    wraps each term into **cost ∈ [0,1]** (``0`` perfect, ``1`` worst, lower is
    better — the optimizer minimizes) via the shared :func:`to_cost` helper.

    To add a scorer (the canonical contract — kept in sync with
    ``tune/CLAUDE.md`` and ``docs/source/contrib_guide/contributing.rst``):

    1. Implement ``_score_terms(image, measurements) -> dict[str, float]``
       returning your **natural** per-term values — do **not** flip or normalize
       by hand.
    2. Declare ``_TERM_SENSE`` (``Sense.LOWER_BETTER`` if larger = worse — the
       default; ``Sense.HIGHER_BETTER`` if larger = better, e.g. Dice/ICC).
    3. Override ``_term_anchor`` **only** if a term is unbounded, returning the
       half-cost scale (for a QC-backed term, its check's ``fail_threshold``);
       bounded ``[0, 1]`` terms need nothing.
    4. Do **not** add scalarization parameters (``ε``, ``ρ``, normalization,
       default weights are framework-derived).
    5. Register: re-export from ``tune/__init__.py`` and the class registry, or
       the GUI and ``from_json`` cannot see it.

    The framework then orients (``to_cost``), robust-aggregates, reduces per
    child, and combines (augmented Tchebycheff) — the author writes none of that.

    Production scorers must be **stateless** across :meth:`score_image` calls:
    the engine reuses one scorer instance for every trial, so per-trial mutable
    state would bleed across candidates. (A test double that deliberately returns
    a preset sequence via a private cursor is the documented exception.)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Sense of this scorer's natural per-term values (v1: uniform per scorer).
    #: ``LOWER_BETTER`` is cost-native — a raw-loss scorer needs no annotation; a
    #: goodness-emitting scorer must declare ``HIGHER_BETTER``.
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER

    @abstractmethod
    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """This scorer's **natural** per-term values (its own convention).

        Args:
            image: The (already-processed) image — duck-typed; reference-free
                scorers read its mask/objmap, the ``QCScorer`` ignores it.
            measurements: The measurement frame the candidate pipeline produced
                for ``image`` (the output of ``ImagePipeline.measure``).

        Returns:
            A mapping of term name → natural value for this image (its own
            sense, not yet oriented to cost). Keys must be stable across images
            so the ``Evaluator`` can aggregate per term.
        """
        raise NotImplementedError

    def _term_anchor(self, term: str) -> float | None:
        """The half-cost anchor for an unbounded term, else ``None``.

        Args:
            term: The term name from :meth:`_score_terms`.

        Returns:
            ``None`` when the term is already bounded in ``[0,1]`` (the default —
            no anchoring); a positive float (the half-cost scale) for an
            unbounded magnitude that :func:`to_cost` should threshold-anchor.
        """
        return None

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Orient this scorer's natural terms into **cost ∈ [0,1]** (lower = better).

        The single orientation point: each natural term from
        :meth:`_score_terms` is mapped to cost via :func:`to_cost`, using this
        scorer's :attr:`_TERM_SENSE` and :meth:`_term_anchor`.

        Args:
            image: The processed image (passed through to :meth:`_score_terms`).
            measurements: The candidate pipeline's measurement frame.

        Returns:
            A mapping of term name → cost in ``[0,1]`` (``0`` perfect, ``1``
            worst). Keys are stable across images for per-term aggregation.
        """
        return {
            term: to_cost(
                value, sense=self._TERM_SENSE, anchor=self._term_anchor(term)
            )
            for term, value in self._score_terms(image, measurements).items()
        }

    def availability(self) -> bool:
        """Whether this scorer can run as configured (default: yes).

        Returns:
            ``True`` if scoring is possible; subclasses override to report a
            missing prerequisite (e.g. a layout frame the ``QCScorer`` needs).
        """
        return True

    def finalize(self, terms: Mapping[str, float]) -> float | dict[str, float]:
        """Combine robust-aggregated per-term scores into the optimizer objective.

        The default is the arithmetic mean of the term values — a single scalar
        (a single term passes through unchanged); composite single-objective
        scorers override to weight terms. A **multi-objective** scorer (plan §0a)
        may instead return a ``dict[str, float]`` of *named objectives*: the
        ``Evaluator`` then stashes that dict on ``EvaluationResult.objectives``
        and projects it to the scalar ``score`` as ``mean(objectives.values())``.
        The single-objective scalar path is byte-identical — the dict branch only
        fires when an override returns a dict.

        Args:
            terms: Term name → robust-aggregated score (already reduced across
                the calibration set by the ``Evaluator``).

        Returns:
            The scalar objective **cost** (lower = better; ``0.0`` for no terms)
            for single-objective scorers, or a ``dict[str, float]`` of named
            objectives for a multi-objective scorer.
        """
        return project_objectives_to_scalar(terms)


#: A ``Scorer``-valued field that round-trips any subclass via the ``phenotypic``
#: class registry (Phase-0 ``polymorphic_field`` + ``_find_class_in_phenotypic``
#: += ``phenotypic.tune``). Defined here — beside the ``Scorer`` base it widens,
#: the lowest module in the ``_scoring`` import graph — so both ``CompositeScorer``
#: (``_composite``) and ``TuningSpec`` (``_spec``) consume one canonical field
#: without either re-building it. Typed ``TypeAlias`` so mypy accepts the
#: ``Annotated`` core (erased to ``Any``) as a field annotation — the
#: ``AfterValidator`` restores the runtime guard.
ScorerField: TypeAlias = Any  # = polymorphic_field(base=Scorer); see below
ScorerField = polymorphic_field(base=Scorer)  # type: ignore[misc]

"""The pluggable scoring objective — a pydantic ABC.

A ``Scorer`` turns one image's measurement frame into a dict of **named term
scores** (``score_image``), where *higher is better* and each term is a clean,
comparable signal (typically normalized to ``[0, 1]``). The ``Evaluator``
collects the per-image terms across a calibration set, robust-aggregates each
term, then asks the scorer to ``finalize`` the aggregated terms into the single
scalar objective the optimizer maximizes. ``availability`` lets a scorer report
that it cannot run (e.g. missing metadata) so the engine can degrade gracefully.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict

from phenotypic.tools_.typing_ import polymorphic_field


def project_objectives_to_scalar(mapping: Mapping[str, float]) -> float:
    """Project a named-objectives mapping to its mean scalar (``0.0`` if empty).

    The single canonical "mean of a dict's values, ``0.0`` for an empty dict"
    reduction shared by every multi-objective → scalar projection: the
    :meth:`Scorer.finalize` default, ``CompositeScorer._as_scalar``, and the
    ``Evaluator``'s ``_project_finalize`` sidecar all collapse a dict of named
    objectives the same way (higher = better, ``0.0`` is the worst score).

    Args:
        mapping: Objective name → value (the robust-aggregated terms, or a
            scorer's named-objectives sidecar).

    Returns:
        ``mean(mapping.values())`` as a ``float``; ``0.0`` for an empty mapping.
    """
    values = list(mapping.values())
    if not values:
        return 0.0
    return float(sum(values) / len(values))


class Scorer(BaseModel, ABC):
    """Base class for tuning objectives (no-GT, supervised, reference-free, …).

    Production scorers must be **stateless** across ``score_image`` calls: the
    engine (Phase 1d) reuses one scorer instance for every trial, so per-trial
    mutable state would bleed across candidates. (A test double that deliberately
    returns a preset sequence via a private cursor is the documented exception.)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Score one image's measurements as named terms (higher = better).

        Args:
            image: The (already-processed) image — duck-typed; reference-free
                scorers read its mask/objmap, the ``QCScorer`` ignores it.
            measurements: The measurement frame the candidate pipeline produced
                for ``image`` (the output of ``ImagePipeline.measure``).

        Returns:
            A mapping of term name → score for this image. Keys must be stable
            across images so the ``Evaluator`` can aggregate per term.
        """
        raise NotImplementedError

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
            The scalar objective (higher = better; ``0.0`` for no terms) for
            single-objective scorers, or a ``dict[str, float]`` of named
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

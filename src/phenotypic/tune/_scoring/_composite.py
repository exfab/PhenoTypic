"""The composite scoring objective — Phase 4 chunk B (4.5).

``CompositeScorer`` nests a ``list[Scorer]`` (via the polymorphic
``ScorerField``) into one objective, the small *complementary panel* the
supervised-scorers catalogue calls for (§1: "no single metric covers split,
merge, boundary, count, and small-colony errors at once — the scorer must be a
small complementary panel"). It composes children two ways:

* **single-objective** (default): :meth:`finalize` returns one ``float`` — a
  **weighted** arithmetic blend of the per-child scalars when ``weights`` are
  given, else their **geometric** mean (so a single weak axis cannot be hidden
  by a strong one — the geometric mean is dominated by its smallest term).
* **multi-objective** (``multi_objective=True``): :meth:`finalize` returns a
  ``dict[str, float]`` of per-child objectives — the plan §0a *sidecar* path.
  The ``Evaluator`` stashes that dict on ``EvaluationResult.objectives`` and
  projects it to the scalar ``score`` as ``mean(objectives.values())``, so a
  composite can drive a true multi-objective (Pareto) study.

**Collision-free merging.** :meth:`score_image` prefixes every child term with
the child's positional handle ``s{i}`` (``"s0.Region"``, ``"s1.Count"``), so two
children emitting the same term name (e.g. both a ``"Count"``) never clash and
:meth:`finalize` can re-group the merged terms back to their originating child.

**Cycle / self-nesting rejection.** A ``model_validator`` walks the nested
scorer graph by object identity and rejects any composite reachable from itself
(direct self-nesting or a deeper back-edge), so a malformed recipe fails at
construction rather than recursing forever at score time.
"""
from __future__ import annotations

import math
from typing import Any, Final, Mapping, Optional, TypeAlias

import pandas as pd
from pydantic import ConfigDict, model_validator

from phenotypic.tools_.typing_ import polymorphic_field

from ._scorer import Scorer

#: A ``Scorer``-valued field that round-trips any subclass via the ``phenotypic``
#: class registry — defined here (not imported from ``.._spec``) so ``_composite``
#: stays *below* ``_spec`` in the import graph and the ``_scoring`` package can
#: re-export ``CompositeScorer`` without dragging in the evaluation/strategy
#: stack. Identical machinery to ``_spec.ScorerField`` (same ``polymorphic_field``
#: base), so a child round-trips the same way through either field.
ScorerField: TypeAlias = Any  # = polymorphic_field(base=Scorer); see below
ScorerField = polymorphic_field(base=Scorer)  # type: ignore[misc]

#: The per-child handle prefix: child ``i`` owns the ``"s{i}."`` term namespace,
#: so collisions across children are impossible and ``finalize`` can re-group
#: merged terms by their leading handle.
_CHILD_HANDLE: Final[str] = "s"

#: The separator between a child handle and the child's own term name.
_SEP: Final[str] = "."


class CompositeScorer(Scorer):
    """Blend several ``Scorer`` children into one objective (scalar or Pareto).

    Args:
        scorers: The child objectives to compose. Each is a ``ScorerField`` so
            any ``Scorer`` subclass — including a nested ``CompositeScorer`` —
            round-trips through the polymorphic registry. Child ``i`` owns the
            ``"s{i}."`` term namespace.
        weights: Optional per-child weights for the **single-objective** scalar
            blend, keyed by child handle (``"s0"``, ``"s1"``, …). When given,
            :meth:`finalize` returns the weighted arithmetic mean of the
            per-child scalars; when ``None`` it returns their geometric mean.
            Ignored when ``multi_objective`` is ``True``.
        multi_objective: When ``True``, :meth:`finalize` returns a
            ``dict[str, float]`` of per-child objectives (the plan §0a sidecar)
            instead of a scalar, so the composite can drive a Pareto study.

    Raises:
        pydantic.ValidationError: If the nested scorer graph contains a cycle —
            a composite reachable from itself (direct self-nesting or a deeper
            back-edge).

    Examples:
        Compose two children and read the prefixed, merged per-image terms (the
        ``QCScorer`` here scores a perfect 96-well count match):

        >>> import pandas as pd
        >>> from phenotypic.analysis import ExpectedVsDetectedCount
        >>> from phenotypic.tune import CompositeScorer, QCScorer
        >>> layout = pd.DataFrame(
        ...     {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
        ... )
        >>> qc = QCScorer(
        ...     check=ExpectedVsDetectedCount(
        ...         metadata=layout, groupby=["Metadata_ImageName"]
        ...     )
        ... )
        >>> comp = CompositeScorer(scorers=[qc, qc])
        >>> terms = comp.score_image(None, layout)
        >>> sorted(terms)
        ['s0.Count', 's1.Count']
        >>> round(comp.finalize(terms), 3)  # geometric mean of the two child scalars
        1.0

        Flip to multi-objective and ``finalize`` returns the per-child sidecar:

        >>> comp_mo = CompositeScorer(scorers=[qc, qc], multi_objective=True)
        >>> {k: round(v, 3) for k, v in comp_mo.finalize(terms).items()}
        {'s0': 1.0, 's1': 1.0}
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scorers: list[ScorerField] = []
    weights: Optional[dict[str, float]] = None
    multi_objective: bool = False

    @model_validator(mode="after")
    def _reject_cycles(self) -> "CompositeScorer":
        """Reject a nested scorer graph in which a composite reaches itself.

        Walks the composite tree by object identity from each composite node and
        rejects the recipe if any ``CompositeScorer`` is reachable from itself
        (direct self-nesting ``scorers=[self]`` or a deeper back-edge). This runs
        at construction / validation so a malformed recipe fails fast instead of
        recursing forever in :meth:`score_image`.

        Returns:
            ``self`` (unchanged) when the graph is acyclic.

        Raises:
            ValueError: If a cycle (a composite reachable from itself) exists.
        """

        def _visit(node: CompositeScorer, ancestors: frozenset[int]) -> None:
            if id(node) in ancestors:
                raise ValueError(
                    "CompositeScorer nesting contains a cycle: a composite is "
                    "reachable from itself (self-nesting or a back-edge). The "
                    "nested scorer graph must be acyclic."
                )
            deeper = ancestors | {id(node)}
            for child in node.scorers:
                if isinstance(child, CompositeScorer):
                    _visit(child, deeper)

        _visit(self, frozenset())
        return self

    def _handle(self, index: int) -> str:
        """The term-namespace handle for child ``index`` (``"s0"``, ``"s1"``…).

        Args:
            index: The child's position in :attr:`scorers`.

        Returns:
            The child handle string.
        """
        return f"{_CHILD_HANDLE}{index}"

    def availability(self) -> bool:
        """Whether the composite can contribute any signal.

        **Pinned rule:** a composite is available iff **at least one** child is
        available. A child that abstains contributes no terms, but as long as one
        child can score, the composite produces a usable (partial) objective; an
        empty composite, or one all of whose children abstain, is unavailable and
        the engine degrades to its fallback.

        Returns:
            ``True`` if any child reports :meth:`Scorer.availability`; ``False``
            for an empty composite or one whose children all abstain.
        """
        return any(child.availability() for child in self.scorers)

    def score_image(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Merge every child's per-image terms under a per-child prefix.

        Each child ``i`` is scored on ``(image, measurements)`` and its terms are
        re-keyed ``"s{i}.<term>"`` so two children emitting the same term name
        never collide and :meth:`finalize` can re-group them.

        Args:
            image: The processed image, passed to every child unchanged.
            measurements: The candidate pipeline's measurement frame, passed to
                every child unchanged.

        Returns:
            The union of all children's terms, each prefixed with its child
            handle.
        """
        merged: dict[str, float] = {}
        for index, child in enumerate(self.scorers):
            handle = self._handle(index)
            for term, value in child.score_image(image, measurements).items():
                merged[f"{handle}{_SEP}{term}"] = float(value)
        return merged

    def finalize(
        self, terms: Mapping[str, float]
    ) -> float | dict[str, float]:
        """Blend the per-child scalars — scalar (default) or dict (Pareto).

        Re-groups the prefixed ``terms`` back to their originating child, calls
        each child's own :meth:`Scorer.finalize` over its un-prefixed sub-terms
        (projecting a child's own multi-objective dict to its mean), then:

        * ``multi_objective=True`` → returns ``{handle: child_scalar}`` — the
          plan §0a sidecar the ``Evaluator`` stashes on
          ``EvaluationResult.objectives``.
        * with :attr:`weights` → the weighted arithmetic mean of the per-child
          scalars (missing weights default to ``1.0``).
        * otherwise → the geometric mean of the per-child scalars (so a single
          weak axis dominates — it cannot be masked by a strong one).

        Args:
            terms: The robust-aggregated, child-prefixed terms (the output of
                :meth:`score_image` after the ``Evaluator``'s per-term
                aggregation).

        Returns:
            The scalar objective (``0.0`` for no children/terms) for the
            single-objective path, or the per-child ``dict`` for the
            multi-objective path.
        """
        child_scalars = self._per_child_scalars(terms)
        if self.multi_objective:
            return child_scalars
        values = list(child_scalars.values())
        if not values:
            return 0.0
        if self.weights is not None:
            return self._weighted_mean(child_scalars)
        return self._geometric_mean(values)

    def _per_child_scalars(
        self, terms: Mapping[str, float]
    ) -> dict[str, float]:
        """Project the merged terms to one finalized scalar per child.

        Args:
            terms: The child-prefixed terms from :meth:`score_image`.

        Returns:
            ``{handle: scalar}`` — each child's own :meth:`Scorer.finalize` over
            its un-prefixed sub-terms, with a child's own multi-objective dict
            projected to its mean. Children with no terms in ``terms`` are
            omitted.
        """
        scalars: dict[str, float] = {}
        for index, child in enumerate(self.scorers):
            handle = self._handle(index)
            prefix = f"{handle}{_SEP}"
            sub = {
                key[len(prefix):]: value
                for key, value in terms.items()
                if key.startswith(prefix)
            }
            if not sub:
                continue
            scalars[handle] = self._as_scalar(child.finalize(sub))
        return scalars

    @staticmethod
    def _as_scalar(finalized: float | Mapping[str, float]) -> float:
        """Reduce a child's ``finalize`` result to a scalar.

        Mirrors the ``Evaluator``'s sidecar projection: a child that is itself a
        multi-objective composite returns a ``dict``, projected here to the mean
        of its values so the parent can still blend a single number per child.

        Args:
            finalized: The child's :meth:`Scorer.finalize` return — a ``float``
                or a ``dict`` of named objectives.

        Returns:
            The scalar form (``mean`` of a dict's values; ``0.0`` for an empty
            dict).
        """
        if isinstance(finalized, Mapping):
            vals = list(finalized.values())
            return float(sum(vals) / len(vals)) if vals else 0.0
        return float(finalized)

    def _weighted_mean(self, child_scalars: dict[str, float]) -> float:
        """The weight-weighted arithmetic mean of the per-child scalars.

        Args:
            child_scalars: ``{handle: scalar}`` per scored child.

        Returns:
            ``Σ wᵢ·sᵢ / Σ wᵢ`` over scored children (missing weights default to
            ``1.0``); ``0.0`` if the total weight is zero.
        """
        weights = self.weights or {}
        total_weight = 0.0
        weighted_sum = 0.0
        for handle, scalar in child_scalars.items():
            weight = float(weights.get(handle, 1.0))
            weighted_sum += weight * scalar
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _geometric_mean(values: list[float]) -> float:
        """The geometric mean of non-negative per-child scalars.

        A single near-zero axis drags the product toward ``0`` — the property
        that makes the geometric mean the default composite blend (a weak axis
        cannot be masked by a strong one).

        Args:
            values: The per-child scalars (clamped at ``0`` — scores are
                higher-is-better in ``[0, 1]``, never negative).

        Returns:
            ``(Π max(vᵢ, 0))^(1/n)``; ``0.0`` for an empty list.
        """
        if not values:
            return 0.0
        product = 1.0
        for value in values:
            product *= max(value, 0.0)
        return float(math.pow(product, 1.0 / len(values)))

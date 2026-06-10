"""The composite scoring objective — Phase 4 chunk B (4.5).

``CompositeScorer`` nests a ``list[Scorer]`` (via the polymorphic
``ScorerField``) into one objective, the small *complementary panel* the
supervised-scorers catalogue calls for (§1: "no single metric covers split,
merge, boundary, count, and small-colony errors at once — the scorer must be a
small complementary panel"). It composes children two ways:

* **single-objective** (default): :meth:`finalize` returns one ``float`` — the
  conjunctive **augmented Tchebycheff** cost over the per-child cost scalars
  (``blend="tchebycheff"``), so a single weak (high-cost) axis dominates the
  ``max`` and cannot be masked by a strong one. ``blend="weighted_mean"`` is the
  compensatory arithmetic-mean opt-out (``weights`` then weight that mean). The
  geometric-mean blend is no longer offered (it inverts the conjunctive property
  under cost — a single perfect axis would annihilate the product).
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

from typing import Any, Final, Mapping, Optional

import pandas as pd
from pydantic import ConfigDict, PrivateAttr, model_validator

from phenotypic.tools_.typing_ import CompositeBlend

from ._orient import clamp01
from ._scorer import Scorer, ScorerField, project_objectives_to_scalar

#: The per-child handle prefix: child ``i`` owns the ``"s{i}."`` term namespace,
#: so collisions across children are impossible and ``finalize`` can re-group
#: merged terms by their leading handle.
_CHILD_HANDLE: Final[str] = "s"

#: The separator between a child handle and the child's own term name.
_SEP: Final[str] = "."

#: The utopia-point shift ``z*ᵢ = −ε`` for the augmented Tchebycheff combiner.
#: A small, fixed numerical safety margin (~0.1% of the [0,1] cost scale): it
#: pushes the reference strictly below the achievable front so every
#: ``bᵢ − z*ᵢ = bᵢ + ε > 0`` (the unsigned ``max`` is valid) and caps the
#: weight realizer ``1/(bᵢ + ε)`` at ``≤ 1000×``. Internal — never a field
#: (spec §6.4/§6.6).
_UTOPIA_EPS: Final[float] = 1e-3


class CompositeScorer(Scorer):
    """Blend several ``Scorer`` children into one objective (scalar or Pareto).

    Args:
        scorers: The child objectives to compose. Each is a ``ScorerField`` so
            any ``Scorer`` subclass — including a nested ``CompositeScorer`` —
            round-trips through the polymorphic registry. Child ``i`` owns the
            ``"s{i}."`` term namespace.
        weights: Optional per-child weights for the **single-objective** scalar
            blend, keyed by child handle (``"s0"``, ``"s1"``, …). The meaning is
            **blend-dependent**: under ``blend="tchebycheff"`` they are per-axis
            Tchebycheff weights (they steer the ``max`` term, uniform ``1.0``
            when ``None``); under ``blend="weighted_mean"`` they are arithmetic
            weights for the compensatory mean. Ignored when ``multi_objective``
            is ``True``.
        multi_objective: When ``True``, :meth:`finalize` returns a
            ``dict[str, float]`` of per-child objectives (the plan §0a sidecar)
            instead of a scalar, so the composite can drive a Pareto study.
        blend: The single-objective combiner. ``"tchebycheff"`` (default) is the
            conjunctive augmented-Tchebycheff cost over the pinned study-global
            active set — worst-axis-dominant, so a single weak axis cannot be
            masked by a strong one. ``"weighted_mean"`` is the compensatory
            arithmetic-mean opt-out. The geometric-mean-of-cost blend is no
            longer offered (it inverts the conjunctive property — a single
            perfect axis would annihilate the product).
        rho: The augmented-Tchebycheff augmentation coefficient (advanced-only;
            default ``0.05``). It scales the weighted-L1 term that breaks ties
            between weakly-dominated points; ``→0`` recovers the plain
            Tchebycheff (admits weakly-dominated winners), large values drift
            toward the weighted sum (spec §6.4). Ignored under
            ``blend="weighted_mean"`` / ``multi_objective``.

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
        >>> round(comp.finalize(terms), 3)  # augmented Tchebycheff of two perfect (cost-0) children
        0.001

        Flip to multi-objective and ``finalize`` returns the per-child sidecar:

        >>> comp_mo = CompositeScorer(scorers=[qc, qc], multi_objective=True)
        >>> {k: round(v, 3) for k, v in comp_mo.finalize(terms).items()}
        {'s0': 0.0, 's1': 0.0}
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scorers: list[ScorerField] = []
    weights: Optional[dict[str, float]] = None
    multi_objective: bool = False
    blend: CompositeBlend = "tchebycheff"
    rho: float = 0.05

    #: The pinned study-global active set — child handles available study-wide,
    #: fixed once at study start by :meth:`set_active_set`. Used as the roster for
    #: BOTH the Tchebycheff ``max`` numerator and the normalizer so the
    #: normalizer is a study-global constant (§6.2/§6.3). ``None`` (never pinned —
    #: e.g. a direct ``finalize`` unit call) falls back to the in-call roster.
    #: A ``PrivateAttr`` so it never serializes (it is run/study state, not recipe).
    _active_handles: Optional[tuple[str, ...]] = PrivateAttr(default=None)

    def set_active_set(self, handles: tuple[str, ...]) -> None:
        """Pin the study-global active set (child handles available study-wide).

        Called once by the engine after meta-validation, before the trial loop,
        so every trial's Tchebycheff ``max`` and normalizer use the same fixed
        roster (§6.3 plumbing SF3). Idempotent.

        Args:
            handles: The available child handles (a subset of
                :meth:`objective_names`), in objective order.
        """
        self._active_handles = tuple(handles)

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

    def objective_names(self) -> list[str]:
        """The ordered objective-axis names of a multi-objective composite.

        Returns the per-child handles (``["s0", "s1", …]``) in :attr:`scorers`
        order — exactly the keys :meth:`finalize` emits when
        :attr:`multi_objective` is ``True``, and therefore the
        ``Trial.objectives`` / ``objectives_json`` keys and the ``pareto/`` axis
        labels. Stable across a study so the per-objective directions, the
        per-objective best pipelines, and the front parquet all align.

        Returns:
            The child handles in order; ``[]`` for an empty composite.
        """
        return [self._handle(index) for index in range(len(self.scorers))]

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

    def _score_terms(
        self, image: Any, measurements: pd.DataFrame
    ) -> dict[str, float]:
        """Not used — the composite overrides :meth:`score_image` instead.

        ``_score_terms`` is abstract on the base, but a composite's children
        already returned **cost** (each child's own ``score_image`` oriented its
        terms), so the composite must **not** re-orient: it overrides the merge
        in :meth:`score_image` directly. This stub satisfies the abstract base.

        Raises:
            NotImplementedError: Always — call :meth:`score_image`.
        """
        raise NotImplementedError(
            "CompositeScorer overrides score_image (it merges already-cost "
            "children); _score_terms is not used."
        )

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

        * ``multi_objective=True`` → returns ``{handle: child_scalar}`` over
          **every** axis in :meth:`objective_names` order — the plan §0a sidecar
          the ``Evaluator`` stashes on ``EvaluationResult.objectives``. A child
          that abstains (no terms this run) is floored to ``1.0`` (the worst
          cost; the study minimizes) rather than dropped, so the dict keys +
          order stay invariant and exactly match the multi-objective study's
          fixed ``directions`` (now ``minimize``). A dropped axis would otherwise
          make the NSGA-II value vector the wrong length and crash Optuna's
          ``tell`` mid-run; the ``1.0`` floor also mirrors the journal Pareto
          ``_vector`` fill so the journal and Optuna backends agree on an
          abstaining axis.
        * ``blend="weighted_mean"`` → the compensatory weighted arithmetic mean
          of the per-child **cost** scalars (missing weights default to ``1.0``).
        * ``blend="tchebycheff"`` (default) → the conjunctive augmented
          Tchebycheff cost (:meth:`_tchebycheff`) over the pinned study-global
          active set, so a single weak (high-cost) axis dominates the ``max`` and
          cannot be masked by a strong one. The roster is restricted to the
          pinned active set (the children available study-wide, §6.3): a
          study-wide abstainer is simply not an objective (dropped from both the
          ``max`` and the normalizer), while per-image abstention is already a
          fewer-samples matter handled upstream by the robust aggregate — so a
          present-but-absent-this-call child is NOT flooded into the ``max``.

        Args:
            terms: The robust-aggregated, child-prefixed terms (the output of
                :meth:`score_image` after the ``Evaluator``'s per-term
                aggregation).

        Returns:
            The scalar objective for the single-objective path (worst cost
            ``1.0`` for no scored children/terms), or the per-child ``dict`` (one
            entry per axis, abstainers floored) for the multi-objective path.
        """
        child_scalars = self._per_child_scalars(terms)
        if self.multi_objective:
            return {
                handle: child_scalars.get(handle, 1.0)  # was 0.0 (goodness worst)
                for handle in self.objective_names()
            }
        if self.blend == "weighted_mean":
            if not child_scalars:
                return 1.0  # worst cost (cost-convention floor; was 0.0 goodness)
            return self._weighted_mean(child_scalars)
        # Default conjunctive blend: augmented Tchebycheff over the pinned
        # study-global active set (§6.3). Restrict to the active roster so a
        # study-wide abstainer is simply not an objective (dropped from both the
        # max and the normalizer); per-image abstention is already a fewer-samples
        # matter handled by the robust aggregate, so a present-but-absent-this-call
        # child is NOT flooded into the max.
        if self._active_handles is None:
            roster = dict(child_scalars)
        else:
            roster = {
                handle: child_scalars[handle]
                for handle in self._active_handles
                if handle in child_scalars
            }
        return self._tchebycheff(roster)

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
            return project_objectives_to_scalar(finalized)
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

    def _tchebycheff(self, child_costs: dict[str, float]) -> float:
        """Augmented weighted Tchebycheff of per-child **cost** scalars (§6.1/§6.2).

        ``Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε) + ρ·Σᵢ wᵢ·bᵢ``, minimized, with utopia point
        ``z*ᵢ = −ε`` (``_UTOPIA_EPS``). The ``max`` drops the absolute value
        because ``z*ᵢ = −ε < 0 ≤ bᵢ`` makes every ``bᵢ − z*ᵢ = bᵢ + ε > 0`` — an
        invariant asserted here (the Phase 1 ``[0,1]`` clamp guarantees the upper
        bound; the assert fires loudly if that clamp ever regresses).

        **Normalization is a study-global constant (§6.2).** The raw ``Tᵨ``
        ranges over ``[ε·(…), (1+ε)·(…)]``, so it is normalized by the theoretical
        worst ``Tᵨ(1…1)`` to give a normalized cost in ``(0, 1]``. That worst-case
        denominator is computed over the **full pinned active set**
        (:attr:`_active_handles`) — every child available study-wide — **not**
        the in-call roster. Otherwise a pinned child that produces zero terms
        across a whole trial's calibration set would drop the trial's denominator
        from ``(1+ε)+ρ·n`` to ``(1+ε)+ρ·(n−1)`` while other trials keep ``n``,
        normalizing different trials by different constants and perturbing the
        cross-trial argmin (the winner-equivalence this migration must preserve).
        The **numerator** (``max_term`` / ``l1``) stays over the in-call
        ``child_costs``: a child that genuinely abstained this trial contributes
        no cost to the numerator, which is correct. When :attr:`_active_handles`
        is ``None`` (direct-``finalize`` unit calls with no engine pin), the
        denominator falls back to the in-call roster.

        Args:
            child_costs: ``{handle: cost}`` over the children that scored this
                call (each ``bᵢ ∈ [0,1]``).

        Returns:
            The normalized composite cost in ``(0, 1]``. Empty roster → ``1.0``
            (the worst floor; the engine degrades — guards the ``max([])``).
        """
        if not child_costs:
            return 1.0
        eps = _UTOPIA_EPS
        weights = self.weights or {}
        max_term = 0.0
        l1 = 0.0
        for handle, cost in child_costs.items():
            assert 0.0 <= cost <= 1.0, (  # noqa: S101 — B1 invariant guard
                f"per-child cost {handle}={cost!r} escaped [0,1]; the Phase 1 "
                "robust-aggregate clamp regressed"
            )
            weight = float(weights.get(handle, 1.0))
            max_term = max(max_term, weight * (cost + eps))
            l1 += weight * cost
        # Denominator roster = the full pinned active set (study-global constant),
        # falling back to the in-call roster when unpinned (direct-finalize unit
        # calls). Each pinned handle contributes its weight at the worst cost 1.0,
        # independent of whether it scored this call.
        denom_handles = (
            self._active_handles if self._active_handles is not None
            else tuple(child_costs)
        )
        denom_max = 0.0
        denom_l1 = 0.0
        for handle in denom_handles:
            weight = float(weights.get(handle, 1.0))
            denom_max = max(denom_max, weight * (1.0 + eps))
            denom_l1 += weight
        numerator = max_term + self.rho * l1
        denominator = denom_max + self.rho * denom_l1
        if denominator <= 0.0:
            return 1.0
        return clamp01(numerator / denominator)

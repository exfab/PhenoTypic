"""The candidate evaluator — the uniform 3-step robust-evaluation loop.

For one parameter combo: build the candidate pipeline, ``score_image`` over the
calibration set, robust-aggregate each term as ``median - λ·IQR`` (the spread
penalty rewards parameters that are stable across images, not just good on
average), then ``finalize`` to the scalar objective the optimizer maximizes.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from .._scoring._scorer import Scorer
from .._strategies._pruning import NoOpChannel, PruningChannel
from ._builder import build_pipeline

#: The worst possible per-image term score (higher-is-better objective floor).
#: A per-image exception contributes this to every term so it honestly drags
#: the aggregate (robust-eval §10) rather than dodging a bad plate by crashing.
_WORST_TERM = 0.0


def _project_finalize(
    finalized: float | Mapping[str, float],
) -> tuple[float, Optional[dict[str, float]]]:
    """Split a ``Scorer.finalize`` result into ``(scalar_score, objectives)``.

    The multi-objective sidecar (plan §0a): a scalar ``finalized`` is the
    single-objective path and carries no sidecar (``objectives is None``); a
    ``dict`` of named objectives is stashed as ``objectives`` and projected to the
    scalar ``score`` as ``mean(objectives.values())`` (``0.0`` for an empty dict).
    The single-objective branch is byte-identical to the pre-sidecar code.

    Args:
        finalized: The return of ``scorer.finalize`` — a ``float`` for a
            single-objective scorer or a ``dict[str, float]`` of named objectives
            for a multi-objective one.

    Returns:
        A ``(score, objectives)`` pair: ``objectives is None`` on the scalar path,
        otherwise the named-objectives dict with ``score`` its mean projection.
    """
    if isinstance(finalized, Mapping):
        objectives = {key: float(value) for key, value in finalized.items()}
        values = list(objectives.values())
        score = float(sum(values) / len(values)) if values else 0.0
        return score, objectives
    return float(finalized), None


def _robust_aggregate(values: list[float], stability_weight: float) -> float:
    """Reduce a term's per-image scores to ``median - stability_weight·IQR``.

    Args:
        values: The per-image scores for one term (higher = better).
        stability_weight: λ — how hard cross-image spread is penalized.

    Returns:
        The stability-penalized central tendency. For a single value the IQR is
        ``0`` and the result is that value.
    """
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return median - stability_weight * float(q75 - q25)


class EvaluationResult(BaseModel):
    """The outcome of evaluating one candidate over the calibration set.

    Args:
        score: The finalized scalar objective (higher = better). For a
            multi-objective candidate this is the **scalar projection** of
            ``objectives`` (``mean(objectives.values())``).
        terms: Robust-aggregated per-term scores (``median - λ·IQR`` each).
        n_images: Number of calibration images evaluated.
        objectives: The named multi-objective values (plan §0a sidecar), or
            ``None`` for a single-objective candidate. Set only when the scorer's
            ``finalize`` returns a ``dict``; ``score`` is then their mean. The
            sidecar leaves the single-objective scalar path untouched.
        failed: ``True`` when the candidate raised and was floored to
            ``failure_score``.
        pruned: ``True`` when the rung ladder early-stopped this candidate via
            the pruning channel. Distinct from ``failed``: a pruned trial ran
            cleanly on a partial set and carries its partial aggregate.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    terms: dict[str, float]
    n_images: int
    objectives: Optional[dict[str, float]] = None
    failed: bool = False
    pruned: bool = False


class Evaluator(BaseModel):
    """Score a candidate combo over a calibration set (CV-only MVP).

    Args:
        stability_weight: λ in ``median - λ·IQR`` — how hard cross-image spread
            is penalized when aggregating a term across the calibration set.
        failure_score: The score assigned when a candidate fails to build,
            measure, or score; the floor of the higher-is-better objective.
        rung_floor: The minimum first-rung size for the ASHA-style fidelity
            ladder (robust-eval §7) — never prune on fewer plates than this.
        rung_factor: The geometric growth factor between rungs (×3 by default).
        min_rungs: The fewest distinct rungs worth running a ladder for; below
            this the ladder self-disables to a single full-fidelity rung.
    """

    model_config = ConfigDict(frozen=True)

    stability_weight: float = 0.5
    failure_score: float = 0.0
    rung_floor: int = 6
    rung_factor: int = 3
    min_rungs: int = 2

    def _rung_sizes(self, n_images: int) -> list[int]:
        """The cumulative rung sizes for ``n_images`` calibration plates.

        First rung = ``max(rung_floor, ceil(n / rung_factor))``; each subsequent
        rung multiplies by ``rung_factor``; the last rung is always all images.
        Self-disables to a single ``[n_images]`` rung when the set is too small
        to yield ``min_rungs`` distinct rungs (robust-eval §7 — never prune on a
        few unrepresentative plates).

        Args:
            n_images: The total calibration-image count (``>= 1``).

        Returns:
            Strictly increasing cumulative rung sizes ending at ``n_images``.
        """
        first = max(self.rung_floor, math.ceil(n_images / self.rung_factor))
        if first >= n_images:
            return [n_images]  # cannot split → single full-fidelity rung
        sizes = [first]
        while sizes[-1] * self.rung_factor < n_images:
            sizes.append(sizes[-1] * self.rung_factor)
        sizes.append(n_images)
        if len(sizes) < self.min_rungs:
            return [n_images]
        return sizes

    def evaluate(
        self,
        base: Any,
        scorer: Scorer,
        params: dict[str, Any],
        images: list,
        *,
        channel: PruningChannel = NoOpChannel(),
    ) -> EvaluationResult:
        """Build, score over a rung ladder, robust-aggregate, and finalize.

        The candidate is scored in growing rung blocks (:meth:`_rung_sizes`) over
        a **deterministic, id-sorted** subset (metadata-stratified rungs are
        deferred). After each rung the running ``median - λ·IQR`` is reported to
        ``channel`` and ``channel.should_prune()`` is checked *between* rungs; a
        prune short-circuits to a partial ``EvaluationResult(pruned=True)``. Each
        image is scored **once** (memoized across rungs). Failure taxonomy
        (robust-eval §10): a candidate that won't build is a true ``failed``; one
        image raising mid-scoring contributes the worst term and the loop
        continues; only **all** images erroring is a whole-candidate ``failed``.

        Args:
            base: The base pipeline embedded in the ``TuningSpec``.
            scorer: The objective.
            params: The sampled combo (``{root-relative-key: value}``).
            images: The calibration images (must be non-empty).
            channel: The pruning channel (default :class:`NoOpChannel`, which
                never prunes). With the no-op default the unpruned full pass is
                identical to a single full-set pass.

        Returns:
            The candidate's :class:`EvaluationResult`.

        Raises:
            ValueError: If ``images`` is empty.
        """
        if not images:
            raise ValueError(
                "Evaluator.evaluate requires at least one calibration image"
            )

        try:
            candidate = build_pipeline(base, params)
        except Exception:
            # Candidate won't build → a true FAIL, no scoring (robust-eval §10).
            return EvaluationResult(
                score=self.failure_score,
                terms={},
                n_images=len(images),
                failed=True,
            )

        ordered = sorted(images, key=id)
        rungs = self._rung_sizes(len(ordered))

        per_term: dict[str, list[float]] = {}
        n_exceptions = 0
        scored = 0
        for rung_index, cutoff in enumerate(rungs):
            for image in ordered[scored:cutoff]:
                raised = self._score_one_image(candidate, scorer, image, per_term)
                if raised:
                    n_exceptions += 1
            scored = cutoff

            # All images errored so far AND that is the whole set → FAIL.
            if n_exceptions == scored == len(ordered):
                return EvaluationResult(
                    score=self.failure_score,
                    terms={},
                    n_images=len(ordered),
                    failed=True,
                )

            running = self._aggregate(per_term, n_exceptions)
            running_score, running_objectives = _project_finalize(
                scorer.finalize(running)
            )
            channel.report(running_score, scored)
            # Check between rungs only (never after the final, full-fidelity rung).
            if rung_index < len(rungs) - 1 and channel.should_prune():
                return EvaluationResult(
                    score=running_score,
                    terms=running,
                    n_images=scored,
                    objectives=running_objectives,
                    pruned=True,
                )

        aggregated = self._aggregate(per_term, n_exceptions)
        final_score, final_objectives = _project_finalize(
            scorer.finalize(aggregated)
        )
        return EvaluationResult(
            score=final_score,
            terms=aggregated,
            n_images=len(ordered),
            objectives=final_objectives,
        )

    @staticmethod
    def _score_one_image(
        candidate: Any,
        scorer: Scorer,
        image: Any,
        per_term: dict[str, list[float]],
    ) -> bool:
        """Measure + score one image, appending each term to ``per_term``.

        Args:
            candidate: The built candidate pipeline.
            scorer: The objective.
            image: The image to measure and score.
            per_term: The term → per-image-scores accumulator (mutated).

        Returns:
            ``True`` if measuring/scoring this image raised (a per-image
            exception — the caller records it as a worst-term contribution),
            ``False`` on a clean score.
        """
        try:
            measurements = candidate.measure(image, apply_post=False)
            for term, value in scorer.score_image(image, measurements).items():
                per_term.setdefault(term, []).append(float(value))
        except Exception:
            return True
        return False

    def _aggregate(
        self, per_term: dict[str, list[float]], n_exceptions: int
    ) -> dict[str, float]:
        """Robust-aggregate each term, padding worst-term values for failures.

        Each per-image exception contributes :data:`_WORST_TERM` to every term
        so the failing plate honestly drags the aggregate (robust-eval §10).

        Args:
            per_term: Term → clean per-image scores.
            n_exceptions: How many images raised (padded into every term).

        Returns:
            Term → robust-aggregated score.
        """
        return {
            term: _robust_aggregate(
                values + [_WORST_TERM] * n_exceptions, self.stability_weight
            )
            for term, values in per_term.items()
        }

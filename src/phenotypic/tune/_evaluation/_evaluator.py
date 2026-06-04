"""The candidate evaluator — the uniform 3-step robust-evaluation loop.

For one parameter combo: build the candidate pipeline, ``score_image`` over the
calibration set, robust-aggregate each term as ``median - λ·IQR`` (the spread
penalty rewards parameters that are stable across images, not just good on
average), then ``finalize`` to the scalar objective the optimizer maximizes.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .._scoring._scorer import Scorer
from .._strategies._pruning import NoOpChannel, PruningChannel
from ._builder import build_pipeline

#: The worst possible per-image term score (higher-is-better objective floor).
#: A per-image exception contributes this to every term so it honestly drags
#: the aggregate (robust-eval §10) rather than dodging a bad plate by crashing.
_WORST_TERM = 0.0


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
        score: The finalized scalar objective (higher = better).
        terms: Robust-aggregated per-term scores (``median - λ·IQR`` each).
        n_images: Number of calibration images evaluated.
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
            channel.report(float(scorer.finalize(running)), scored)
            # Check between rungs only (never after the final, full-fidelity rung).
            if rung_index < len(rungs) - 1 and channel.should_prune():
                return EvaluationResult(
                    score=float(scorer.finalize(running)),
                    terms=running,
                    n_images=scored,
                    pruned=True,
                )

        aggregated = self._aggregate(per_term, n_exceptions)
        return EvaluationResult(
            score=float(scorer.finalize(aggregated)),
            terms=aggregated,
            n_images=len(ordered),
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

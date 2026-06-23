"""The candidate evaluator — the uniform 3-step robust-evaluation loop.

For one parameter combo: build the candidate pipeline, ``score_image`` over the
calibration set, robust-aggregate each term as ``clamp01(median + λ·IQR)`` (the
spread penalty pushes the cost up, toward the worst), then ``finalize`` to the
scalar cost the optimizer **minimizes** (lower = better).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from ..score._orient import clamp01
from ..score._scorer import Scorer, project_objectives_to_scalar
from ..strategy._pruning import NoOpChannel, PruningChannel
from ._aggregate_math import _median_iqr, _relative
from ._builder import build_pipeline

#: The worst possible per-image term **cost** (lower-is-better objective
#: ceiling). A per-image exception contributes this to every term so it
#: honestly drags the aggregate toward the worst (robust-eval §10) rather than
#: dodging a bad plate by crashing.
_WORST_TERM = 1.0


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
        score = project_objectives_to_scalar(objectives)
        return score, objectives
    return float(finalized), None


def _robust_aggregate(values: list[float], stability_weight: float) -> float:
    """Reduce a term's per-image **costs** to ``clamp01(median + λ·IQR)``.

    Cost convention (lower = better): the spread penalty *adds* to the central
    tendency, so an unstable term is penalized toward the worst cost (``1``).
    The reflected aggregate ranges ``[0, 1+λ]``, so it is clamped to ``[0,1]``
    (B1) — the clamp is monotone and only bites on *terrible* terms (cost > 1,
    i.e. unstable **and** bad), so it is winner-preserving.

    Args:
        values: The per-image costs for one term (lower = better).
        stability_weight: λ — how hard cross-image spread is penalized.

    Returns:
        The clamped stability-penalized central tendency in ``[0,1]``. For a
        single value the IQR is ``0`` and the result is that value.
    """
    median, iqr = _median_iqr(values)
    return clamp01(median + stability_weight * iqr)


def _per_trial_dispersion(
    per_term_scores: dict[str, list[float]], *, min_n: int
) -> Optional[float]:
    """The relative across-plate dispersion of the **primary (first) term**.

    The per-trial ``gap`` signal: the relative IQR
    ``(q75 - q25) / max(1 - median, eps)`` of the first term's per-image scores — a
    cheap instability / overfit-risk flag, NOT a held-out generalization gap.
    Under the cost convention a good candidate's median ≈ 0, so the ratio is
    taken against the goodness-equivalent ``1 - median`` (keeps it finite — the
    singularity moves to the harmless bad end — and ``GAP_FLAG_THRESHOLD`` valid). A
    single-image trial has no dispersion (``0.0``); below ``min_n`` images the
    estimate is unreliable (``None``, mirroring the stability small-n guard); a
    perfectly flat term is maximally stable (``0.0``).

    Args:
        per_term_scores: Term → clean per-image scores (the evaluator's
            ``per_term`` accumulator). The **first** key is the primary term
            (e.g. ``"Count"`` for the QC objective).
        min_n: The minimum image count for a reliable estimate
            (``Evaluator.min_stability_n``).

    Returns:
        The relative IQR of the primary term's scores, ``0.0`` for a single
        image, or ``None`` when there is no primary term or too few images.
    """
    if not per_term_scores:
        return None
    primary = next(iter(per_term_scores))
    values = per_term_scores[primary]
    n = len(values)
    if n == 1:
        return 0.0
    if n < min_n:
        return None
    median, iqr = _median_iqr(values)
    # Cost convention: a good candidate's median ≈ 0 would blow up the ratio, so
    # divide by the goodness-equivalent (1 - median ≈ 1) — reflection-clean.
    return _relative(iqr, 1.0 - median)


def _is_suspicious(
    score: float,
    terms: Mapping[str, float],
    *,
    score_floor: float,
    count_floor: float,
) -> bool:
    """Flag the qc §5 "great cost on under-detection" gaming signature.

    A candidate is suspicious when a **low** finalized ``score`` (great cost) is
    paired with a **high** aggregated ``Count`` cost — the signature of a pipeline
    that scores well precisely *because* it under-detects (detecting fewer
    colonies dodges the spread/quality penalties). Read from already-computed
    aggregates; a heuristic review flag, not a hard rejection. The intuitive
    floors are mapped into cost-space here (``1 - floor``): a missing ``Count``
    term defaults to ``0.0`` (faithful = best cost) so a non-count objective is
    never flagged.

    Args:
        score: The finalized scalar cost (lower = better).
        terms: The robust-aggregated per-term costs; ``terms["Count"]`` is read.
        score_floor: The "great score" threshold expressed intuitively; the cost
            half fires when ``score <= 1 - score_floor``.
        count_floor: The "under-detection" threshold expressed intuitively; the
            cost half fires when ``terms["Count"] >= 1 - count_floor``.

    Returns:
        ``True`` when ``score <= (1 - score_floor)`` **and**
        ``terms["Count"] >= (1 - count_floor)``.
    """
    count_cost = float(terms.get("Count", 0.0))
    return score <= (1.0 - score_floor) and count_cost >= (1.0 - count_floor)


class EvaluationResult(BaseModel):
    """The outcome of evaluating one candidate over the calibration set.

    Args:
        score: The finalized scalar cost the optimizer **minimizes** (lower =
            better). For a multi-objective candidate this is the **scalar
            projection** of ``objectives`` (``mean(objectives.values())``).
        terms: Robust-aggregated per-term costs (``clamp01(median + λ·IQR)`` each).
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
        gap: The candidate's relative across-plate dispersion of the primary
            term — a cheap instability / overfit-risk flag (the relative IQR of
            the per-image primary-term scores). This is **not** a held-out
            generalization gap (that is Phase 4.5 part 2). ``None`` when the
            signal is unavailable — too few images to estimate dispersion, or a
            failed/pruned candidate.
        suspicious: ``True`` when the candidate matches the qc §5 "great score on
            under-detection" gaming signature (a high ``score`` paired with a
            low ``terms["Count"]``). A heuristic flag for downstream review, not
            a hard rejection.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    terms: dict[str, float]
    n_images: int
    objectives: Optional[dict[str, float]] = None
    failed: bool = False
    pruned: bool = False
    gap: Optional[float] = None
    suspicious: bool = False


class Evaluator(BaseModel):
    """Score a candidate combo over a calibration set (CV-only MVP).

    Args:
        stability_weight: λ in ``clamp01(median + λ·IQR)`` — how hard cross-image
            spread is penalized when aggregating a term (cost) across the
            calibration set.
        failure_score: The worst-cost ceiling assigned when a candidate fails
            to build, measure, or score (lower-is-better objective).
        rung_floor: The minimum first-rung size for the ASHA-style fidelity
            ladder (robust-eval §7) — never prune on fewer plates than this.
        rung_factor: The geometric growth factor between rungs (×3 by default).
        min_rungs: The fewest distinct rungs worth running a ladder for; below
            this the ladder self-disables to a single full-fidelity rung.
        min_stability_n: The minimum image count for a reliable per-trial ``gap``
            (relative across-plate dispersion); below it ``gap`` is ``None``.
        suspicious_score_floor: A trial is flagged ``suspicious`` only when its
            ``score`` is at least this high (the qc §5 "great score" half of the
            under-detection gaming signature).
        suspicious_count_floor: A trial is flagged ``suspicious`` only when its
            primary ``terms["Count"]`` is at most this low (the "on
            under-detection" half of the signature).
    """

    model_config = ConfigDict(frozen=True)

    stability_weight: float = 0.5
    failure_score: float = 1.0
    rung_floor: int = 6
    rung_factor: int = 3
    min_rungs: int = 2
    min_stability_n: int = 4  # TODO: review (unverified vs literature)
    suspicious_score_floor: float = 0.7  # TODO: review (unverified vs literature)
    suspicious_count_floor: float = 0.3  # TODO: review (unverified vs literature)

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
        deferred). After each rung the running ``clamp01(median + λ·IQR)`` is reported to
        ``channel`` and ``channel.should_prune()`` is checked *between* rungs; a
        prune short-circuits to a partial ``EvaluationResult(pruned=True)``. Each
        image is scored **once** (memoized across rungs). Failure taxonomy
        (robust-eval §10): a candidate that won't build is a true ``failed``; one
        image raising mid-scoring contributes the worst-cost term and the loop
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
        gap = _per_trial_dispersion(per_term, min_n=self.min_stability_n)
        suspicious = _is_suspicious(
            final_score,
            aggregated,
            score_floor=self.suspicious_score_floor,
            count_floor=self.suspicious_count_floor,
        )
        return EvaluationResult(
            score=final_score,
            terms=aggregated,
            n_images=len(ordered),
            objectives=final_objectives,
            gap=gap,
            suspicious=suspicious,
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
            # Apply the candidate's processing ops (crop/enhance/detect/refine)
            # before measuring — ``measure`` alone only runs measurement ops on
            # whatever object state already exists, so a raw (undetected) image
            # would yield zero objects. ``inplace=False`` works on a copy so the
            # shared calibration image stays pristine across trials and rungs.
            measurements = candidate.apply_and_measure(
                image, inplace=False, apply_post=False
            )
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

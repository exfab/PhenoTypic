"""The candidate evaluator — the uniform 3-step robust-evaluation loop.

For one parameter combo: build the candidate pipeline, ``score_image`` over the
calibration set, robust-aggregate each term as ``median - λ·IQR`` (the spread
penalty rewards parameters that are stable across images, not just good on
average), then ``finalize`` to the scalar objective the optimizer maximizes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .._scoring._scorer import Scorer
from ._builder import build_pipeline


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
    """

    model_config = ConfigDict(frozen=True)

    score: float
    terms: dict[str, float]
    n_images: int
    failed: bool = False


class Evaluator(BaseModel):
    """Score a candidate combo over a calibration set (CV-only MVP).

    Args:
        stability_weight: λ in ``median - λ·IQR`` — how hard cross-image spread
            is penalized when aggregating a term across the calibration set.
        failure_score: The score assigned when a candidate fails to build,
            measure, or score; the floor of the higher-is-better objective.
    """

    model_config = ConfigDict(frozen=True)

    stability_weight: float = 0.5
    failure_score: float = 0.0

    def evaluate(
        self,
        base: Any,
        scorer: Scorer,
        params: dict[str, Any],
        images: list,
    ) -> EvaluationResult:
        """Build, score, robust-aggregate, and finalize one candidate.

        Args:
            base: The base pipeline embedded in the ``TuningSpec``.
            scorer: The objective.
            params: The sampled combo (``{root-relative-key: value}``).
            images: The calibration images (must be non-empty).

        Returns:
            The candidate's :class:`EvaluationResult`.

        Raises:
            ValueError: If ``images`` is empty.
        """
        if not images:
            raise ValueError(
                "Evaluator.evaluate requires at least one calibration image"
            )

        candidate = build_pipeline(base, params)

        per_term: dict[str, list[float]] = {}
        try:
            for image in images:
                measurements = candidate.measure(image, apply_post=False)
                for term, value in scorer.score_image(image, measurements).items():
                    per_term.setdefault(term, []).append(float(value))
        except Exception:
            # A broken candidate scores worst, never crashing the sweep.
            return EvaluationResult(
                score=self.failure_score,
                terms={},
                n_images=len(images),
                failed=True,
            )

        aggregated = {
            term: _robust_aggregate(values, self.stability_weight)
            for term, values in per_term.items()
        }
        score = float(scorer.finalize(aggregated))
        return EvaluationResult(
            score=score, terms=aggregated, n_images=len(images)
        )

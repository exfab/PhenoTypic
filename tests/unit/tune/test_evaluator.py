from __future__ import annotations

import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune._evaluation._evaluator import (
    EvaluationResult,
    Evaluator,
    _robust_aggregate,
)
from phenotypic.tune._scoring._scorer import Scorer
from phenotypic.tune._strategies._pruning import NoOpChannel


def test_robust_aggregate_penalizes_spread():
    # cost = median + λ·IQR: median 2.5, IQR 1.5 → 2.5 + 0.5*1.5 = 3.25, clamped to 1.0
    assert _robust_aggregate([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(1.0)


def test_robust_aggregate_single_value_is_that_value():
    assert _robust_aggregate([0.2], 0.5) == pytest.approx(0.2)  # IQR 0


def test_robust_aggregate_clamps_to_unit_interval():
    # A high-variance bad term: np.percentile([0.1,0.8,0.9],[75,25]) = [0.85,0.45]
    # → median 0.8, IQR 0.40 → 0.8 + 0.5*0.40 = 1.0 (exactly at the ceiling; a
    # slightly worse term would exceed 1 and be clamped — B1: bᵢ ∈ [0,1] holds).
    assert _robust_aggregate([0.1, 0.8, 0.9], 0.5) == pytest.approx(1.0)


def test_robust_aggregate_in_unit_interval_is_not_clamped():
    # cost stays < 1: np.percentile([0.3,0.4,0.5],[75,25]) = [0.45,0.35], IQR 0.10
    # → median 0.4 + 0.5*0.10 = 0.45 (no clamp).
    assert _robust_aggregate([0.3, 0.4, 0.5], 0.5) == pytest.approx(0.45)


def test_robust_aggregate_above_one_is_clamped():
    # Genuinely > 1 before clamp: np.percentile([0.0,0.9,1.0],[75,25]) = [0.95,0.45],
    # IQR 0.50 → median 0.9 + 0.5*0.50 = 1.15 → clamped to 1.0.
    assert _robust_aggregate([0.0, 0.9, 1.0], 0.5) == pytest.approx(1.0)


class _SequenceScorer(Scorer):
    """Returns preset per-call values (term ``"X"``), ignoring its inputs.

    Emits its values as **cost** directly (the ``LOWER_BETTER`` default), so the
    base ``score_image`` passes them through unchanged.
    """

    values: list[float]
    _cursor: int = PrivateAttr(default=0)

    def _score_terms(self, image, measurements) -> dict[str, float]:
        value = self.values[self._cursor % len(self.values)]
        self._cursor += 1
        return {"X": float(value)}


class _RaisingScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        raise RuntimeError("scoring blew up")


def test_evaluate_runs_3_step_loop_and_aggregates():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    scorer = _SequenceScorer(values=[1.0, 2.0, 3.0])
    result = Evaluator().evaluate(base, scorer, {}, [img, img, img])
    assert isinstance(result, EvaluationResult)
    assert result.n_images == 3
    # term X aggregated: median 2.0, IQR (2.5 - 1.5)=1.0 → 2.0 - 0.5*1.0 = 1.5
    assert result.terms == {"X": pytest.approx(1.5)}
    # default finalize = mean of one term → 1.5
    assert result.score == pytest.approx(1.5)
    assert result.failed is False


def test_evaluate_requires_images():
    with pytest.raises(ValueError):
        Evaluator().evaluate(
            ImagePipeline(ops=[OtsuDetector()]), _SequenceScorer(values=[1.0]), {}, []
        )


def test_evaluate_failure_assigns_failure_score():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator(failure_score=0.0).evaluate(base, _RaisingScorer(), {}, [img])
    assert result.score == 0.0
    assert result.terms == {}
    assert result.n_images == 1
    assert result.failed is True


class _RecordingChannel:
    """A pruning channel that records every ``report`` and never prunes."""

    def __init__(self) -> None:
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return False


def test_evaluate_accepts_channel_keyword():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    scorer = _SequenceScorer(values=[1.0, 2.0, 3.0])
    channel = _RecordingChannel()
    result = Evaluator().evaluate(
        base, scorer, {}, [img, img, img], channel=channel
    )
    assert isinstance(result, EvaluationResult)
    # The channel saw at least one interim report (one per rung).
    assert channel.reports


def test_noop_channel_default_preserves_score():
    # Omitting the channel (NoOp default) must yield exactly the legacy score.
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    scorer = _SequenceScorer(values=[1.0, 2.0, 3.0])
    without = Evaluator().evaluate(base, scorer, {}, [img, img, img])
    scorer2 = _SequenceScorer(values=[1.0, 2.0, 3.0])
    with_noop = Evaluator().evaluate(
        base, scorer2, {}, [img, img, img], channel=NoOpChannel()
    )
    assert with_noop.score == pytest.approx(without.score)
    assert with_noop.pruned is False


def test_result_carries_pruned_flag():
    # The frozen result exposes a pruned flag (default False), distinct from
    # failed: a pruned trial early-stopped but did not error.
    default = EvaluationResult(score=1.0, terms={"X": 1.0}, n_images=3)
    assert default.pruned is False
    pruned = EvaluationResult(score=0.2, terms={"X": 0.2}, n_images=1, pruned=True)
    assert pruned.pruned is True
    assert pruned.failed is False

from __future__ import annotations

import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune._evaluation._evaluator import (
    EvaluationResult,
    Evaluator,
    _is_suspicious,
    _robust_aggregate,
)
from phenotypic.tune.score._scorer import Scorer
from phenotypic.tune.strategy._pruning import NoOpChannel


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
    # term X aggregated: median 2.0, IQR (2.5 - 1.5)=1.0 → 2.0 + 0.5*1.0 = 2.5 → clamped 1.0
    assert result.terms == {"X": pytest.approx(1.0)}
    # default finalize = mean of one term → 1.0
    assert result.score == pytest.approx(1.0)
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


def test_evaluate_failure_assigns_worst_cost():
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    # Default failure_score is now the worst cost (1.0): a candidate that won't
    # score floors to the worst, not the best.
    result = Evaluator().evaluate(base, _RaisingScorer(), {}, [img])
    assert result.score == pytest.approx(1.0)
    assert result.terms == {}
    assert result.n_images == 1
    assert result.failed is True


def test_per_image_exception_pads_worst_cost():
    class _OneGoodOneRaise(Scorer):
        """First call returns cost 0.0, the second raises."""

        _n: int = PrivateAttr(default=0)

        def _score_terms(self, image, measurements) -> dict[str, float]:
            self._n += 1
            if self._n == 1:
                return {"X": 0.0}
            raise RuntimeError("second image blew up")

    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator(stability_weight=0.0).evaluate(
        base, _OneGoodOneRaise(), {}, [img, img]
    )
    # term X = aggregate of [0.0 (good), 1.0 (worst-term pad)] with λ=0
    # → median 0.5 (not 0.0); the failing plate drags the cost UP.
    assert result.terms["X"] == pytest.approx(0.5)
    assert result.failed is False  # not ALL images errored


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


def test_is_suspicious_flags_low_cost_with_high_count_cost():
    # Cost convention: a GREAT finalized cost (0.1 <= 1 - 0.7 = 0.3) paired with a
    # HIGH Count cost (0.8 >= 1 - 0.3 = 0.7, i.e. under-detection) is suspicious.
    assert _is_suspicious(
        0.1, {"Count": 0.8}, score_floor=0.7, count_floor=0.3
    ) is True


def test_is_suspicious_not_flagged_when_count_is_faithful():
    # Low Count cost (faithful detection) -> not suspicious even at a great score.
    assert _is_suspicious(
        0.1, {"Count": 0.2}, score_floor=0.7, count_floor=0.3
    ) is False


def test_is_suspicious_not_flagged_when_cost_is_mediocre():
    # A mediocre cost (0.6 > 0.3) is never flagged, regardless of Count.
    assert _is_suspicious(
        0.6, {"Count": 0.9}, score_floor=0.7, count_floor=0.3
    ) is False


def test_is_suspicious_missing_count_defaults_faithful():
    # A non-count objective: missing Count term defaults to 0.0 (faithful = best
    # cost) so it is NEVER flagged.
    assert _is_suspicious(
        0.0, {}, score_floor=0.7, count_floor=0.3
    ) is False

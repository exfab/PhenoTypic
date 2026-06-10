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
    # median 2.5, IQR (3.25 - 1.75) = 1.5 → 2.5 - 0.5*1.5 = 1.75
    assert _robust_aggregate([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(1.75)


def test_robust_aggregate_single_value_is_that_value():
    assert _robust_aggregate([0.8], 0.5) == pytest.approx(0.8)  # IQR 0


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

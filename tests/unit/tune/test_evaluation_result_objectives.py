"""4.0b — ``EvaluationResult.objectives`` multi-objective sidecar (plan §0a).

The sidecar is NOT a union over ``score``: ``score`` stays a ``float``, and a new
``objectives: dict[str, float] | None = None`` rides alongside. The dict branch only
fires when ``scorer.finalize`` returns a dict — then ``objectives`` holds the named
objectives and ``score`` is their scalar projection (``mean(objectives.values())``).
Single-objective evaluation is byte-identical: ``objectives is None``.
"""
from __future__ import annotations

import pytest
from pydantic import PrivateAttr

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune._evaluation._evaluator import EvaluationResult, Evaluator
from phenotypic.tune._scoring._scorer import Scorer


class _SequenceScorer(Scorer):
    """Single-objective: returns preset per-call cost values (term ``"X"``)."""

    values: list[float]
    _cursor: int = PrivateAttr(default=0)

    def _score_terms(self, image, measurements) -> dict[str, float]:
        value = self.values[self._cursor % len(self.values)]
        self._cursor += 1
        return {"X": float(value)}


class _MultiObjectiveScorer(Scorer):
    """Multi-objective: two named cost objectives passed straight through finalize."""

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Dice": 0.8, "IoU": 0.4}

    def finalize(self, terms):
        return dict(terms)


def test_objectives_defaults_to_none():
    result = EvaluationResult(score=1.0, terms={"X": 1.0}, n_images=3)
    assert result.objectives is None


def test_objectives_dict_round_trips_on_result():
    result = EvaluationResult(
        score=0.6, terms={"Dice": 0.8, "IoU": 0.4}, n_images=2,
        objectives={"Dice": 0.8, "IoU": 0.4},
    )
    assert result.objectives == {"Dice": 0.8, "IoU": 0.4}


def test_objectives_field_is_frozen():
    result = EvaluationResult(score=1.0, terms={"X": 1.0}, n_images=1)
    with pytest.raises(Exception):
        result.objectives = {"a": 1.0}  # type: ignore[misc]


def test_single_objective_evaluate_leaves_objectives_none():
    # The scalar finalize path is unchanged — objectives stays None.
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator().evaluate(
        base, _SequenceScorer(values=[1.0, 2.0, 3.0]), {}, [img, img, img]
    )
    assert result.objectives is None
    # With cost convention: values [1.0, 2.0, 3.0] → robust = clamp01(2.0 + 0.5·1.0) = 1.0.
    assert result.score == pytest.approx(1.0)


def test_multi_objective_evaluate_sets_objectives_and_projects_score():
    # A dict-returning finalize → objectives set + score = mean(values).
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()
    result = Evaluator().evaluate(
        base, _MultiObjectiveScorer(), {}, [img, img]
    )
    assert result.objectives == {
        "Dice": pytest.approx(0.8), "IoU": pytest.approx(0.4)
    }
    # scalar projection = mean of objective values
    assert result.score == pytest.approx(0.6)


def test_multi_objective_empty_objectives_projects_to_zero():
    # A finalize returning an empty dict projects to a 0.0 scalar (no values).
    base = ImagePipeline(ops=[OtsuDetector()])
    img = load_synth_yeast_plate()

    class _EmptyDictScorer(Scorer):
        def _score_terms(self, image, measurements) -> dict[str, float]:
            return {}

        def finalize(self, terms):
            return {}

    result = Evaluator().evaluate(base, _EmptyDictScorer(), {}, [img])
    assert result.objectives == {}
    assert result.score == 0.0

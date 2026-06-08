"""4.0a — ``Scorer.finalize`` return-type widening to ``float | dict[str, float]``.

The multi-objective sidecar (plan §0a) keeps single-objective ``finalize`` returning a
scalar, but a composite scorer may instead return a ``dict[str, float]`` of named
objectives. This locks both shapes: the default base ``finalize`` (and ``QCScorer``)
still returns a ``float``; a scorer whose ``finalize`` returns a dict type-checks and
runs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune._scoring._qc_scorer import QCScorer
from phenotypic.tune._scoring._scorer import Scorer


class _ScalarScorer(Scorer):
    """Default ``finalize`` (mean) — the single-objective scalar shape."""

    def score_image(self, image, measurements) -> dict[str, float]:
        return {"X": 1.0}


class _MultiObjectiveScorer(Scorer):
    """A composite scorer whose ``finalize`` returns named objectives (a dict)."""

    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Dice": 0.8, "IoU": 0.6}

    def finalize(self, terms):
        # Pass the aggregated terms straight through as named objectives.
        return dict(terms)


def test_default_finalize_returns_float():
    # The default mean-reduction path is unchanged — a plain scalar.
    out = _ScalarScorer().finalize({"X": 0.8, "Y": 0.4})
    assert isinstance(out, float)
    assert out == pytest.approx(0.6)


def test_qc_scorer_finalize_returns_float():
    # QCScorer inherits the default scalar finalize.
    layout = pd.DataFrame(
        {"Metadata_ImageName": ["p1"] * 96, "Object_Label": list(range(96))}
    )
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(
            metadata=layout, groupby=["Metadata_ImageName"]
        )
    )
    out = scorer.finalize({"Count": 0.7})
    assert isinstance(out, float)
    assert out == pytest.approx(0.7)


def test_dict_returning_finalize_type_checks_and_runs():
    # The widened return type lets a multi-objective scorer return named objectives.
    out = _MultiObjectiveScorer().finalize({"Dice": 0.8, "IoU": 0.6})
    assert isinstance(out, dict)
    assert out == {"Dice": 0.8, "IoU": 0.6}


def test_dict_finalize_round_trips_through_score_image():
    # The dict-returning scorer remains a valid Scorer end to end.
    scorer = _MultiObjectiveScorer()
    assert scorer.score_image(None, pd.DataFrame()) == {"Dice": 0.8, "IoU": 0.6}

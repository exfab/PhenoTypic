from __future__ import annotations

import pandas as pd
import pytest

from phenotypic.tune.score._scorer import Scorer


class _FixedScorer(Scorer):
    """Concrete test double: returns preset terms, ignores its inputs."""

    terms: dict[str, float]

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return dict(self.terms)


def test_scorer_is_abstract():
    # _score_terms is abstract — the bare base cannot be instantiated.
    with pytest.raises(TypeError):
        Scorer()  # type: ignore[abstract]


def test_concrete_scorer_score_image():
    s = _FixedScorer(terms={"Count": 0.8})
    assert s.score_image(None, pd.DataFrame()) == {"Count": 0.8}


def test_default_finalize_is_mean_of_terms():
    s = _FixedScorer(terms={})
    assert s.finalize({"Count": 0.8}) == pytest.approx(0.8)          # single term passes through
    assert s.finalize({"a": 0.2, "b": 0.8}) == pytest.approx(0.5)    # mean
    assert s.finalize({}) == 0.0                                      # empty → floor


def test_default_availability_true():
    assert _FixedScorer(terms={}).availability() is True

from __future__ import annotations

import pytest

from phenotypic.tune.score._orient import Sense
from phenotypic.tune.score._scorer import Scorer


class _HigherBetterLeaf(Scorer):
    """Emits a bounded [0,1] goodness term; base must complement it to cost."""

    _TERM_SENSE = Sense.HIGHER_BETTER

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"G": 0.8}


class _LowerBetterLeaf(Scorer):
    """Emits a bounded [0,1] cost term directly (the cost-native default)."""

    # _TERM_SENSE defaults to Sense.LOWER_BETTER

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"L": 0.2}


def test_default_term_sense_is_lower_better():
    assert Scorer._TERM_SENSE is Sense.LOWER_BETTER


def test_higher_better_leaf_is_complemented_to_cost():
    # to_cost(0.8, HIGHER_BETTER) == 1 - 0.8 == 0.2
    assert _HigherBetterLeaf().score_image(None, None) == {"G": pytest.approx(0.2)}


def test_lower_better_leaf_passes_through_as_cost():
    # to_cost(0.2, LOWER_BETTER) == 0.2 (identity)
    assert _LowerBetterLeaf().score_image(None, None) == {"L": pytest.approx(0.2)}


def test_term_anchor_defaults_to_none():
    assert _LowerBetterLeaf()._term_anchor("L") is None


def test_score_terms_is_abstract():
    # A subclass that forgets _score_terms cannot be instantiated.
    class _Incomplete(Scorer):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_composite_score_terms_stub_raises():
    from phenotypic.tune.score import CompositeScorer

    with pytest.raises(NotImplementedError):
        CompositeScorer(scorers=[])._score_terms(None, None)

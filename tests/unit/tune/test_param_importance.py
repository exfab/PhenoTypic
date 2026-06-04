from __future__ import annotations

from phenotypic.tune._screening import compute_param_importance
from phenotypic.tune._study_store import StudyStore, Trial


def test_importance_finds_the_driving_param():
    # score depends entirely on `a` (True→1.0, False→0.0); `b` is noise.
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        b = (i // 2) % 3  # irrelevant
        store.append(Trial(
            number=i, params={"a": a, "b": b},
            score=1.0 if a else 0.0, terms={"Count": 1.0 if a else 0.0},
            n_images=2,
        ))
    imp = compute_param_importance(store)
    assert set(imp) == {"a", "b"}
    assert imp["a"] > imp["b"]


def test_importance_empty_below_two_trials():
    store = StudyStore()
    store.append(Trial(number=0, params={"a": 1}, score=0.5, terms={}, n_images=1))
    assert compute_param_importance(store) == {}

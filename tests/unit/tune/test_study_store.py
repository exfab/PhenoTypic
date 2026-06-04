from __future__ import annotations

from phenotypic.tune._study_store import StudyStore, Trial


def _trial(n: int, score: float, *, failed: bool = False, **params) -> Trial:
    return Trial(
        number=n, params=params, score=score,
        terms={"Count": score}, n_images=2, failed=failed,
    )


def test_append_and_len():
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    assert len(store) == 2


def test_best_picks_max_score_ignoring_failures():
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    store.append(_trial(2, 0.99, a=3, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 1 and best.score == 0.9


def test_best_none_when_empty_or_all_failed():
    assert StudyStore().best() is None
    store = StudyStore()
    store.append(_trial(0, 0.0, failed=True))
    assert store.best() is None


def test_parquet_round_trip(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1, mode="x"))
    store.append(_trial(1, 0.9, a=2, mode="y"))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    assert back.best().params == {"a": 2, "mode": "y"}
    assert back.best().terms == {"Count": 0.9}

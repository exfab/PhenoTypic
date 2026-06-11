from __future__ import annotations

from phenotypic.tune._study_store import StudyStore, Trial


def _trial(
    n: int, score: float, *, failed: bool = False, pruned: bool = False, **params
) -> Trial:
    return Trial(
        number=n, params=params, score=score,
        terms={"Count": score}, n_images=2, failed=failed, pruned=pruned,
    )


def test_append_and_len():
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    assert len(store) == 2


def test_best_picks_min_cost_ignoring_failures():
    # Cost convention: lower score is better; best() returns the minimum.
    store = StudyStore()
    store.append(_trial(0, 0.3, a=1))
    store.append(_trial(1, 0.9, a=2))
    store.append(_trial(2, 0.05, a=3, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 0 and best.score == 0.3


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
    # Lower cost wins under minimize → trial 0 (score 0.3).
    assert back.best().params == {"a": 1, "mode": "x"}
    assert back.best().terms == {"Count": 0.3}


def test_parquet_round_trip_empty_store(tmp_path):
    # An empty store must still write a valid (schema'd) parquet and reload to
    # an empty store — run_tuning writes trials.parquet unconditionally.
    path = tmp_path / "trials.parquet"
    StudyStore().to_parquet(path)
    back = StudyStore.from_parquet(path)
    assert len(back) == 0
    assert back.best() is None


def test_parquet_round_trip_preserves_pruned(tmp_path):
    # Trial.pruned must survive the parquet round-trip (resume-symmetric).
    store = StudyStore()
    store.append(_trial(0, 0.5, a=1, pruned=True))
    store.append(_trial(1, 0.9, a=2))  # default pruned=False
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    by_number = {t.number: t for t in back.trials}
    assert by_number[0].pruned is True
    assert by_number[1].pruned is False


def test_parquet_round_trip_all_failed(tmp_path):
    store = StudyStore()
    store.append(_trial(0, 0.0, a=1, failed=True))
    store.append(_trial(1, 0.0, a=2, failed=True))
    path = tmp_path / "trials.parquet"
    store.to_parquet(path)
    back = StudyStore.from_parquet(path)
    assert len(back) == 2
    assert all(t.failed for t in back.trials)
    assert back.best() is None

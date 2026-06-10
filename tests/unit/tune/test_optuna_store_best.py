# tests/unit/tune/test_optuna_store_best.py
"""Phase 2: ``OptunaStudyStore.best()`` returns the lowest-cost trial (minimize)."""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _store(url: str):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(storage_url=url, study_name="tune_cost_v1")


def _trial(n, score, *, failed=False):
    from phenotypic.tune._study_store import Trial

    return Trial(
        number=n, params={"a": n}, score=score,
        terms={"Count": score}, n_images=2, failed=failed,
    )


def test_best_returns_lowest_cost(tmp_path):
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9))
    store.append(_trial(2, 0.05, failed=True))  # failed → excluded
    best = store.best()
    assert best is not None and best.number == 0 and best.score == 0.3


def test_best_none_when_all_failed(tmp_path):
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.1, failed=True))
    assert store.best() is None

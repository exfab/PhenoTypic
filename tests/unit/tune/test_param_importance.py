from __future__ import annotations

import importlib.util

import pytest

from phenotypic.tune._screening import (
    ImportanceReport,
    compute_param_importance,
    compute_param_importance_report,
)
from phenotypic.tune._study_store import StudyStore, Trial

_OPTUNA = importlib.util.find_spec("optuna") is not None


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


# --- G1: fANOVA-vs-RF dispatch (polymorphic, on store capability) -------------


def _rf_store() -> StudyStore:
    store = StudyStore()
    for i in range(24):
        a = i % 2 == 0
        store.append(Trial(
            number=i, params={"a": a, "b": (i // 2) % 3},
            score=1.0 if a else 0.0, terms={"Count": 1.0 if a else 0.0},
            n_images=2,
        ))
    return store


def test_report_falls_back_to_rf_for_journal():
    # A journal store has no native param_importances() → RF-permutation path.
    report = compute_param_importance_report(_rf_store())
    assert isinstance(report, ImportanceReport)
    assert report.method == "rf-permutation"
    assert report.interactions_estimated is False
    assert set(report.importances) == {"a", "b"}
    assert report.importances["a"] > report.importances["b"]


def test_journal_param_importances_capability_is_none():
    # The journal declares it cannot compute native importances.
    assert _rf_store().param_importances() is None


def test_compute_param_importance_unchanged_for_journal():
    # The thin dict-returning wrapper is back-compat: still the RF importances.
    imp = compute_param_importance(_rf_store())
    assert set(imp) == {"a", "b"}
    assert imp["a"] > imp["b"]


def test_dispatch_uses_capability_not_isinstance():
    # A duck-typed store exposing param_importances() drives the fANOVA branch
    # WITHOUT being an OptunaStudyStore instance (capability, not isinstance).
    class _DuckStore:
        def __init__(self, trials):
            self._trials = trials

        @property
        def trials(self):
            return list(self._trials)

        def param_importances(self):
            return {"a": 0.9, "b": 0.1}

    store = _DuckStore(_rf_store().trials)
    report = compute_param_importance_report(store)  # type: ignore[arg-type]
    assert report.method == "fanova"
    assert report.interactions_estimated is True
    assert report.importances == {"a": 0.9, "b": 0.1}


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_fanova_for_optuna_store_with_native_params(tmp_path):
    # An Optuna study driven through native suggest_* carries real trial params,
    # so param_importances() returns the fANOVA importances and the report
    # dispatches to the "fanova" method with interactions_estimated=True.
    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = OptunaStudyStore(storage_url=url, study_name="fanova")

    def objective(trial):
        a = trial.suggest_float("a", 0.0, 1.0)
        trial.suggest_float("b", 0.0, 1.0)  # noise
        return a

    store._study.optimize(objective, n_trials=30)  # populate native params

    native = store.param_importances()
    assert native is not None
    assert native["a"] > native["b"]

    report = compute_param_importance_report(store)
    assert report.method == "fanova"
    assert report.interactions_estimated is True
    assert report.importances["a"] > report.importances["b"]


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_optuna_store_without_native_params_falls_back_to_rf(tmp_path):
    # The engine/CLI append path stores params in user_attrs (empty native
    # params), so fANOVA can't compute → param_importances() returns None and
    # the report falls back to the RF-permutation path.
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = OptunaStudyStore(storage_url=url, study_name="appended")
    for i in range(24):
        a = i % 2 == 0
        store.append(Trial(
            number=i, params={"a": a, "b": (i // 2) % 3},
            score=1.0 if a else 0.0, terms={"Count": 1.0 if a else 0.0},
            n_images=2,
        ))
    assert store.param_importances() is None
    report = compute_param_importance_report(store)
    assert report.method == "rf-permutation"
    assert report.interactions_estimated is False
    assert report.importances["a"] > report.importances["b"]

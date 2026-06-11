"""``OptunaStudyStore`` — the StudyStore Protocol over a live Optuna study (F2).

Hermetic: every test uses a SQLite ``study.db`` under ``tmp_path``. Covers WAL
mode, best/trials/completed_count reads from the study, in-place resumability,
re-opening by name+URL restoring persisted trials, Protocol conformance by
calling, and the ``tune_cache_study_db_path`` helper. ``skipif`` when the extra is absent.
"""
from __future__ import annotations

import importlib.util
import sqlite3

import pytest

from phenotypic.tune._study_store import Trial

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _store(tmp_path, name="s"):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    return OptunaStudyStore(storage_url=url, study_name=name)


def _trial(number, score, *, failed=False, pruned=False):
    return Trial(
        number=number,
        params={"a": number},
        score=score,
        terms={"t": score},
        n_images=2,
        failed=failed,
        pruned=pruned,
    )


def test_sqlite_uses_wal_mode(tmp_path):
    _store(tmp_path)
    db = tmp_path / "study.db"
    con = sqlite3.connect(db)
    try:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        con.close()
    assert mode.lower() == "wal"


def test_is_resumable_in_place_true(tmp_path):
    store = _store(tmp_path)
    assert store.is_resumable_in_place() is True


def test_append_and_trials_round_trip(tmp_path):
    store = _store(tmp_path)
    store.append(_trial(0, 0.5))
    store.append(_trial(1, 0.9))
    trials = store.trials
    assert [t.number for t in trials] == [0, 1]
    assert [t.score for t in trials] == [0.5, 0.9]
    assert trials[0].params == {"a": 0}
    assert trials[1].terms == {"t": 0.9}
    assert trials[0].n_images == 2
    assert len(store) == 2


def test_best_reads_from_study(tmp_path):
    store = _store(tmp_path)
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.95))
    store.append(_trial(2, 0.7))
    best = store.best()
    assert best is not None
    # Cost convention (minimize): the lowest-cost trial wins.
    assert best.score == 0.3
    assert best.number == 0


def test_best_none_when_only_failures(tmp_path):
    store = _store(tmp_path)
    store.append(_trial(0, 0.0, failed=True))
    assert store.best() is None


def test_completed_count_excludes_failures_includes_pruned(tmp_path):
    store = _store(tmp_path)
    store.append(_trial(0, 1.0))
    store.append(_trial(1, 0.0, failed=True))
    store.append(_trial(2, 0.4, pruned=True))
    # completed (1) + pruned (1) = 2; the failure is excluded.
    assert store.completed_count() == 2


def test_pruned_flag_round_trips(tmp_path):
    store = _store(tmp_path)
    store.append(_trial(0, 0.4, pruned=True))
    assert store.trials[0].pruned is True


def test_resume_loads_persisted_trials(tmp_path):
    store = _store(tmp_path, name="resume")
    store.append(_trial(0, 0.6))
    store.append(_trial(1, 0.8))
    # A fresh handle on the same study (name + URL) sees the persisted trials.
    reopened = _store(tmp_path, name="resume")
    assert len(reopened) == 2
    assert [t.score for t in reopened.trials] == [0.6, 0.8]
    # Cost convention (minimize): the lowest-cost trial wins.
    assert reopened.best().score == 0.6


def test_satisfies_study_store_protocol_by_calling(tmp_path):
    from phenotypic.tune._study._protocol import StudyStore

    store = _store(tmp_path)
    assert isinstance(store, StudyStore)
    store.append(_trial(0, 1.0))
    assert isinstance(store.trials, list)
    assert isinstance(len(store), int)
    assert isinstance(store.completed_count(), int)
    assert store.is_resumable_in_place() is True
    assert store.best().number == 0


def test_study_db_path_resolves_to_tune_cache(tmp_path):
    from phenotypic.tools_ import _io_constants as io

    p = io.tune_cache_study_db_path(tmp_path)
    assert p == tmp_path / ".pht-tune-cache" / "study.db"
    assert p.name == io.STUDY_DB


# ---------------------------------------------------------------------------
# 4.6 — Pareto front / knee over the Optuna study's native best_trials
# ---------------------------------------------------------------------------


def _mo_store(tmp_path, name="mo"):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    url = f"sqlite:///{tmp_path / 'study.db'}"
    # Cost convention: every axis minimizes (Optuna's best_trials domination
    # must agree with the store-agnostic minimize-cost Pareto math).
    return OptunaStudyStore(
        storage_url=url, study_name=name, directions=["minimize", "minimize"]
    )


def _mo_trial(number, objectives, *, failed=False):
    score = sum(objectives.values()) / len(objectives)
    return Trial(
        number=number, params={"a": number}, score=score,
        terms={}, n_images=2, objectives=objectives, failed=failed,
    )


def test_single_objective_optuna_store_has_empty_pareto_front(tmp_path):
    # A single-objective study has no multi-objective front (back-compat lock).
    store = _store(tmp_path)
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9))
    assert store.pareto_front() == []
    # Cost convention (minimize): the lowest-cost trial wins.
    assert store.best().score == 0.3


def test_multi_objective_optuna_store_pareto_front_excludes_dominated(tmp_path):
    # Cost coordinates (lower is better): best_trials must drop the HIGH-cost
    # interior point and keep the non-dominated set.
    store = _mo_store(tmp_path)
    store.append(_mo_trial(0, {"Dice": 0.1, "IoU": 0.8}))  # non-dominated
    store.append(_mo_trial(1, {"Dice": 0.5, "IoU": 0.5}))  # non-dominated
    store.append(_mo_trial(2, {"Dice": 0.8, "IoU": 0.1}))  # non-dominated
    store.append(_mo_trial(3, {"Dice": 0.6, "IoU": 0.6}))  # dominated by #1
    front_numbers = {t.number for t in store.pareto_front()}
    assert front_numbers == {0, 1, 2}


def test_multi_objective_objectives_round_trip(tmp_path):
    # The objectives sidecar survives the study round-trip (axis labels intact).
    store = _mo_store(tmp_path)
    store.append(_mo_trial(0, {"Dice": 0.8, "IoU": 0.3}))
    reloaded = {t.number: t for t in store.trials}
    assert reloaded[0].objectives == {"Dice": 0.8, "IoU": 0.3}


def test_multi_objective_knee_point_matches_shared_math(tmp_path):
    # The Optuna front's knee agrees with the store-agnostic knee math.
    from phenotypic.tune._study._pareto import knee_point_of

    store = _mo_store(tmp_path)
    # Cost coordinates: the elbow toward the origin (the low-cost corner) is #1.
    store.append(_mo_trial(0, {"Dice": 0.0, "IoU": 1.0}))
    store.append(_mo_trial(1, {"Dice": 0.1, "IoU": 0.1}))  # elbow (low-cost corner)
    store.append(_mo_trial(2, {"Dice": 1.0, "IoU": 0.0}))
    front = store.pareto_front()
    knee = store.knee_point(front)
    assert knee is not None and knee.number == 1
    assert knee.number == knee_point_of(front).number

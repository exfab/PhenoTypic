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


# --- trials the store must NOT rank ------------------------------------------


def _orphan_running_trial(store):
    """Leave one trial ``RUNNING`` and un-told, as a killed worker does."""
    return store._study.ask()


def test_an_orphaned_running_trial_is_not_the_best(tmp_path):
    """A trial nobody told is not ``failed`` — and must still never win.

    Under the minimize-cost convention ``0.0`` is the BEST possible score, so a
    store that substituted ``0.0`` for an un-told trial's missing value ranked a
    never-evaluated trial above every real one. This is the store-level half of
    that blocker; ``tests/unit/tune/test_distributed_finalize.py`` pins the
    published-artifact half.
    """
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9))
    _orphan_running_trial(store)

    best = store.best()
    assert best is not None
    assert best.number == 0 and best.score == 0.3
    # Anti-vacuity: the orphan really is in the store, and really is unfailed.
    assert len(store.trials) == 3
    assert [t.failed for t in store.trials] == [False, False, False]


def test_terminal_trials_excludes_the_orphan_that_trials_keeps(tmp_path):
    """The two views differ by exactly the in-flight trial — both are needed."""
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9, failed=True))
    _orphan_running_trial(store)

    assert len(store.trials) == 3
    assert [t.number for t in store.terminal_trials()] == [0, 1]


def test_completed_count_measures_what_the_budget_measures(tmp_path):
    """``COMPLETE + PRUNED`` — the unit ``OptunaStrategy.is_exhausted`` uses.

    Counting the raw trial list instead over-reported progress by
    ``#failed + #in-flight``, which let the finalize budget gate open on a fleet
    that still had trials to run.
    """
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    store.append(_trial(1, 0.9, failed=True))
    _orphan_running_trial(store)

    assert len(store.trials) == 3, "raw view still sees everything"
    assert store.completed_count() == 1


def _tell_pruned(store, *, score, stamp_score: bool):
    """Tell a trial PRUNED the way the live strategy does: a state, no value."""
    import optuna

    from phenotypic.tune.strategy._optuna_support import set_trial_user_attrs

    trial = store._study.ask()

    class _Result:
        terms = {"Count": score}
        n_images = 2
        objectives = None
        gap = None
        suspicious = False

    if stamp_score:
        _Result.score = score
    set_trial_user_attrs(trial, params={"a": 9}, result=_Result())
    store._study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    return trial.number


def test_a_pruned_trial_keeps_the_cost_it_actually_scored(tmp_path):
    """``study.tell(trial, state=PRUNED)`` stores no value; the sidecar does.

    Without it a pruned trial's cost read back as "missing" — and the missing
    substitute was ``0.0``, the best score there is. A pruned candidate that
    scored a terrible 0.9 therefore beat a completed one that scored 0.3, on the
    ordinary distributed path, with no dead worker involved.
    """
    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    pruned_number = _tell_pruned(store, score=0.9, stamp_score=True)

    pruned = next(t for t in store.trials if t.number == pruned_number)
    assert pruned.pruned is True
    assert pruned.score == pytest.approx(0.9)

    best = store.best()
    assert best is not None and best.number == 0


def test_a_trial_whose_cost_is_unrecoverable_cannot_win(tmp_path):
    """A pre-sidecar PRUNED row: terminal, unfailed, and genuinely unscored.

    It must be unrankable rather than perfect — the score is *unknown*, and
    guessing the best possible value for an unknown is what published a phantom
    winner in the first place.
    """
    import math

    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    legacy_number = _tell_pruned(store, score=0.9, stamp_score=False)

    legacy = next(t for t in store.trials if t.number == legacy_number)
    assert legacy.pruned is True and math.isinf(legacy.score)

    best = store.best()
    assert best is not None and best.number == 0
    # It still consumed a slot of the budget — pruned work is real work.
    assert store.completed_count() == 2


def test_an_in_flight_trial_with_a_stamped_cost_still_cannot_win(tmp_path):
    """The case that separates the terminal filter from the ``inf`` fallback.

    ``OptunaStrategy.register_result`` stamps the score sidecar and *then* calls
    ``study.tell``. A worker killed between those two DB round-trips leaves a
    ``RUNNING`` trial carrying a perfectly readable cost — so the ``inf``
    substitution never fires and cannot hide it. Only excluding non-terminal
    trials keeps it out of the ranking, and it must be excluded: the ``tell``
    never landed and the trial consumed none of the budget, so the study and the
    fleet would otherwise disagree about what happened.

    Do **not** weaken this on the theory that something will clean the trial up.
    Nothing will, on the backend that matters: measured against optuna 4.9.0,
    the ``journal://`` storage a ``--slurm`` fleet now defaults to exposes
    neither ``record_heartbeat`` nor ``get_heartbeat_interval``, and
    ``optuna.storages.fail_stale_trials`` silently **no-ops** against it —
    returns cleanly, changes nothing, warns about nothing. The zombie survives a
    reopen and is permanent for the life of the study. (Under an RDB the
    ``_fail_stale_trials`` call in ``OptunaStrategy.suggest`` does reclaim it
    once the grace period expires, which is why this went unnoticed.)
    """
    from phenotypic.tune.strategy._optuna_support import set_trial_user_attrs

    store = _store(f"sqlite:///{tmp_path / 'study.db'}")
    store.append(_trial(0, 0.3))
    in_flight = store._study.ask()

    class _Result:
        score = 0.0  # the best cost there is under the minimize convention
        terms = {"Count": 0.0}
        n_images = 2
        objectives = None
        gap = None
        suspicious = False

    set_trial_user_attrs(in_flight, params={"a": 9}, result=_Result())
    # ... and no `tell`.

    # Anti-vacuity: the cost really did survive the round trip, so this test
    # would fail against a store that merely refused to read it.
    orphan = next(t for t in store.trials if t.number == in_flight.number)
    assert orphan.score == pytest.approx(0.0)
    assert orphan.failed is False

    best = store.best()
    assert best is not None and best.number == 0
    assert store.completed_count() == 1

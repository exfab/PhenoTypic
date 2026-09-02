"""One shared study handle (Change 4) + bounded transient-DB retry (Change 5).

Change 4: ``OptunaConfig.build`` must hand the strategy the store's ONE study
object (no second ``create_study``), and the strategy must re-attach ITS sampler
+ pruner to that shared handle (the sampler/pruner live on the in-memory study,
not in storage, so reusing the store's default-TPE study without re-attaching
would silently drop the chosen sampler and ASHA pruning).

Change 5: ``retry_on_transient_db_error`` retries a transient
``sqlalchemy.exc.OperationalError`` with bounded backoff, but propagates a
non-transient error immediately.
"""
from __future__ import annotations

import errno

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

_STUDY = "tune_cost_v1"


def _store(url: str):
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    return OptunaStudyStore(storage_url=url, study_name=_STUDY)


def _space():
    from phenotypic.tune import Categorical, Knob, SearchSpace

    return SearchSpace(
        knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),)
    )


# --- Change 4: one shared study handle ----------------------------------------


def test_strategy_reuses_store_study_object(tmp_path):
    from phenotypic.tune.strategy._config import OptunaConfig

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = _store(url)
    strat = OptunaConfig(sampler="tpe", n_trials=3).build(_space(), store)
    # The SAME Python object — not a second create_study handle.
    assert strat._study is store.study


def test_strategy_reattaches_its_chosen_sampler_to_shared_study(tmp_path):
    import optuna

    from phenotypic.tune.strategy._config import OptunaConfig

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = _store(url)
    # The store created the study with Optuna's DEFAULT (TPE) sampler.
    assert isinstance(store.study.sampler, optuna.samplers.TPESampler)

    # Building a CMA-ES strategy must re-attach CMA-ES to the shared handle, not
    # leave the store's default TPE in place.
    strat = OptunaConfig(sampler="cmaes", n_trials=3).build(_space(), store)
    assert isinstance(strat._study.sampler, optuna.samplers.CmaEsSampler)
    # And the ASHA pruner the strategy derives is attached too.
    assert isinstance(
        strat._study.pruner, optuna.pruners.SuccessiveHalvingPruner
    )
    # It is still the store's one object.
    assert strat._study is store.study


def test_strategy_without_store_study_opens_its_own(tmp_path):
    # A non-Optuna store (no ``.study``) → the strategy falls back to its own
    # create_study from the URL + name (the screening-rounds journal path).
    from phenotypic.tune.strategy._optuna import OptunaStrategy

    url = f"sqlite:///{tmp_path / 'study.db'}"

    class _JournalLike:
        study_name = None
        storage_url = None
        # no ``study`` attribute

    strat = OptunaStrategy(
        _space(),
        sampler="tpe",
        n_trials=2,
        storage_url=url,
        study_name=_STUDY,
        store=_JournalLike(),
    )
    # It still built a usable study (just not the store's — there was none).
    assert strat._study is not None


# --- Change 5: bounded transient-DB retry -------------------------------------


def test_retry_succeeds_after_transient_operational_errors():
    from sqlalchemy.exc import OperationalError

    from phenotypic.tune.strategy._optuna_support import (
        retry_on_transient_db_error,
    )

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("SELECT 1", {}, Exception("database is locked"))
        return "ok"

    # 3 attempts; fails twice then succeeds on the third.
    result = retry_on_transient_db_error(_flaky, trial_number=7, attempts=3)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_and_reraises_transient_error(monkeypatch):
    from sqlalchemy.exc import OperationalError

    import phenotypic.tune.strategy._optuna_support as support
    from phenotypic.tune.strategy._optuna_support import (
        retry_on_transient_db_error,
    )

    # Don't actually sleep through the backoff.
    monkeypatch.setattr(support.time, "sleep", lambda *_: None)

    def _always_locked():
        raise OperationalError("X", {}, Exception("database is locked"))

    with pytest.raises(OperationalError):
        retry_on_transient_db_error(_always_locked, attempts=3)


def test_retry_succeeds_after_transient_os_error(monkeypatch):
    import phenotypic.tune.strategy._optuna_support as support
    from phenotypic.tune.strategy._optuna_support import (
        retry_on_transient_db_error,
    )

    monkeypatch.setattr(support.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def _temporarily_unavailable():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EAGAIN, "try again")
        return "ok"

    assert retry_on_transient_db_error(_temporarily_unavailable, attempts=2) == "ok"
    assert calls["n"] == 2


def test_retry_does_not_retry_non_transient_error():
    from phenotypic.tune.strategy._optuna_support import (
        retry_on_transient_db_error,
    )

    calls = {"n": 0}

    def _bug():
        calls["n"] += 1
        raise ValueError("a real bug, not a transient DB error")

    with pytest.raises(ValueError, match="real bug"):
        retry_on_transient_db_error(_bug, attempts=3)
    # Called exactly once — a non-transient error is never retried.
    assert calls["n"] == 1

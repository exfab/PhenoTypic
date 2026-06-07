"""Stale RUNNING-trial reconciliation keeps the shared budget honest (Change 3).

A worker killed mid-trial leaves its trial stuck in ``RUNNING``. Before a fresh
worker enters the ask/tell loop, ``fail_stale_running_trials`` transitions every
such zombie to ``FAIL`` so it (a) no longer lingers in the study and (b) cannot
inflate the COMPLETE+PRUNED budget count the engine/strategy gate on. These are
hermetic SQLite tests — the reconciliation is storage-agnostic, so no Postgres
is required.
"""
from __future__ import annotations

import importlib.util

import pytest

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _make_study(url: str, name: str = "tune"):
    import optuna

    return optuna.create_study(
        storage=url, study_name=name, direction="maximize", load_if_exists=True
    )


def test_fail_stale_running_trials_marks_running_as_fail(tmp_path):
    import optuna

    from phenotypic.tune._strategies._optuna_support import (
        fail_stale_running_trials,
    )

    url = f"sqlite:///{tmp_path / 'study.db'}"
    study = _make_study(url)
    # Simulate a killed worker: ask leaves a RUNNING trial that is never told.
    trial = study.ask()
    trial.suggest_float("x", 0.0, 1.0)
    running = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    assert len(running) == 1

    # A fresh worker reconciles from its own handle before the loop.
    fresh = _make_study(url)
    n_failed = fail_stale_running_trials(fresh)
    assert n_failed == 1

    running_after = fresh.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    failed_after = fresh.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.FAIL,)
    )
    assert running_after == []
    assert len(failed_after) == 1


def test_reconciliation_is_a_noop_without_running_trials(tmp_path):
    from phenotypic.tune._strategies._optuna_support import (
        fail_stale_running_trials,
    )

    url = f"sqlite:///{tmp_path / 'study.db'}"
    study = _make_study(url)
    study.add_trial(
        __import__("optuna").trial.create_trial(value=1.0)
    )  # a COMPLETE trial
    assert fail_stale_running_trials(study) == 0


def test_budget_no_longer_overshoots_after_reconciliation(tmp_path):
    """A failed zombie does not consume budget; the post-reconcile run is bounded.

    Leaves a RUNNING zombie, reconciles it to FAIL, then runs the engine to its
    full ``n_trials``. The failed trial must NOT count toward the budget (the
    strategy counts only COMPLETE+PRUNED), so the study ends with exactly the
    failed zombie plus ``n_trials`` real completed trials — not fewer (the zombie
    must not have eaten a slot) and not unbounded.
    """
    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.tune import (
        Categorical,
        Evaluator,
        Knob,
        OptunaConfig,
        Scorer,
        SearchSpace,
    )
    from phenotypic.tune._engine import TuningEngine
    from phenotypic.tune._spec import Budget, TuningSpec
    from phenotypic.tune._strategies._optuna_support import (
        fail_stale_running_trials,
    )
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    import optuna

    class _ConstScorer(Scorer):
        def score_image(self, image, measurements) -> dict[str, float]:
            return {"Count": 1.0}

    url = f"sqlite:///{tmp_path / 'study.db'}"
    n_trials = 4

    # A killed worker leaves a RUNNING zombie in the shared study.
    zombie_study = _make_study(url)
    z = zombie_study.ask()
    z.suggest_float("x", 0.0, 1.0)

    space = SearchSpace(
        knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),)
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=space,
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="tpe", n_trials=n_trials, storage_url=url),
        budget=Budget(),
    )
    store = OptunaStudyStore(storage_url=url, study_name="tune")

    # Reconcile BEFORE the loop (what run_worker does), then run to budget.
    fail_stale_running_trials(store.study)
    TuningEngine(spec, store=store).optimize([load_synth_yeast_plate()])

    final = _make_study(url)
    completed = final.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    failed = final.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.FAIL,)
    )
    running = final.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    # Exactly n_trials real completions (the zombie ate no slot), the zombie is
    # FAILed, and nothing is left RUNNING.
    assert len(completed) == n_trials
    assert len(failed) == 1
    assert running == []


def test_worker_startup_does_not_fail_live_running_trials(tmp_path, monkeypatch):
    """A second worker must not mark a peer's active trial failed at startup."""
    import optuna

    from phenotypic.tune._tune_cli import _worker

    url = f"sqlite:///{tmp_path / 'study.db'}"
    study = _make_study(url)
    live = study.ask()
    live.suggest_float("x", 0.0, 1.0)

    from phenotypic.tune import OptunaConfig

    from tests.unit.tune.test_run_tuning_slurm import _grid_input_spec

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        _grid_input_spec()
        .model_copy(update={"strategy": OptunaConfig(n_trials=1, storage_url=url)})
        .model_dump_json()
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        '{"calibration":["cal"],"held_out":[],"kind":"none",'
        '"group_key":null,"dataset_identity":"x",'
        '"within_group_caveat":false,"seed_entropy":[]}'
    )

    class _FakeImage:
        name = "cal"

    class _FakeStore:
        def __init__(self):
            self.study = _make_study(url)

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            pass

    monkeypatch.setattr(_worker, "_load_images", lambda _path: [_FakeImage()])
    monkeypatch.setattr(_worker, "build_worker_store", lambda **_kw: _FakeStore())
    monkeypatch.setattr("phenotypic.tune._engine.TuningEngine", _FakeEngine)

    _worker.run_worker(
        spec_path=spec_path,
        images_dir=tmp_path,
        storage_url=url,
        study_name="tune",
        split_path=split_path,
    )

    fresh = _make_study(url)
    assert fresh.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )[0].number == live.number
    assert fresh.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.FAIL,)
    ) == []


def test_optuna_store_configures_rdb_heartbeat(monkeypatch):
    """Persistent stores should use Optuna's heartbeat stale-trial mechanism."""
    import phenotypic.tune._study._optuna_store as store_mod

    captured = {}

    class _FakeStorage:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeOptuna:
        class storages:
            RDBStorage = _FakeStorage

        @staticmethod
        def create_study(**kwargs):
            captured["create_storage"] = kwargs["storage"]
            return object()

    monkeypatch.setattr(store_mod.OptunaStudyStore, "_enable_sqlite_wal", lambda *a: None)
    monkeypatch.setitem(__import__("sys").modules, "optuna", _FakeOptuna)

    store_mod.OptunaStudyStore(storage_url="postgresql://host/db", study_name="tune")

    assert captured["url"] == "postgresql://host/db"
    assert captured["heartbeat_interval"] == 60
    assert captured["grace_period"] == 180
    assert isinstance(captured["create_storage"], _FakeStorage)

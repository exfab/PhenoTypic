from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import Scorer
from phenotypic.tune.strategy import (
    GridConfig,
    RandomConfig,
)
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._spec import Budget, TuningSpec


class _ConstScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


class _FailScorer(Scorer):
    """Always raises — the Evaluator catches it and marks the trial failed."""

    def _score_terms(self, image, measurements) -> dict[str, float]:
        raise RuntimeError("forced failure")


def _grid_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.GaussianBlur.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.GaussianBlur.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(budget: Budget, store_pipeline) -> TuningSpec:
    return TuningSpec(
        pipeline=store_pipeline,
        search_space=_grid_space(),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=budget,
    )


def _base():
    from phenotypic.enhance import GaussianBlur
    return ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])


class _SuspiciousScorer(Scorer):
    """Low finalized cost (~0.267) paired with a high Count cost (0.8).

    Trips the suspicious gate (``cost <= 0.3 and Count_cost >= 0.7``) and yields
    a computable per-trial dispersion, so a run exercises both robust-eval signals
    end-to-end (Evaluator → Trial → store).
    """

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 0.8, "A": 0.0, "B": 0.0}


def test_engine_carries_gap_and_suspicious_to_store():
    # Regression (4.5p1): the engine's Trial construction must carry result.gap
    # AND result.suspicious — not only objectives/failed/pruned. Dropping them
    # left every real run's journal gap=None/suspicious=False and silently killed
    # the data-poor generalization fallback's calibration_stability (which reads
    # the winner's gap). min_stability_n=4, so 4 images make the per-trial
    # dispersion computable (a flat term → 0.0, which is *not* None — that's what
    # distinguishes the dropped-field bug from the fix).
    images = [load_synth_yeast_plate() for _ in range(4)]
    spec = TuningSpec(
        pipeline=_base(),
        search_space=_grid_space(),
        scorer=_SuspiciousScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(n_trials=1),
    )
    engine = TuningEngine(spec)
    engine.optimize(images)
    best = engine.store.best()
    assert best is not None
    assert best.gap is not None        # the dropped-field bug left this None
    assert best.suspicious is True     # low cost + high Count cost (under-detection signature)


def test_engine_runs_full_grid():
    spec = _spec(Budget(), _base())
    engine = TuningEngine(spec)
    best = engine.optimize([load_synth_yeast_plate()])
    assert len(engine.store) == 6           # the conditional Cartesian product
    assert best is not None
    # all six param-combos are distinct
    seen = {tuple(sorted(t.params.items())) for t in engine.store.trials}
    assert len(seen) == 6


class _AvailScorer(Scorer):
    """A const scorer whose availability is the ``ok`` flag (cost-native)."""

    ok: bool = True

    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 0.2}

    def availability(self) -> bool:
        return self.ok


def test_engine_pins_composite_active_set_to_available_children():
    # Phase 3 §6.3: the engine pins the augmented Tchebycheff composite's
    # study-global active set to the children available study-wide, once before
    # the trial loop. An unavailable child is dropped from the roster.
    from phenotypic.tune.score._composite import CompositeScorer

    comp = CompositeScorer(
        scorers=[_AvailScorer(ok=True), _AvailScorer(ok=False)],
    )
    spec = TuningSpec(
        pipeline=_base(),
        search_space=_grid_space(),
        scorer=comp,
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(n_trials=1),
    )
    engine = TuningEngine(spec)
    engine.optimize([load_synth_yeast_plate()])
    # Only the available child (s0) is pinned as an objective axis.
    assert comp._active_handles == ("s0",)


def test_engine_budget_caps_trials():
    spec = _spec(Budget(n_trials=3), _base())
    engine = TuningEngine(spec)
    engine.optimize([load_synth_yeast_plate()])
    assert len(engine.store) == 3


def test_engine_resumes_from_store():
    img = [load_synth_yeast_plate()]
    # first run: 3 of 6
    e1 = TuningEngine(_spec(Budget(n_trials=3), _base()))
    e1.optimize(img)
    # resume: same store, no cap → completes to 6 with no duplicates
    e2 = TuningEngine(_spec(Budget(), _base()), store=e1.store)
    e2.optimize(img)
    assert len(e2.store) == 6
    seen = {tuple(sorted(t.params.items())) for t in e2.store.trials}
    assert len(seen) == 6


def _random_spec(budget: Budget, n_trials: int, seed: int) -> TuningSpec:
    return TuningSpec(
        pipeline=_base(),
        search_space=_grid_space(),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=RandomConfig(n_trials=n_trials, seed=seed),
        budget=budget,
    )


def test_engine_resumes_random_config_continues_sequence():
    # Locks the re-seed/fast-forward contract: a RandomConfig run resumed from a
    # partial store must CONTINUE the same seeded draw sequence, not restart it.
    img = [load_synth_yeast_plate()]
    full = TuningEngine(_random_spec(Budget(), n_trials=5, seed=7))
    full.optimize(img)
    reference = [t.params for t in full.store.trials]

    partial = TuningEngine(_random_spec(Budget(n_trials=2), n_trials=5, seed=7))
    partial.optimize(img)
    assert len(partial.store) == 2
    resumed = TuningEngine(
        _random_spec(Budget(), n_trials=5, seed=7), store=partial.store
    )
    resumed.optimize(img)
    assert [t.params for t in resumed.store.trials] == reference


class _SpyChannel:
    """A no-op channel tagged so the engine-threading test can identify it."""

    def __init__(self) -> None:
        self.reported = False

    def report(self, value: float, step: int) -> None:
        self.reported = True

    def should_prune(self) -> bool:
        return False


def test_engine_passes_channel_from_suggest_to_evaluate(monkeypatch):
    # The engine must hand the channel returned by suggest() to evaluate(),
    # not discard it (the channel reports during evaluation).
    from phenotypic.tune.strategy._config import GridConfig as _GridConfig

    spy = _SpyChannel()

    class _OneShotGrid:
        def __init__(self) -> None:
            self._done = False

        def suggest(self):
            self._done = True
            return {"1.ignore_zeros": True}, spy

        def register_result(self, params, result, *, pruned=False) -> None:
            return None

        def is_exhausted(self) -> bool:
            return self._done

    monkeypatch.setattr(
        _GridConfig, "build",
        lambda self, space, store, *, directions=None: _OneShotGrid(),
    )
    engine = TuningEngine(_spec(Budget(n_trials=1), _base()))
    engine.optimize([load_synth_yeast_plate()])
    assert spy.reported, "the channel from suggest() must reach evaluate()"


def test_engine_registers_pruned_flag(monkeypatch):
    # A pruned EvaluationResult must persist Trial.pruned=True and flow
    # pruned=True into register_result; pruned counts toward budget, not failures.
    from phenotypic.tune._evaluation._evaluator import EvaluationResult
    from phenotypic.tune.strategy._config import GridConfig as _GridConfig

    seen_pruned: list[bool] = []

    class _PruneOnceGrid:
        def __init__(self) -> None:
            self._n = 0

        def suggest(self):
            self._n += 1
            return {"1.ignore_zeros": True}, _SpyChannel()

        def register_result(self, params, result, *, pruned=False) -> None:
            seen_pruned.append(pruned)

        def is_exhausted(self) -> bool:
            return self._n >= 1

    def _fake_eval(self, base, scorer, params, images, *, channel=None):
        return EvaluationResult(score=0.1, terms={"X": 0.1}, n_images=1, pruned=True)

    monkeypatch.setattr(
        _GridConfig, "build",
        lambda self, space, store, *, directions=None: _PruneOnceGrid(),
    )
    monkeypatch.setattr(Evaluator, "evaluate", _fake_eval)
    engine = TuningEngine(_spec(Budget(n_trials=1), _base()))
    engine.optimize([load_synth_yeast_plate()])
    assert seen_pruned == [True]
    trial = engine.store.trials[0]
    assert trial.pruned is True
    assert trial.failed is False


def test_max_failures_counts_recorded_failures_on_resume():
    # The failure safety-valve must survive resume: failures already journaled
    # count toward max_failures (else a resumed run resets the cap to 0).
    img = [load_synth_yeast_plate()]
    spec = lambda: TuningSpec(  # noqa: E731
        pipeline=_base(),
        search_space=_grid_space(),
        scorer=_FailScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(max_failures=2),
    )
    e1 = TuningEngine(spec())
    e1.optimize(img)
    assert len(e1.store) == 2 and all(t.failed for t in e1.store.trials)
    # resume: cap (2) already reached by recorded failures → no new trials run
    e2 = TuningEngine(spec(), store=e1.store)
    e2.optimize(img)
    assert len(e2.store) == 2

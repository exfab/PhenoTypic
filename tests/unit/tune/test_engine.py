from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    RandomConfig,
    Scorer,
    SearchSpace,
)
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._spec import Budget, TuningSpec


class _ConstScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


class _FailScorer(Scorer):
    """Always raises — the Evaluator catches it and marks the trial failed."""

    def score_image(self, image, measurements) -> dict[str, float]:
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


def test_engine_runs_full_grid():
    spec = _spec(Budget(), _base())
    engine = TuningEngine(spec)
    best = engine.optimize([load_synth_yeast_plate()])
    assert len(engine.store) == 6           # the conditional Cartesian product
    assert best is not None
    # all six param-combos are distinct
    seen = {tuple(sorted(t.params.items())) for t in engine.store.trials}
    assert len(seen) == 6


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

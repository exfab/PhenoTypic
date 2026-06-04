from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    Scorer,
    SearchSpace,
)
from phenotypic.tune._engine import TuningEngine
from phenotypic.tune._spec import Budget, TuningSpec


class _ConstScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


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

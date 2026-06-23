"""The lazy-import lock — the phase's standing invariant.

``import phenotypic`` (and any Grid/Random tuning path) must NEVER import
``optuna``. Optuna ships only behind the ``tune`` extra and is imported lazily
inside ``_optuna_support._require_optuna`` (optuna-integration.md §10). These
tests snapshot ``sys.modules`` and assert ``"optuna"`` is absent — re-run them
in every later gate.
"""
from __future__ import annotations

import importlib
import sys


def _optuna_absent() -> bool:
    """Whether ``optuna`` is currently unimported."""
    return "optuna" not in sys.modules


def test_import_phenotypic_does_not_import_optuna():
    # A clean import of the umbrella package must not drag optuna in.
    sys.modules.pop("optuna", None)
    importlib.import_module("phenotypic")
    importlib.import_module("phenotypic.tune")
    assert _optuna_absent(), "importing phenotypic must not import optuna"


def test_grid_random_paths_do_not_import_optuna():
    # Running a tiny grid optimize() exercises suggest→evaluate→register without
    # ever touching the Optuna-backed strategy, so optuna stays unimported.
    sys.modules.pop("optuna", None)

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
    TuningEngine,
    TuningSpec,
)
    from phenotypic.tune.score import Scorer
    from phenotypic.tune.strategy import GridConfig

    class _ConstScorer(Scorer):
        def _score_terms(self, image, measurements) -> dict[str, float]:
            return {"Count": 1.0}

    space = SearchSpace(knobs=(
        Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=space,
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    TuningEngine(spec).optimize([load_synth_yeast_plate()])
    assert _optuna_absent(), "the Grid/Random path must not import optuna"

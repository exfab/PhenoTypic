"""4.7 — ``run_tuning`` writes ``deliverables/pareto/`` only when multi-objective.

The cross-cutting back-compat lock (plan §0b): a **single-objective** spec writes
exactly the Phase-1 ``deliverables/`` set and **no** ``pareto/`` directory. A
**multi-objective** spec (a ``CompositeScorer(multi_objective=True)`` whose
``finalize`` returns a dict, surfaced as ``EvaluationResult.objectives``) writes
the Pareto front parquet, one ``best_<objective>.json`` pipeline per objective
axis, and the **knee** as the top-level ``best_pipeline.json``. Multi-objective
requires an Optuna NSGA-II strategy (grid/random are rejected at validation,
4.8), so the multi-objective body is ``skipif`` when the ``tune`` extra is absent.
"""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Budget,
    Categorical,
    CompositeScorer,
    Evaluator,
    GridConfig,
    Knob,
    OptunaConfig,
    QCScorer,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._tune_cli._run import run_tuning
from phenotypic.tune._study_store import Trial

_OPTUNA = importlib.util.find_spec("optuna") is not None


def _layout_csv(tmp_path) -> str:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return str(csv)


def _qc(tmp_path) -> QCScorer:
    return QCScorer(check=ExpectedVsDetectedCount(
        metadata=_layout_csv(tmp_path), groupby=["Metadata_ImageName"]))


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _single_objective_spec(tmp_path) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=_qc(tmp_path),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def _multi_objective_spec(tmp_path) -> TuningSpec:
    composite = CompositeScorer(
        scorers=[_qc(tmp_path), _qc(tmp_path)], multi_objective=True
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=composite,
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=4),
        budget=Budget(n_trials=4),
    )


# ---------------------------------------------------------------------------
# Single-objective: the back-compat lock — NO pareto/ dir
# ---------------------------------------------------------------------------


def test_single_objective_writes_no_pareto_dir(tmp_path):
    out = tmp_path / "run"
    run_tuning(_single_objective_spec(tmp_path), [load_synth_yeast_plate()], out)

    # Phase-1 deliverables/ set, unchanged.
    assert io.best_pipeline_path(out).exists()
    assert io.tuning_spec_path(out).exists()
    assert io.param_importance_path(out).exists()
    assert io.trials_parquet_path(out).exists()
    # ... and crucially, NO pareto/ directory.
    assert not io.pareto_dir(out).exists()


# ---------------------------------------------------------------------------
# Multi-objective: pareto/ front + per-objective pipelines + knee best_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_multi_objective_writes_pareto_front_and_knee(tmp_path):
    out = tmp_path / "run_mo"
    run_tuning(_multi_objective_spec(tmp_path), [load_synth_yeast_plate()], out)

    # The pareto/ dir exists with a non-empty front parquet.
    assert io.pareto_dir(out).exists()
    front_path = io.pareto_front_parquet_path(out)
    assert front_path.exists()
    front = pd.read_parquet(front_path)
    assert len(front) >= 1
    # objectives_json is populated for the front (it is multi-objective).
    assert front["objectives_json"].notna().any()

    # A per-objective best pipeline lands for each composite axis (s0, s1).
    for objective in ("s0", "s1"):
        per_axis = io.pareto_best_pipeline_path(out, objective)
        assert per_axis.exists(), f"missing best_{objective}.json"
        ImagePipeline.from_json(per_axis.read_text())  # reloads as a pipeline

    # The knee is the top-level best_pipeline.json (reloads runnable).
    assert io.best_pipeline_path(out).exists()
    ImagePipeline.from_json(io.best_pipeline_path(out).read_text())


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_multi_objective_per_objective_param_importance(tmp_path):
    # param_importance.json still lands (run-level), alongside the pareto/ set.
    out = tmp_path / "run_mo2"
    run_tuning(_multi_objective_spec(tmp_path), [load_synth_yeast_plate()], out)
    assert io.param_importance_path(out).exists()
    assert io.pareto_dir(out).exists()


def test_per_axis_pareto_export_picks_lowest_cost_trial(tmp_path):
    """The per-objective ``best_<axis>.json`` exports the LOWER-cost trial.

    Cost convention: for each axis, the front trial with the *lowest* cost on
    that axis is the per-axis winner (under the old maximize logic it would have
    been the worst). Two non-dominated trials with distinct params per axis:
    A (sigma=1.0) wins s0 (0.1 < 0.9); B (sigma=2.0) wins s1 (0.1 < 0.9).
    """
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli._run import _finalize_pareto_outputs

    spec = _single_objective_spec(tmp_path)
    store = JournalStudyStore([
        Trial(number=0, params={"0.sigma": 1.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": 0.1, "s1": 0.9}),  # A: best s0
        Trial(number=1, params={"0.sigma": 2.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": 0.9, "s1": 0.1}),  # B: best s1
    ])
    out = tmp_path / "per_axis"
    _finalize_pareto_outputs(store, spec, out)

    best_s0 = ImagePipeline.from_json(
        io.pareto_best_pipeline_path(out, "s0").read_text()
    )
    best_s1 = ImagePipeline.from_json(
        io.pareto_best_pipeline_path(out, "s1").read_text()
    )
    # The first op is GaussianBlur; its sigma reflects the winning trial's param.
    assert list(best_s0.get_ops().values())[0].sigma == 1.0  # A wins s0 (lowest cost)
    assert list(best_s1.get_ops().values())[0].sigma == 2.0  # B wins s1 (lowest cost)


def test_headline_winner_prefers_pareto_knee_over_scalar_best():
    from phenotypic.tune._tune_cli._run import _headline_winner

    scalar_best = Trial(
        number=0,
        params={"0.sigma": 1.0},
        score=0.95,
        terms={},
        n_images=1,
        objectives={"s0": 0.95, "s1": 0.10},
    )
    knee = Trial(
        number=1,
        params={"0.sigma": 2.0},
        score=0.80,
        terms={},
        n_images=1,
        objectives={"s0": 0.80, "s1": 0.80},
    )

    class _Store:
        def best(self):
            return scalar_best

        def pareto_front(self):
            return [scalar_best, knee]

        def knee_point(self, front):
            assert front == [scalar_best, knee]
            return knee

    assert _headline_winner(_Store()) is knee

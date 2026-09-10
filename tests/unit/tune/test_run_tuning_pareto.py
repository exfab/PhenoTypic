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
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune.score import (
    CompositeScorer,
    QCScorer,
)
from phenotypic.tune.strategy import (
    GridConfig,
    OptunaConfig,
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
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
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
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=composite,
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=4),
        budget=Budget(n_trials=4),
    )


class _UnsafeAxisCompositeScorer(CompositeScorer):
    """Supported scorer extension whose axis cannot become a filename."""

    def objective_names(self) -> list[str]:
        return ["safe", "../escape"]


def _unsafe_axis_spec(tmp_path) -> TuningSpec:
    scorer = _UnsafeAxisCompositeScorer(
        scorers=[_qc(tmp_path), _qc(tmp_path)], multi_objective=True
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=_space(), scorer=scorer, evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=1),
        budget=Budget(n_trials=1),
    )


class _DuplicateAxisCompositeScorer(CompositeScorer):
    """Supported scorer extension whose declared vector shape is invalid."""

    def objective_names(self) -> list[str]:
        return ["s0", "s0"]


def _duplicate_axis_spec(tmp_path) -> TuningSpec:
    scorer = _DuplicateAxisCompositeScorer(
        scorers=[_qc(tmp_path), _qc(tmp_path)], multi_objective=True
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=1),
        budget=Budget(n_trials=1),
    )


class _CasefoldAxisCompositeScorer(CompositeScorer):
    """Supported scorer extension whose artifacts collide on Windows."""

    def objective_names(self) -> list[str]:
        return ["Dice", "dice"]


def _casefold_axis_spec(tmp_path) -> TuningSpec:
    scorer = _CasefoldAxisCompositeScorer(
        scorers=[_qc(tmp_path), _qc(tmp_path)], multi_objective=True
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=_space(), scorer=scorer, evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=1),
        budget=Budget(n_trials=1),
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
    # The first op is BlurGauss; its sigma reflects the winning trial's param.
    assert list(best_s0.get_ops().values())[0].sigma == 1.0  # A wins s0 (lowest cost)
    assert list(best_s1.get_ops().values())[0].sigma == 2.0  # B wins s1 (lowest cost)


def test_pareto_publication_excludes_non_finite_complete_trials(tmp_path):
    """NaN/inf COMPLETE trials consume budget but cannot become published winners."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli._run import _finalize_pareto_outputs

    spec = _single_objective_spec(tmp_path)
    store = JournalStudyStore([
        Trial(number=0, params={"0.sigma": 99.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": float("nan"), "s1": 0.0}),
        Trial(number=1, params={"0.sigma": 88.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": 0.0, "s1": float("inf")}),
        Trial(number=2, params={"0.sigma": 1.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": 0.1, "s1": 0.9}),
        Trial(number=3, params={"0.sigma": 2.0}, score=0.5, terms={},
              n_images=1, objectives={"s0": 0.9, "s1": 0.1}),
    ])
    out = tmp_path / "finite_pareto"

    assert store.completed_count() == 4
    _finalize_pareto_outputs(store, spec, out)

    front = pd.read_parquet(io.pareto_front_parquet_path(out))
    assert sorted(front["number"].tolist()) == [2, 3]
    best_s0 = ImagePipeline.from_json(
        io.pareto_best_pipeline_path(out, "s0").read_text()
    )
    best_s1 = ImagePipeline.from_json(
        io.pareto_best_pipeline_path(out, "s1").read_text()
    )
    best_knee = ImagePipeline.from_json(io.best_pipeline_path(out).read_text())
    assert list(best_s0.get_ops().values())[0].sigma == 1.0
    assert list(best_s1.get_ops().values())[0].sigma == 2.0
    assert list(best_knee.get_ops().values())[0].sigma in {1.0, 2.0}


def test_all_nonfinite_multiobjective_history_has_no_headline_or_scalar_label():
    """Multi-objective identity must survive an empty finite Pareto front."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli._run import (
        _headline_winner,
        _selection_label,
    )

    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": float("nan"), "s1": 0.0},
            ),
            Trial(
                number=1,
                params={},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.0, "s1": float("inf")},
            ),
        ]
    )

    assert store.pareto_front() == []
    assert _headline_winner(store) is None
    assert _selection_label(store) != "single_best"


def test_authoritative_axes_leave_all_partial_history_without_a_headline():
    """Headline identity and shape come from the scorer, not sidecar samples."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli._run import (
        _headline_winner,
        _selection_label,
    )

    required_axes = ("s0", "s1")
    partial_store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": 0.1},
            ),
            Trial(
                number=1,
                params={},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.2},
            ),
        ]
    )

    assert partial_store.pareto_front()
    assert (
        _headline_winner(
            partial_store,
            multi_objective=True,
            objective_axes=required_axes,
        )
        is None
    )
    assert (
        _selection_label(
            partial_store,
            multi_objective=True,
            objective_axes=required_axes,
        )
        == "pareto_knee"
    )

    full_trial = Trial(
        number=2,
        params={},
        score=0.3,
        terms={},
        n_images=1,
        objectives={"s0": 0.3, "s1": 0.4},
    )
    full_store = JournalStudyStore([full_trial])
    assert (
        _headline_winner(
            full_store,
            multi_objective=True,
            objective_axes=required_axes,
        )
        == full_trial
    )

    scalar_trial = Trial(
        number=3, params={}, score=0.05, terms={}, n_images=1
    )
    scalar_store = JournalStudyStore([scalar_trial])
    assert _headline_winner(scalar_store, multi_objective=False) == scalar_trial
    assert _selection_label(scalar_store, multi_objective=False) == "single_best"

@pytest.mark.parametrize(
    "objective_axes",
    [(), ("s0",), ("s0", "s0"), ("Dice", "dice"), ("", "s1")],
)
def test_selection_consumers_reject_invalid_authoritative_axes(objective_axes):
    """Direct selection helpers validate axes before inspecting store state."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli._run import (
        _headline_winner,
        _selection_label,
    )

    store = JournalStudyStore()

    with pytest.raises(ValueError, match="at least two|unique|non.?empty"):
        _headline_winner(store, objective_axes=objective_axes)
    with pytest.raises(ValueError, match="at least two|unique|non.?empty"):
        _selection_label(store, objective_axes=objective_axes)




def test_local_multiobjective_run_refuses_winnerless_publication(
    monkeypatch, tmp_path
):
    """Local finalization must stop before winner artifacts for an empty front."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli import _run as run_module

    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": float("nan"), "s1": 0.0},
            ),
            Trial(
                number=1,
                params={},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.0, "s1": float("inf")},
            ),
        ]
    )

    class _Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, _images):
            return store.best()

    monkeypatch.setattr(
        run_module,
        "_resolve_calibration_images",
        lambda *_args: (None, {}, []),
    )
    monkeypatch.setattr(
        run_module, "_open_store", lambda *_args, **_kwargs: store
    )
    monkeypatch.setattr(run_module, "TuningEngine", _Engine)
    out = tmp_path / "winnerless_local"

    with pytest.raises(RuntimeError, match="no valid winner"):
        run_tuning(_multi_objective_spec(tmp_path), [], out)

    assert not io.trials_parquet_path(out).exists()
    assert not io.param_importance_path(out).exists()
    assert not io.best_pipeline_path(out).exists()
    assert not io.best_params_path(out).exists()
    assert not io.pareto_dir(out).exists()


def test_local_multiobjective_run_refuses_all_partial_vectors_before_publication(
    monkeypatch, tmp_path
):
    """Scorer-required sibling axes must exist before winner publication."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli import _run as run_module

    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": 0.1},
            ),
            Trial(
                number=1,
                params={},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.2},
            ),
        ]
    )

    class _Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, _images):
            return store.best()

    monkeypatch.setattr(
        run_module,
        "_resolve_calibration_images",
        lambda *_args: (None, {}, []),
    )
    monkeypatch.setattr(
        run_module, "_open_store", lambda *_args, **_kwargs: store
    )
    monkeypatch.setattr(run_module, "TuningEngine", _Engine)
    monkeypatch.setattr(
        run_module,
        "_finalize_generalization",
        lambda *_args, **_kwargs: None,
    )
    out = tmp_path / "all_partial_local"
    publication_paths = (
        io.trials_parquet_path(out),
        io.param_importance_path(out),
        io.best_pipeline_path(out),
        io.best_params_path(out),
        io.pareto_front_parquet_path(out),
        io.pareto_best_pipeline_path(out, "s0"),
        io.pareto_importance_path(out, "s0"),
    )
    for index, path in enumerate(publication_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"incumbent-{index}\n".encode())
    before = {path: path.read_bytes() for path in publication_paths}

    with pytest.raises(RuntimeError, match="no valid winner"):
        run_tuning(_multi_objective_spec(tmp_path), [], out)

    assert {path: path.read_bytes() for path in publication_paths} == before


def test_local_multiobjective_run_rejects_unsafe_axis_before_publication(
    tmp_path
):
    """Local publication fails before an unsafe scorer axis can mutate outputs."""
    out = tmp_path / "unsafe_axis_local"
    incumbent = out / "incumbent.txt"
    incumbent.parent.mkdir(parents=True)
    incumbent.write_bytes(b"unchanged\n")

    with pytest.raises(ValueError, match="safe filename"):
        run_tuning(_unsafe_axis_spec(tmp_path), [], out)

    assert incumbent.read_bytes() == b"unchanged\n"
    assert list(out.rglob("*")) == [incumbent]


def test_local_multiobjective_run_rejects_duplicate_scorer_axes_before_publication(
    monkeypatch, tmp_path
):
    """Duplicate scorer axes must not collapse into one publishable coordinate."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli import _run as run_module

    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": 0.1},
            )
        ]
    )

    class _Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, _images):
            return store.best()

    monkeypatch.setattr(
        run_module,
        "_resolve_calibration_images",
        lambda *_args: (None, {}, []),
    )
    monkeypatch.setattr(
        run_module, "_open_store", lambda *_args, **_kwargs: store
    )
    monkeypatch.setattr(run_module, "TuningEngine", _Engine)
    monkeypatch.setattr(
        run_module,
        "_finalize_generalization",
        lambda *_args, **_kwargs: None,
    )
    out = tmp_path / "duplicate_axes_local"
    publication_paths = (
        io.trials_parquet_path(out),
        io.param_importance_path(out),
        io.best_pipeline_path(out),
        io.best_params_path(out),
        io.pareto_front_parquet_path(out),
        io.pareto_best_pipeline_path(out, "s0"),
        io.pareto_importance_path(out, "s0"),
    )
    for index, path in enumerate(publication_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"incumbent-{index}\n".encode())
    before = {path: path.read_bytes() for path in publication_paths}

    with pytest.raises(ValueError, match="unique"):
        run_tuning(_duplicate_axis_spec(tmp_path), [], out)

    assert {path: path.read_bytes() for path in publication_paths} == before


def test_local_multiobjective_rejects_casefold_axes_before_mutation(tmp_path):
    """Case-insensitive artifact aliases fail before local output mutation."""
    out = tmp_path / "casefold_axes_local"
    incumbent = out / "incumbent.txt"
    incumbent.parent.mkdir(parents=True)
    incumbent.write_bytes(b"unchanged\n")
    before = {
        path: path.read_bytes() for path in out.rglob("*") if path.is_file()
    }

    with pytest.raises(ValueError, match="case-insensitive|casefold|unique"):
        run_tuning(
            _casefold_axis_spec(tmp_path), [load_synth_yeast_plate()], out
        )

    assert {
        path: path.read_bytes() for path in out.rglob("*") if path.is_file()
    } == before



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

    assert _headline_winner(_Store(), multi_objective=True) is knee

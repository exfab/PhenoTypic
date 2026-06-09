"""4.9 — Phase-4 end-to-end: supervised + composite + Pareto over the synth plate.

The full multi-objective path on hermetic data: synthetic GT masks for
``load_synth_yeast_plate()`` drive a ``SupervisedScorer`` (region tier), composed
with a ``QCScorer`` (count) into a ``CompositeScorer(multi_objective=True)``, and
tuned via an Optuna NSGA-II strategy (grid/random are rejected for
multi-objective, 4.8). The run must publish ``deliverables/pareto/`` (a non-empty
front + per-objective best pipelines), with the **knee** as the top-level
``best_pipeline.json``; a single-objective sibling run writes **no** ``pareto/``
dir (the back-compat lock). NSGA-II body is ``skipif`` when the ``tune`` extra is
absent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Budget,
    Categorical,
    CompositeScorer,
    Evaluator,
    GridConfig,
    GroundTruthMasks,
    Knob,
    OptunaConfig,
    QCScorer,
    SearchSpace,
    SupervisedScorer,
    TuningSpec,
)
from phenotypic.tune._tune_cli._run import run_tuning

_OPTUNA = importlib.util.find_spec("optuna") is not None


def _synthetic_gt_mask_dir(tmp_path: Path) -> Path:
    """Write a synthetic per-image GT mask matching the synth plate's name.

    The mask is the plate's own foreground (its objmap > 0) — a perfect-GT
    boolean array — saved as ``<image.name>.npy`` so
    ``GroundTruthMasks.masks_for`` resolves it by stem.
    """
    image = load_synth_yeast_plate()
    mask = np.asarray(image.objmap[:]) > 0
    gt_dir = tmp_path / "gt_masks"
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.save(gt_dir / f"{image.name}.npy", mask)
    return gt_dir


def _count_csv(tmp_path: Path) -> str:
    csv = tmp_path / "counts.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return str(csv)


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _supervised_region_scorer(tmp_path: Path) -> SupervisedScorer:
    return SupervisedScorer(
        gt=GroundTruthMasks(gt_masks_source=_synthetic_gt_mask_dir(tmp_path)),
        region_metric="dice",
        match_strategy="grid_cell",
    )


def _qc_count_scorer(tmp_path: Path) -> QCScorer:
    return QCScorer(check=ExpectedVsDetectedCount(
        metadata=_count_csv(tmp_path), groupby=["Metadata_ImageName"]))


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_supervised_composite_pareto_end_to_end(tmp_path):
    composite = CompositeScorer(
        scorers=[
            _supervised_region_scorer(tmp_path),  # s0 = region (Dice)
            _qc_count_scorer(tmp_path),           # s1 = count
        ],
        multi_objective=True,
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=composite,
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="nsga2", n_trials=4),
        budget=Budget(n_trials=4),
    )
    out = tmp_path / "run_mo"
    run_tuning(spec, [load_synth_yeast_plate()], out)

    # deliverables/pareto/ exists with a non-empty front parquet.
    assert io.pareto_dir(out).exists()
    front = pd.read_parquet(io.pareto_front_parquet_path(out))
    assert len(front) >= 1
    assert front["objectives_json"].notna().any()

    # Per-objective best pipelines land for both composite axes.
    for objective in ("s0", "s1"):
        per_axis = io.pareto_best_pipeline_path(out, objective)
        assert per_axis.exists()
        ImagePipeline.from_json(per_axis.read_text())

    # best_pipeline.json IS the knee (reloads runnable + equals the knee build).
    from phenotypic.tune._evaluation import build_pipeline
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    store = OptunaStudyStore(
        storage_url=f"sqlite:///{io.tune_cache_study_db_path(out)}",
        study_name="tune",
        directions=["maximize", "maximize"],
    )
    knee = store.knee_point(store.pareto_front())
    assert knee is not None
    expected_knee_json = build_pipeline(spec.pipeline, knee.params).to_json()
    assert io.best_pipeline_path(out).read_text() == expected_knee_json
    best_params = pd.read_json(io.best_params_path(out), typ="series").to_dict()
    assert best_params["selection"] == "pareto_knee"
    assert best_params["params"] == knee.params


def test_single_objective_sibling_writes_no_pareto(tmp_path):
    # The single-objective sibling: a plain QCScorer + grid → NO pareto/ dir.
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=_space(),
        scorer=_qc_count_scorer(tmp_path),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    out = tmp_path / "run_so"
    run_tuning(spec, [load_synth_yeast_plate()], out)

    assert io.best_pipeline_path(out).exists()
    assert io.best_params_path(out).exists()
    assert io.trials_parquet_path(out).exists()
    assert not io.pareto_dir(out).exists()

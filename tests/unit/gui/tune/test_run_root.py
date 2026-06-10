"""``TuneRunRoot.discover`` — validate + describe a tune output directory.

Covers the three marker-precedence cases (run.json first, then the resolved
``tuning_spec.json``, then a legacy trials-only root) and the no-marker error.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import (
    best_pipeline_path,
    trials_parquet_path,
    tune_cache_run_marker_path,
    tuning_spec_path,
)
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    OptunaConfig,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec

from phenotypic.gui.tune import TuneRunRoot, TuneRunRootError


def _optuna_spec(tmp_path, *, storage_url: str) -> TuningSpec:
    """A minimal valid single-objective ``TuningSpec`` on an ``OptunaConfig``.

    Mirrors ``tests/unit/tune/test_run_marker.py::_spec`` but swaps the grid
    strategy for an ``OptunaConfig`` so ``strategy.storage_url`` is populated.
    """
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["Metadata_ImageName"])),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=4, storage_url=storage_url),
        budget=Budget(),
    )


def test_discover_reads_storage_url_and_trials(tmp_path):
    """No run.json: fall back to ``tuning_spec.json`` for the URL + study name."""
    out = tmp_path / "out"
    (out / "deliverables").mkdir(parents=True)
    spec = _optuna_spec(tmp_path, storage_url="sqlite:///x.db")
    tuning_spec_path(out).write_text(spec.model_dump_json())
    trials_parquet_path(out).write_bytes(b"")

    root = TuneRunRoot.discover(out)

    assert root.storage_url == "sqlite:///x.db"
    assert root.study_name == "tune_cost_v1"
    assert root.trials_path == trials_parquet_path(out)
    assert root.best_pipeline_path == best_pipeline_path(out)
    # Single-objective spec → no Pareto directions.
    assert root.directions is None


def test_discover_reads_legacy_tuning_spec_json(tmp_path):
    """Legacy plain JSON tuning specs still seed tune-run discovery."""
    out = tmp_path / "out"
    deliverables = out / "deliverables"
    deliverables.mkdir(parents=True)
    spec = _optuna_spec(tmp_path, storage_url="sqlite:///legacy.db")
    (deliverables / "tuning_spec.json").write_text(spec.model_dump_json())
    trials_parquet_path(out).write_bytes(b"")

    root = TuneRunRoot.discover(out)

    assert root.storage_url == "sqlite:///legacy.db"
    assert root.study_name == "tune_cost_v1"
    assert root.best_pipeline_path == best_pipeline_path(out)


def test_discover_reads_run_marker_first(tmp_path):
    """``run.json`` wins: its URL + images_dir + study_name are authoritative."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    # A conflicting tuning_spec.json must be IGNORED in favour of the marker.
    (out / "deliverables").mkdir(parents=True)
    spec = _optuna_spec(tmp_path, storage_url="sqlite:///ignored.db")
    tuning_spec_path(out).write_text(spec.model_dump_json())

    marker = tune_cache_run_marker_path(out)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "version": 1,
        "study_name": "tune",
        "storage_url": "postgresql://host/tune",
        "images_dir": str(tmp_path / "calib"),
        "strategy": "optuna",
        "n_trials": 4,
        "is_multi_objective": True,
        "slurm": False,
        "start_time": "2026-06-05T00:00:00+00:00",
    }))
    trials_parquet_path(out).write_bytes(b"")

    root = TuneRunRoot.discover(out)

    assert root.storage_url == "postgresql://host/tune"
    assert root.study_name == "tune"
    assert root.images_dir == tmp_path / "calib"
    # is_multi_objective=True → two-axis maximize directions.
    assert root.directions is not None
    assert len(root.directions) >= 2


def test_discover_raises_when_neither_study_nor_trials(tmp_path):
    """An empty directory is not a tune output — discover rejects it."""
    with pytest.raises(TuneRunRootError):
        TuneRunRoot.discover(tmp_path)

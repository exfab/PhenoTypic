from __future__ import annotations

import json

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._tune_cli._run import run_tuning


def _spec(tmp_path) -> TuningSpec:
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
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_run_tuning_writes_deliverables(tmp_path):
    out = tmp_path / "run"
    best = run_tuning(_spec(tmp_path), [load_synth_yeast_plate()], out)

    assert io.best_pipeline_path(out).exists()
    assert io.tuning_spec_path(out).exists()
    assert io.param_importance_path(out).exists()
    assert io.trials_parquet_path(out).exists()
    # the written best pipeline reloads as a runnable ImagePipeline
    winner = ImagePipeline.from_json(io.best_pipeline_path(out).read_text())
    assert "OtsuDetector" in winner.get_ops()
    # importance covers the tuned knob
    imp = json.loads(io.param_importance_path(out).read_text())
    assert "1.ignore_zeros" in imp
    assert best is not None


def test_cli_main_invokes_run(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"

    # patch image loading (no PNG fixtures needed)
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])
    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()

"""``python -m phenotypic.tune`` subcommand split + back-compat (P3-6).

The CLI splits into ``run`` (the engine) and ``auto-space`` (infer-only). A bare
``spec.json`` positional with no subcommand still defaults to ``run`` so the
Phase-1 invocation ``python -m phenotypic.tune spec.json -i … -o …`` keeps
working.
"""
from __future__ import annotations

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


def test_bare_spec_positional_defaults_to_run(tmp_path, monkeypatch):
    """Back-compat: ``tune spec.json -i … -o …`` (no subcommand) runs the engine."""
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])

    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()
    assert io.trials_parquet_path(out).exists()  # the engine ran


def test_explicit_run_subcommand_runs_the_engine(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_load_images", lambda _p: [load_synth_yeast_plate()])

    cli.main(["run", str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()


def test_auto_space_subcommand_infers_without_running_engine(tmp_path):
    from phenotypic.tune import __main__ as cli

    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(
        ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()]).to_json() or ""
    )
    out = tmp_path / "auto"

    cli.main(["auto-space", str(pipe_path), "-o", str(out)])

    assert io.tuning_spec_path(out).exists()
    assert not io.trials_parquet_path(out).exists()  # no engine run

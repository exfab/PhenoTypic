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
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig
from phenotypic.tune._spec import Budget, TuningSpec


def _spec(tmp_path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {"MetadataImage_ImageName": ["Synthetic96PlateWithObjects"] * 96,
         "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(csv), groupby=["MetadataImage_ImageName"])),
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
    monkeypatch.setattr(
        cli,
        "_load_images",
        lambda _p, *, nrows=None, ncols=None: [load_synth_yeast_plate()],
    )

    cli.main([str(spec_path), "-i", str(tmp_path), "-o", str(out)])

    assert io.best_pipeline_path(out).exists()
    assert io.trials_parquet_path(out).exists()  # the engine ran


def test_explicit_run_subcommand_runs_the_engine(tmp_path, monkeypatch):
    from phenotypic.tune import __main__ as cli

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli,
        "_load_images",
        lambda _p, *, nrows=None, ncols=None: [load_synth_yeast_plate()],
    )

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


# -- E2: held-out CLI overrides reach HeldOutConfig ----------------------------


def test_held_out_flags_override_spec(tmp_path, monkeypatch):
    """``--held-out-fraction`` + ``--cv-group`` reach the resolved spec's policy."""
    from phenotypic.tune import __main__ as cli
    from phenotypic.tune._tune_cli import _run as run_mod

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli,
        "_load_images",
        lambda _p, *, nrows=None, ncols=None: [load_synth_yeast_plate()],
    )

    captured: dict = {}
    original = run_mod.run_tuning

    def _capture(spec, images, output_dir, **kwargs):
        captured["held_out_fraction"] = kwargs.get("held_out_fraction")
        captured["cv_group"] = kwargs.get("cv_group")
        return original(spec, images, output_dir, **kwargs)

    monkeypatch.setattr(cli, "run_tuning", _capture)

    cli.main([
        "run", str(spec_path), "-i", str(tmp_path), "-o", str(out),
        "--held-out-fraction", "0.25", "--cv-group", "MetadataPlate_Batch",
    ])

    # The flags threaded through _run_command → run_tuning.
    assert captured["held_out_fraction"] == 0.25
    assert captured["cv_group"] == "MetadataPlate_Batch"
    # The resolved + persisted spec carries the overridden HeldOutConfig.
    from phenotypic.tune._spec import TuningSpec as _Spec

    resolved = _Spec.model_validate_json(io.tuning_spec_path(out).read_text())
    assert resolved.held_out.held_out_fraction == 0.25
    assert resolved.held_out.group_key == "MetadataPlate_Batch"


def test_run_tuning_held_out_overrides_directly(tmp_path, monkeypatch):
    """``run_tuning(..., held_out_fraction=, cv_group=)`` overrides the spec block."""
    from phenotypic.tune._tune_cli._run import run_tuning

    spec = _spec(tmp_path)
    out = tmp_path / "out"

    run_tuning(
        spec,
        [load_synth_yeast_plate()],
        out,
        held_out_fraction=0.3,
        cv_group="Metadata_Plate",
    )

    from phenotypic.tune._spec import TuningSpec as _Spec

    resolved = _Spec.model_validate_json(io.tuning_spec_path(out).read_text())
    assert resolved.held_out.held_out_fraction == 0.3
    assert resolved.held_out.group_key == "Metadata_Plate"
    # gap margins are spec-only (untouched by the flags) — keep their defaults.
    assert resolved.held_out.gap_margin_relative == spec.held_out.gap_margin_relative
    assert resolved.held_out.gap_margin_absolute == spec.held_out.gap_margin_absolute

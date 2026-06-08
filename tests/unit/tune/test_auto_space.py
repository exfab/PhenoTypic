"""``run_auto_space`` — infer + persist a proposal, no engine run (P3-6).

``run_auto_space(pipeline, output_dir)`` mines a pipeline with
``infer_search_space``, writes the reviewable ``InferredSearchSpace`` proposal to
``deliverables/tuning_spec.json`` (via ``io.tuning_spec_path``), and returns it.
It runs **no** engine: no ``trials.parquet`` is written.
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, OtsuDetector, RoundPeaksDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import InferredSearchSpace
from phenotypic.tune._tune_cli._auto_space import run_auto_space


def _pipe() -> ImagePipeline:
    return ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])


def test_run_auto_space_writes_tuning_spec_json(tmp_path):
    out = tmp_path / "auto"
    proposal = run_auto_space(_pipe(), out)
    spec_path = io.tuning_spec_path(out)
    assert spec_path.exists()
    assert isinstance(proposal, InferredSearchSpace)


def test_run_auto_space_does_not_run_the_engine(tmp_path):
    out = tmp_path / "auto"
    run_auto_space(_pipe(), out)
    # no engine ran → no trial journal
    assert not io.trials_parquet_path(out).exists()
    assert not io.best_pipeline_path(out).exists()
    assert not io.param_importance_path(out).exists()


def test_written_proposal_round_trips_from_json(tmp_path):
    out = tmp_path / "auto"
    proposal = run_auto_space(_pipe(), out)
    reloaded = InferredSearchSpace.model_validate_json(
        io.tuning_spec_path(out).read_text()
    )
    assert reloaded == proposal


def test_run_auto_space_includes_nested_knobs(tmp_path):
    out = tmp_path / "auto"
    pipe = ImagePipeline(ops=[
        CompositeDetector(detectors=[OtsuDetector(), RoundPeaksDetector()]),
    ])
    proposal = run_auto_space(pipe, out)
    keys = {k.key for k in proposal.knobs}
    assert any(k.startswith("0.detectors[0].") for k in keys)

"""Phase 5 Chunk 0 — the ``.pht-tune-cache/run.json`` tune-run marker.

The marker is written at run START (right after the ``deliverables/`` mkdir,
BEFORE the engine/SLURM branch) so a live tune output is GUI-discoverable
before any deliverable lands. It carries the study identity + a RESOLVED,
non-null storage URL + the run policy. The non-null URL is load-bearing: a
null URL would silently force the GUI Monitor into parquet-only mode for the
env-var-driven distributed-Postgres case.
"""
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
from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV
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


def test_run_marker_written_with_required_keys(tmp_path):
    """A local run writes ``run.json`` with the full key set + a non-null URL."""
    out = tmp_path / "run"
    images_dir = tmp_path / "images"
    run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        images_dir=images_dir,
    )
    marker_path = io.tune_cache_run_marker_path(out)
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text())
    assert marker["version"] == 1
    assert marker["study_name"] == "tune"
    assert marker["strategy"] == "grid"
    assert marker["is_multi_objective"] is False
    assert marker["slurm"] is False
    assert marker["images_dir"] == str(images_dir)
    assert "start_time" in marker and marker["start_time"]
    # The storage URL must be resolved + non-null. A grid run has no explicit
    # URL and no env var → it falls back to the local study.db under the cache.
    assert marker["storage_url"]
    assert "study.db" in marker["storage_url"]


def test_run_marker_records_env_storage_url(tmp_path, monkeypatch):
    """With $PHENOTYPIC_TUNE_STORAGE_URL set and a None param, the marker records
    the env URL (not null) — the distributed-Postgres Monitor case."""
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, "postgresql://host/tune")
    out = tmp_path / "run"
    run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        storage_url=None,
        images_dir=tmp_path / "images",
    )
    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    assert marker["storage_url"] == "postgresql://host/tune"


def test_run_marker_written_before_slurm_branch(tmp_path, monkeypatch):
    """The marker exists even for a fire-and-forget SLURM submission (it is
    written before the engine/SLURM branch). The strategy + slurm flag reflect
    the run."""
    from phenotypic.tune._tune_cli import _run as run_mod

    class _FakeSlurmExecutor:
        def __init__(self, **kwargs):
            pass

        def run(self, work, items):
            return ["9001"]

    monkeypatch.setattr(run_mod, "SlurmExecutor", _FakeSlurmExecutor)

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    out = tmp_path / "slurm_out"
    run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        strategy="tpe",
        n_trials=4,
        storage_url=f"sqlite:///{out / 'explicit.db'}",
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path,
    )
    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    assert marker["slurm"] is True
    assert marker["strategy"] == "tpe"
    assert marker["n_trials"] == 4
    # The explicit URL wins the 3-way fallback.
    assert marker["storage_url"] == f"sqlite:///{out / 'explicit.db'}"

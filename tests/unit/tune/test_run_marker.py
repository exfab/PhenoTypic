"""Phase 5 Chunk 0 — the ``.pht-tune-cache/run.json`` tune-run marker.

The marker is written at run START (right after the ``deliverables/`` mkdir,
BEFORE the engine/SLURM branch) so a live tune output is GUI-discoverable
before any deliverable lands. It carries the study identity + a RESOLVED,
non-null storage URL + the run policy. The non-null URL is load-bearing: a
null URL would silently force the GUI Monitor into parquet-only mode for the
env-var-driven distributed-Postgres case.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.sdk_ import _io_constants as io
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
from phenotypic.tune._tune_cli._run import _STUDY_NAME, run_tuning

_OPTUNA = importlib.util.find_spec("optuna") is not None


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
    """A local grid run writes ``run.json`` with the full key set and no URL."""
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
    # Study name bumped for the minimize-cost cutover (Phase 2, OQ7).
    assert marker["study_name"] == _STUDY_NAME
    assert marker["strategy"] == "grid"
    assert marker["is_multi_objective"] is False
    assert marker["slurm"] is False
    assert marker["images_dir"] == str(images_dir)
    assert "start_time" in marker and marker["start_time"]
    # Non-Optuna runs have no live Optuna storage; the GUI should read the
    # finished parquet journal instead of trying a bogus live study.
    assert marker["storage_url"] is None


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_run_marker_records_env_storage_url_for_optuna(tmp_path, monkeypatch):
    """An Optuna run with no explicit URL records the env URL."""
    from phenotypic.tune._study_store import JournalStudyStore
    from phenotypic.tune._tune_cli import _run as run_mod

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            return None

    monkeypatch.setenv(
        PHENOTYPIC_TUNE_STORAGE_URL_ENV, "postgresql+psycopg://host/tune"
    )
    monkeypatch.setattr(run_mod, "_open_store", lambda *a, **kw: JournalStudyStore())
    monkeypatch.setattr(run_mod, "TuningEngine", _FakeEngine)
    out = tmp_path / "run"
    run_tuning(
        _spec(tmp_path),
        [load_synth_yeast_plate()],
        out,
        strategy="tpe",
        n_trials=2,
        storage_url=None,
        images_dir=tmp_path / "images",
    )
    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    assert marker["storage_url"] == "postgresql+psycopg://host/tune"


def test_run_marker_ignores_env_storage_url_for_grid(tmp_path, monkeypatch):
    """A grid run remains journal-only even when the Optuna env var is set."""
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
    assert marker["storage_url"] is None


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
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


def test_run_proceeds_when_marker_write_fails(tmp_path, monkeypatch, caplog):
    """A read-only / over-quota output FS raises OSError on the marker write.

    The marker is a sidecar, not a deliverable — its failure must NOT abort the
    run (HPCC robustness). ``run_tuning`` catches the OSError, logs a warning,
    and proceeds to write the real deliverables.
    """
    import logging

    from phenotypic.tune._tune_cli import _run as run_mod

    out = tmp_path / "run"
    marker_target = run_mod.io.tune_cache_run_marker_path(out)
    real_atomic_write_text = run_mod.atomic_write_text

    def _exploding_atomic_write_text(path, text, **kwargs):
        # Only the GUI-discovery marker write fails (read-only / over-quota FS);
        # the real deliverables (tuning_spec.json, best_pipeline.json) still write.
        if Path(path) == Path(marker_target):
            raise OSError("Read-only file system")
        return real_atomic_write_text(path, text, **kwargs)

    monkeypatch.setattr(run_mod, "atomic_write_text", _exploding_atomic_write_text)

    with caplog.at_level(logging.WARNING):
        best = run_tuning(
            _spec(tmp_path),
            [load_synth_yeast_plate()],
            out,
            images_dir=tmp_path / "images",
        )

    # The run proceeded past the marker step: deliverables were written.
    assert io.best_pipeline_path(out).exists()
    assert io.tuning_spec_path(out).exists()
    assert best is not None
    # And the marker failure was logged, not swallowed silently.
    assert any(
        record.levelno == logging.WARNING and "run.json" in record.getMessage()
        for record in caplog.records
    )

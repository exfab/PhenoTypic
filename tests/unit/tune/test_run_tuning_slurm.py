"""The ``--slurm`` fleet submits the RESOLVED spec to its workers (not the raw input).

A ``--strategy``/``--n-trials``/``--held-out`` override is folded into a resolved
spec that ``run_tuning`` persists to ``deliverables/tuning_spec.json`` *before*
submitting. The fleet's workers must reload **that** resolved spec — otherwise a
``python -m phenotypic.tune run spec.json --slurm --strategy tpe`` silently runs
the input spec's (grid) strategy on every worker, and the distributed Optuna
study never happens. This locks the worker ``--spec`` onto the resolved file.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tools_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    Scorer,
    SearchSpace,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._tune_cli._run import run_tuning

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


class _ConstScorer(Scorer):
    def score_image(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _grid_input_spec() -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),  # the INPUT spec is grid…
        budget=Budget(),
    )


def test_slurm_fleet_workers_reload_the_resolved_spec(tmp_path, monkeypatch):
    captured: dict = {}

    class _FakeExecutor:
        def __init__(self, **kw):
            captured.update(kw)

        def run(self, work, items):  # no live SLURM
            return None

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )

    spec = _grid_input_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    out = tmp_path / "out"

    # …but the run overrides --strategy tpe; the fleet must carry THAT, not grid.
    run_tuning(
        spec,
        images=[],
        output_dir=out,
        strategy="tpe",
        n_trials=4,
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path / "imgs",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
    )

    # The worker --spec points at the RESOLVED deliverables/tuning_spec.json…
    assert captured["spec_path"] == io.tuning_spec_path(out)
    # …whose strategy is the Optuna override, not the input grid strategy. Parse
    # the JSON directly (the test-local scorer isn't registry-resolvable, but the
    # strategy block — the thing under test — serializes plainly).
    persisted = json.loads(io.tuning_spec_path(out).read_text())
    assert persisted["strategy"]["class"] == "OptunaConfig"
    assert persisted["strategy"]["params"]["n_trials"] == 4
    # …and workers launch with the submitter's own venv interpreter (absolute
    # sys.executable), not a bare ``python`` that a compute node can't resolve.
    assert captured["python_command"] == [sys.executable]


def test_slurm_fleet_pre_creates_the_shared_study(tmp_path, monkeypatch):
    # The submitter must materialize the study (+ RDB schema) BEFORE the fleet
    # starts, so cold-DB workers don't race to CREATE the Optuna schema (the live
    # UniqueViolation on studydirection). After submission the study must already
    # exist in the storage.
    import optuna

    class _FakeExecutor:
        def __init__(self, **kw):
            pass

        def run(self, work, items):
            return None

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )

    spec = _grid_input_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    out = tmp_path / "out"
    url = f"sqlite:///{tmp_path / 'study.db'}"

    run_tuning(
        spec,
        images=[],
        output_dir=out,
        strategy="tpe",
        n_trials=4,
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path / "imgs",
        storage_url=url,
    )

    # The study exists in the storage (pre-created) — load_study does not raise.
    study = optuna.load_study(study_name="tune", storage=url)
    assert study.study_name == "tune"

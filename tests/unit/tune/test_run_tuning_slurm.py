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
from dataclasses import dataclass

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import (
    QCScorer,
    Scorer,
)
from phenotypic.tune.strategy import (
    GridConfig,
    OptunaConfig,
)
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._tune_cli._run import _STUDY_NAME, run_tuning

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


class _ConstScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _grid_input_spec_with_scorer(scorer: Scorer) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=GridConfig(),  # the INPUT spec is grid…
        budget=Budget(),
    )


def _grid_input_spec() -> TuningSpec:
    return _grid_input_spec_with_scorer(_ConstScorer())


def _registry_resolvable_grid_input_spec(tmp_path) -> TuningSpec:
    layout_path = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["cal"], "Object_Label": [1]}
    ).to_csv(layout_path, index=False)
    return _grid_input_spec_with_scorer(
        QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(layout_path), groupby=["Metadata_ImageName"]
            )
        )
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
    assert captured["split_path"] == io.tune_cache_split_assignment_path(out)


def test_slurm_rejects_grid_strategy_before_submission(tmp_path, monkeypatch):
    submitted = False

    class _FakeExecutor:
        def __init__(self, **kw):
            nonlocal submitted
            submitted = True

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )
    spec = _grid_input_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="--slurm.*Optuna"):
        run_tuning(
            spec,
            images=[],
            output_dir=out,
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "imgs",
        )

    assert submitted is False
    assert not io.tuning_spec_path(out).exists()


def test_slurm_rejects_non_positive_worker_count(tmp_path, monkeypatch):
    class _FakeExecutor:
        def __init__(self, **kw):
            raise AssertionError("executor must not be constructed")

    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run.SlurmExecutor", _FakeExecutor
    )
    spec = _grid_input_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())

    with pytest.raises(ValueError, match="n-workers"):
        run_tuning(
            spec,
            images=[],
            output_dir=tmp_path / "out",
            strategy="tpe",
            n_trials=4,
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "imgs",
            storage_url=f"sqlite:///{tmp_path / 'study.db'}",
            n_workers=0,
        )


@dataclass
class _NamedImage:
    name: str


def test_worker_filters_held_out_images_from_split(tmp_path, monkeypatch):
    from phenotypic.tune._evaluation._split import Split, write_split
    from phenotypic.tune._tune_cli import _worker

    split = Split(
        calibration=["cal"],
        held_out=["hold"],
        kind="within_group",
        group_key=None,
        dataset_identity="abc",
        seed_entropy=[1],
    )
    write_split(tmp_path, split)
    split_path = io.tune_cache_split_assignment_path(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        _registry_resolvable_grid_input_spec(tmp_path)
        .model_copy(update={"strategy": OptunaConfig(n_trials=1)})
        .model_dump_json()
    )
    seen = {}

    class _FakeStore:
        pass

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            seen["names"] = [im.name for im in images]

    monkeypatch.setattr(
        _worker,
        "_load_images",
        lambda _path, *, nrows=None, ncols=None: [
            _NamedImage("cal"),
            _NamedImage("hold"),
        ],
    )
    monkeypatch.setattr(_worker, "build_worker_store", lambda **_kw: _FakeStore())
    monkeypatch.setattr("phenotypic.tune._engine.TuningEngine", _FakeEngine)

    _worker.run_worker(
        spec_path=spec_path,
        images_dir=tmp_path,
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
        study_name="tune",
        split_path=split_path,
    )

    assert seen["names"] == ["cal"]


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
    # Study name bumped for the minimize-cost cutover (Phase 2, OQ7).
    study = optuna.load_study(study_name=_STUDY_NAME, storage=url)
    assert study.study_name == _STUDY_NAME

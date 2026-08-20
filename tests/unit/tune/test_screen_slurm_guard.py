"""``--screen`` + ``--slurm`` is refused, and refused *before* anything is written.

The bug this locks: ``run_tuning`` returns at the ``if slurm:`` branch before it
ever reaches ``if screen:``, and ``_worker.run_worker`` constructs a bare
``TuningEngine`` with no ``ScreeningController``. So the combination used to
submit the **full unscreened space** to the fleet — no error, no warning, just
different behaviour than the caller asked for.

Two things are under test, and the second is the one that is easy to get wrong:

1. the combination raises ``ValueError``;
2. it raises from ``_validate_slurm_request``, i.e. **before** the
   ``deliverables/`` mkdir, the ``tuning_spec.json`` echo and the ``run.json``
   marker. A guard sitting immediately above ``if slurm:`` (where the task
   originally put it) still raises, but leaves a half-built output directory
   that the GUI shell classifier then reports as a live tune run.
"""
from __future__ import annotations

import importlib.util

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import Budget, Categorical, Evaluator, Knob, SearchSpace
from phenotypic.tune.score import Scorer
from phenotypic.tune.strategy import GridConfig
from phenotypic.tune._spec import TuningSpec
from phenotypic.tune._study._storage import journal_url_for_path
from phenotypic.tune._tune_cli import _run as run_mod
from phenotypic.tune._tune_cli._run import run_tuning

_OPTUNA = importlib.util.find_spec("optuna") is not None


class _ConstScorer(Scorer):
    def _score_terms(self, image, measurements) -> dict[str, float]:
        return {"Count": 1.0}


def _spec() -> TuningSpec:
    """A minimal, side-effect-free spec: the engine is stubbed in every test."""
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=_ConstScorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def _stub_slurm_executor(monkeypatch) -> dict:
    """Replace ``SlurmExecutor`` so no live ``sbatch`` is reached."""
    captured: dict = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self, work, items):
            captured["submitted"] = list(items)
            return []

    monkeypatch.setattr(run_mod, "SlurmExecutor", _FakeExecutor)
    return captured


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_screen_plus_slurm_is_refused(tmp_path, monkeypatch):
    """The combination raises instead of submitting an unscreened fleet."""
    captured = _stub_slurm_executor(monkeypatch)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec().model_dump_json())
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="screen.*slurm|slurm.*screen"):
        run_tuning(
            _spec(),
            [],
            out,
            strategy="tpe",
            n_trials=4,
            screen=True,
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "imgs",
            storage_url=f"sqlite:///{tmp_path / 'study.db'}",
        )

    assert "submitted" not in captured, "the fleet must not be submitted"


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_the_refusal_writes_no_run_artifacts(tmp_path, monkeypatch):
    """The guard lives in ``_validate_slurm_request``, above every write.

    This is what fails if the guard is moved down to just above ``if slurm:``:
    ``deliverables/tuning_spec.json`` and ``.pht-tune-cache/run.json`` are both
    written between the validator and that branch, so a refused run would leave
    an output directory the GUI classifies as a live tune output.
    """
    _stub_slurm_executor(monkeypatch)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec().model_dump_json())
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="screen"):
        run_tuning(
            _spec(),
            [],
            out,
            strategy="tpe",
            n_trials=4,
            screen=True,
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "imgs",
            storage_url=f"sqlite:///{tmp_path / 'study.db'}",
        )

    assert not io.tuning_spec_path(out).exists()
    assert not io.tune_cache_run_marker_path(out).exists()
    assert not io.deliverables_dir(out).exists()


def test_screen_alone_still_works(tmp_path, monkeypatch):
    """``--screen`` without ``--slurm`` still drives the screening controller.

    The controller is stubbed: this asserts the *routing* (screen → the
    two-round freeze, not the plain engine), which is what the guard could
    plausibly break. The freeze itself is covered by ``test_screening_freeze``.
    """
    seen: dict = {}

    class _FakeResult:
        winner = None

    class _FakeStore:
        trials: list = []

    class _FakeController:
        def __init__(self, spec, config=None):
            seen["constructed"] = True
            self.explore_store = _FakeStore()
            self.focused_store = None

        def run(self, images):
            seen["ran"] = True
            return _FakeResult()

    class _FakeEngine:
        def __init__(self, spec, store):
            raise AssertionError("the plain engine must not run under --screen")

    monkeypatch.setattr(run_mod, "ScreeningController", _FakeController)
    monkeypatch.setattr(run_mod, "TuningEngine", _FakeEngine)

    out = tmp_path / "out"
    run_tuning(_spec(), [], out, screen=True, slurm=False)

    assert seen == {"constructed": True, "ran": True}


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_slurm_alone_still_submits(tmp_path, monkeypatch):
    """``--slurm`` without ``--screen`` is untouched by the guard."""
    captured = _stub_slurm_executor(monkeypatch)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec().model_dump_json())
    out = tmp_path / "out"

    run_tuning(
        _spec(),
        [],
        out,
        strategy="tpe",
        n_trials=4,
        screen=False,
        slurm=True,
        spec_path=spec_path,
        images_dir=tmp_path / "imgs",
        storage_url=journal_url_for_path(tmp_path / "study.log"),
    )

    assert captured["submitted"] == [0, 1, 2, 3]

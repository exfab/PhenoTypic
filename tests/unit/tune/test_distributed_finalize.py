"""A ``--slurm`` study can be finalized after the fact.

``run_tuning`` with ``--slurm`` returns at the submission branch, above every
``deliverables/`` write. So a distributed study ends with a full Optuna store and
an output directory missing ``trials.parquet``, ``param_importance.json``,
``best_pipeline.json``, ``generalization.json`` and — the one that bites —
``best_params.json``, which ``prepare_best_from_run`` hard-requires. The plain
export path therefore raises ``FileNotFoundError`` on **every** distributed study.

The fixtures below build a real finished study: a real Optuna SQLite store with a
drained budget of real ``Trial`` rows, a real resolved ``tuning_spec.json``, a
real ``run.json`` marker, and real plate PNGs on disk under the ``images_dir`` the
marker records. Nothing about the finalize is stubbed except where a test is
explicitly about ordering or interruption.
"""
from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import Budget, Categorical, Evaluator, Knob, SearchSpace
from phenotypic.tune.score import QCScorer
from phenotypic.tune._spec import TuningSpec
from phenotypic.tune._study_store import Trial
from phenotypic.tune._tune_cli._run import _STUDY_NAME

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")

#: The recorded trial budget. The terminal gate compares the store's trial count
#: against it, so "finished" and "running" differ only in how many are written.
_BUDGET = 4

_PLATE_NAMES = ("plate_a", "plate_b", "plate_c", "plate_d")


def _write_plates(images_dir):
    """Save four clones of the synthetic plate so the finalize can re-scan them."""
    images_dir.mkdir(parents=True, exist_ok=True)
    plate = load_synth_yeast_plate()
    for name in _PLATE_NAMES:
        plate.rgb.imsave(images_dir / f"{name}.png")
    return images_dir


def _layout_csv(tmp_path):
    """96 expected objects per plate — the synthetic plate's true count."""
    rows = [
        {"Metadata_ImageName": name, "Object_Label": j}
        for name in _PLATE_NAMES
        for j in range(96)
    ]
    csv = tmp_path / "layout.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _resolved_spec(tmp_path, storage_url: str) -> TuningSpec:
    """The spec a fleet worker would have reloaded: Optuna, registry-resolvable."""
    from phenotypic.tune.strategy import OptunaConfig

    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(_layout_csv(tmp_path)), groupby=["Metadata_ImageName"],
        )),
        evaluator=Evaluator(),
        strategy=OptunaConfig(sampler="tpe", n_trials=_BUDGET,
                              storage_url=storage_url),
        budget=Budget(),
    )


def _build_study(tmp_path, *, n_recorded: int):
    """A distributed study directory holding ``n_recorded`` trials.

    Mirrors exactly what ``run_tuning --slurm`` leaves behind: the resolved spec
    echo and the ``run.json`` marker (both written above the submission branch),
    a shared study the fleet appended to, and NOTHING else.
    """
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    out = tmp_path / "out"
    images_dir = _write_plates(tmp_path / "plates")
    io.deliverables_dir(out).mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{io.tune_cache_study_db_path(out)}"
    io.tune_cache_dir(out).mkdir(parents=True, exist_ok=True)

    spec = _resolved_spec(tmp_path, storage_url)
    io.tuning_spec_path(out).write_text(spec.model_dump_json(indent=2))

    store = OptunaStudyStore(storage_url=storage_url, study_name=_STUDY_NAME)
    for number in range(n_recorded):
        store.append(Trial(
            number=number,
            params={"0.ignore_zeros": bool(number % 2)},
            # Descending cost, so the LAST recorded trial is the winner and a
            # test can tell a genuine winner apart from "whatever came first".
            score=1.0 - 0.1 * number,
            terms={"Count": 1.0 - 0.1 * number},
            n_images=len(_PLATE_NAMES),
        ))

    io.tune_cache_run_marker_path(out).write_text(json.dumps({
        "version": 2,
        "study_name": _STUDY_NAME,
        "storage_url": storage_url,
        "images_dir": str(images_dir),
        "nrows": None,
        "ncols": None,
        "strategy": "tpe",
        "n_trials": _BUDGET,
        "is_multi_objective": False,
        "slurm": True,
        "start_time": "2026-08-19T00:00:00+00:00",
    }, indent=2))
    return out


@pytest.fixture
def finished_distributed_study(tmp_path):
    """A study whose fleet drained the budget: terminal, ready to finalize."""
    return _build_study(tmp_path, n_recorded=_BUDGET)


@pytest.fixture
def running_distributed_study(tmp_path):
    """A study still short of its budget — workers are presumably still alive."""
    return _build_study(tmp_path, n_recorded=_BUDGET - 2)


# --- what the SLURM branch skipped -------------------------------------------


def test_finalize_writes_what_the_slurm_branch_skipped(finished_distributed_study):
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    assert not io.best_params_path(out).is_file(), "fixture precondition"
    assert not io.trials_parquet_path(out).is_file(), "fixture precondition"

    result = finalize_distributed_study(out)

    assert io.best_params_path(out).is_file()
    assert io.trials_parquet_path(out).is_file()
    assert io.param_importance_path(out).is_file()
    assert io.best_pipeline_path(out).is_file()
    assert result.best_params_written is True
    assert result.n_trials == _BUDGET


def test_best_params_names_the_actual_winner(finished_distributed_study):
    """Not just "a file exists": the lowest-cost trial is the one recorded.

    ``best_params.json`` is what ``prepare_best_from_run`` exports, so naming the
    wrong trial here ships the wrong pipeline with no other symptom.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    result = finalize_distributed_study(out)

    payload = json.loads(io.best_params_path(out).read_text())
    # Costs descend with trial number, so the last recorded trial wins.
    assert payload["trial_number"] == _BUDGET - 1
    assert payload["selection"] == "single_best"
    assert payload["params"] == {"0.ignore_zeros": bool((_BUDGET - 1) % 2)}
    assert result.winner_trial_number == _BUDGET - 1


def test_the_generalization_report_is_written_from_a_rescan(
    finished_distributed_study,
):
    """Step 4 needs loaded plates the submitting process no longer holds.

    Dropping it would leave the held-out ``gap`` permanently null for every
    distributed study — the exact signal that catches an arm that won by
    overfitting. So the plates are re-loaded from the marker's ``images_dir``.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    result = finalize_distributed_study(out)

    assert result.generalization_written is True
    payload = json.loads(io.generalization_path(out).read_text())
    for key in ("kind", "calibration_score", "heldout_score", "gap", "flagged"):
        assert key in payload


def test_a_missing_image_directory_is_reported_not_swallowed(
    finished_distributed_study,
):
    """The plates can be gone by the time someone finalizes. Say so."""
    import shutil

    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    shutil.rmtree(marker["images_dir"])

    result = finalize_distributed_study(out)

    assert result.generalization_written is False
    assert not io.generalization_path(out).is_file()
    assert any("generalization" in w for w in result.warnings)
    # Everything that does NOT need the plates still lands.
    assert result.best_params_written is True


# --- the ordering ------------------------------------------------------------


def test_best_params_is_written_last(finished_distributed_study, monkeypatch):
    """It is the completion marker; writing it early breaks export gating."""
    from phenotypic.tune._tune_cli import _finalize

    order: list[str] = []
    for name in ("_finalize_outputs", "_finalize_pareto_outputs",
                 "_finalize_best_params", "_finalize_generalization"):
        real = getattr(_finalize, name)
        monkeypatch.setattr(
            _finalize, name,
            lambda *a, _n=name, _r=real, **k: (order.append(_n), _r(*a, **k))[1],
        )
    _finalize.finalize_distributed_study(finished_distributed_study)

    # len() FIRST: ``order.index(...) == 2`` passes even if step 4 is silently
    # dropped, and it passes on a two-element list that never ran generalization.
    assert len(order) == 4
    assert order == [
        "_finalize_outputs",
        "_finalize_pareto_outputs",
        "_finalize_best_params",
        "_finalize_generalization",
    ]


# --- the two interruption hazards --------------------------------------------


def test_refuses_a_running_study(running_distributed_study):
    """Two finalizes on a live study each publish a winner the other overwrites."""
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    with pytest.raises(RuntimeError, match="not finished|still running"):
        finalize_distributed_study(running_distributed_study)

    assert not io.best_params_path(running_distributed_study).is_file()
    assert not io.tune_finalize_marker_path(running_distributed_study).exists()


def test_force_finalizes_a_study_that_cannot_be_shown_terminal(
    running_distributed_study,
):
    """The escape hatch for a fleet the operator knows is dead."""
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    result = finalize_distributed_study(running_distributed_study, force=True)

    assert result.best_params_written is True
    assert result.n_trials == _BUDGET - 2


def test_interrupted_finalize_leaves_a_marker(
    finished_distributed_study, monkeypatch
):
    """A kill inside step 2 leaves best_pipeline.json mislabelled; refuse after."""
    from phenotypic.tune._tune_cli import _finalize

    monkeypatch.setattr(_finalize, "_finalize_pareto_outputs",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        _finalize.finalize_distributed_study(finished_distributed_study)

    assert io.tune_finalize_marker_path(finished_distributed_study).exists()

    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="finalize_incomplete|incomplete"):
        _finalize.finalize_distributed_study(finished_distributed_study)


def test_force_clears_an_interrupted_finalize(
    finished_distributed_study, monkeypatch
):
    from phenotypic.tune._tune_cli import _finalize

    monkeypatch.setattr(_finalize, "_finalize_pareto_outputs",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        _finalize.finalize_distributed_study(finished_distributed_study)
    monkeypatch.undo()

    _finalize.finalize_distributed_study(finished_distributed_study, force=True)

    assert not io.tune_finalize_marker_path(finished_distributed_study).exists()
    assert io.best_params_path(finished_distributed_study).is_file()


def test_finalize_is_rerunnable(finished_distributed_study):
    """Every step overwrites its own output, so a second pass reproduces it."""
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    first = finalize_distributed_study(finished_distributed_study)
    payload_first = io.best_params_path(finished_distributed_study).read_text()

    second = finalize_distributed_study(finished_distributed_study)  # must not raise

    assert second.winner_trial_number == first.winner_trial_number
    assert io.best_params_path(finished_distributed_study).read_text() == payload_first
    assert not io.tune_finalize_marker_path(finished_distributed_study).exists()


# --- refusals that are not about interruption --------------------------------


def test_a_directory_with_no_run_marker_is_refused(tmp_path):
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    with pytest.raises(FileNotFoundError, match="run.json"):
        finalize_distributed_study(tmp_path)


def test_a_winnerless_study_reports_instead_of_leaving_a_silent_hole(tmp_path):
    """``_finalize_best_params`` no-ops on a null winner; that must not be silent.

    Left unreported it resurfaces much later as a ``FileNotFoundError`` from
    ``prepare_best_from_run``, pointing at the export rather than at the study
    that never produced a trial.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = _build_study(tmp_path, n_recorded=0)
    marker_path = io.tune_cache_run_marker_path(out)
    marker = json.loads(marker_path.read_text())
    marker["n_trials"] = 0  # budget drained trivially: nothing was ever asked for
    marker_path.write_text(json.dumps(marker))

    result = finalize_distributed_study(out, force=True)

    assert result.winner_trial_number is None
    assert result.best_params_written is False
    assert any("no successful trial" in w for w in result.warnings)
    assert not io.best_params_path(out).is_file()

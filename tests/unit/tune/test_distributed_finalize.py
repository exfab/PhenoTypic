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

#: The recorded trial budget. The terminal gate compares the study's
#: budget-consuming (COMPLETE + PRUNED) trial count against it, so "finished"
#: and "running" differ only in how many real trials are written.
_BUDGET = 4

#: Every study built here also carries ONE orphaned ``RUNNING`` trial — the
#: state a worker leaves behind when SLURM kills it at the walltime, and the
#: state no fixture could previously produce. It is not optional scenery: an
#: un-told trial is not ``failed``, so a finalize that ranks or counts the raw
#: trial list treats it as a completed trial and can publish it as the winner.
#: Keeping it in the SHARED builder means every test below runs against a study
#: shaped like a real one, instead of a study the bug cannot reach.
_N_ORPHANS = 1

#: The orphan is given the best score in the study, stamped into its
#: ``user_attrs``. That is a real state, not a contrivance: the strategy stamps
#: the score sidecar and *then* calls ``study.tell``, so a worker killed between
#: those two DB round-trips (a window the transient-DB retry widens) leaves
#: exactly this — a ``RUNNING`` trial carrying a finished-looking cost. It is
#: also the state that separates the two independent guards. Give the orphan no
#: cost at all and the ``inf`` fallback alone hides it, so a test could not tell
#: whether the terminal filter still worked; give it the winning cost and only
#: the terminal filter keeps it out of ``best_params.json``.
_ORPHAN_SCORE = 0.0

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


def _leave_orphaned_running_trial(store, *, score):
    """Ask for a trial, optionally stamp its result, and never tell the study.

    Reproduces a worker killed at its SLURM walltime. The stamp order mirrors
    ``OptunaStrategy.register_result``, which sets the user attrs and only then
    calls ``study.tell`` — so a kill in between is what leaves a ``RUNNING``
    trial holding a real-looking cost.
    """
    trial = store._study.ask()
    if score is None:
        return trial.number
    from phenotypic.tune.strategy._optuna_support import set_trial_user_attrs

    class _Result:
        pass

    _Result.score = score
    _Result.terms = {"Count": score}
    _Result.n_images = len(_PLATE_NAMES)
    _Result.objectives = None
    _Result.gap = None
    _Result.suspicious = False
    set_trial_user_attrs(
        trial, params={"0.ignore_zeros": True}, result=_Result()
    )
    return trial.number


def _build_study(
    tmp_path,
    *,
    n_recorded: int,
    orphans: int = _N_ORPHANS,
    orphan_score=_ORPHAN_SCORE,
):
    """A distributed study directory holding ``n_recorded`` trials.

    Mirrors exactly what ``run_tuning --slurm`` leaves behind: the resolved spec
    echo and the ``run.json`` marker (both written above the submission branch),
    a shared study the fleet appended to, and NOTHING else.

    Args:
        tmp_path: The test's temp directory.
        n_recorded: How many real, COMPLETE trials the fleet wrote.
        orphans: How many ``RUNNING`` trials to leave un-told (see
            :data:`_N_ORPHANS`). ``study.ask()`` without a matching ``tell`` is
            exactly what a killed worker leaves in the study.
        orphan_score: The cost stamped on each orphan, or ``None`` to stamp
            none at all (a worker killed mid-evaluation). See
            :data:`_ORPHAN_SCORE`.
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

    for _ in range(orphans):
        # A worker asked for a trial, stamped its result, and never came back:
        # RUNNING forever, carrying the best cost in the study (_ORPHAN_SCORE).
        _leave_orphaned_running_trial(store, score=orphan_score)

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
    assert result.n_trials == _BUDGET + _N_ORPHANS


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
    assert result.n_trials == _BUDGET - 2 + _N_ORPHANS


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


def _published_bytes(out) -> dict[str, bytes]:
    """Every artifact finalize publishes, by relative path → bytes.

    Deliberately not just ``best_params.json``: idempotence is a claim about
    the WHOLE output, and each of the four steps writes different files.
    ``.pht-tune-cache/`` is excluded — the study DB and its SQLite WAL are
    engine state, not published artifacts, and re-reading a study legitimately
    touches them.
    """
    roots = [io.deliverables_dir(out), io.trials_parquet_path(out)]
    snapshot: dict[str, bytes] = {}
    for root in roots:
        if root.is_file():
            snapshot[root.name] = root.read_bytes()
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                snapshot[str(path.relative_to(out))] = path.read_bytes()
    return snapshot


def test_finalize_is_rerunnable(finished_distributed_study):
    """Idempotent, not merely non-crashing: run two must REPRODUCE run one.

    "Does not raise" is the weak version of this claim and would pass against a
    finalize that appended to ``trials.parquet``, stamped a fresh timestamp into
    the completion marker, or re-derived a different winner on the second pass.
    So every published byte is compared, and the artifact set is compared too —
    a second run that quietly stopped writing one file would otherwise slip
    through a value-only comparison.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    first = finalize_distributed_study(out)
    before = _published_bytes(out)
    # Anti-vacuity: an empty snapshot would make every assertion below trivially
    # true, which is exactly the shape of a test that cannot fail.
    assert "best_params.json" in " ".join(before)
    assert len(before) >= 4, before.keys()

    second = finalize_distributed_study(out)  # must not raise
    after = _published_bytes(out)

    assert set(after) == set(before), "the second run published a different file set"
    for name in sorted(before):
        assert after[name] == before[name], f"{name} changed on the second finalize"
    assert second == first, "the reported result changed on the second finalize"
    assert not io.tune_finalize_marker_path(out).exists()


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


# --- the orphan a killed worker leaves behind --------------------------------


def test_the_orphaned_running_trial_never_becomes_the_winner(
    finished_distributed_study,
):
    """The published artifact, not just the store: ``best_params.json``.

    An un-told ``RUNNING`` trial is not ``failed`` and carries no cost. Ranked
    alongside real trials it won outright — ``0.0`` is the best possible cost
    under the minimize convention — and it won with ``params={}``, which makes
    ``prepare_best_from_run`` export the UNTUNED base pipeline while reporting a
    perfect score. No error, no warning, ``best_params_written=True``.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    result = finalize_distributed_study(out)

    payload = json.loads(io.best_params_path(out).read_text())
    assert payload["trial_number"] == _BUDGET - 1
    # The two signatures of the phantom, asserted separately: it wins by
    # scoring the impossible best, and it wins carrying nothing to export.
    assert payload["score"] == pytest.approx(1.0 - 0.1 * (_BUDGET - 1))
    assert payload["params"] == {"0.ignore_zeros": bool((_BUDGET - 1) % 2)}
    assert result.winner_trial_number == _BUDGET - 1


def test_an_orphan_does_not_count_as_progress_toward_the_budget(tmp_path):
    """The gate must measure the unit the WORKERS stop on.

    A worker stops asking when ``COMPLETE + PRUNED >= n_trials``; failed and
    in-flight trials consume none of the budget. A gate comparing the raw trial
    count against that budget opens early by ``#failed + #in-flight`` — here the
    study looks full (4 rows against a budget of 4) while only 3 trials ran.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = _build_study(tmp_path, n_recorded=_BUDGET - 1, orphans=1)

    with pytest.raises(RuntimeError, match="not finished|still running"):
        finalize_distributed_study(out)

    assert not io.best_params_path(out).is_file()
    assert not io.tune_finalize_marker_path(out).exists()


def test_the_in_flight_trial_is_reported_rather_than_hidden(
    finished_distributed_study,
):
    """Excluded from the winner, but never silently dropped.

    This store cannot tell an orphan from a worker still evaluating, so the
    count is surfaced instead of being decided for the reader — and
    ``n_trials`` stays the honest total so it matches the study itself.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    result = finalize_distributed_study(finished_distributed_study)

    assert result.n_trials == _BUDGET + _N_ORPHANS
    assert any("in flight" in w for w in result.warnings), result.warnings


def test_trials_parquet_publishes_only_evaluated_trials(
    finished_distributed_study,
):
    """The parquet carries no state column, so an in-flight row is undetectable.

    Exported, it reads downstream as a real result with no params and no cost.
    """
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    finalize_distributed_study(out)

    exported = pd.read_parquet(io.trials_parquet_path(out))
    assert len(exported) == _BUDGET
    assert exported["score"].notna().all()
    assert not exported["score"].isin([float("inf")]).any()


def test_a_pruned_trial_with_no_recoverable_cost_does_not_break_finalize(
    tmp_path,
):
    """A study written before the cost sidecar existed must still finalize.

    ``study.tell(trial, state=PRUNED)`` stores no value, so such a row's cost is
    genuinely unknown — the store reads it back as ``inf`` rather than as the
    best possible ``0.0``. That ``inf`` must not reach the importance model,
    which rejects a non-finite target outright ("Input y contains infinity") and
    would take the whole finalize down with it. The row still consumed a slot of
    the budget, so the gate must count it.
    """
    import optuna

    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study
    from phenotypic.tune.strategy._optuna_support import set_trial_user_attrs

    out = _build_study(tmp_path, n_recorded=_BUDGET - 1)
    marker = json.loads(io.tune_cache_run_marker_path(out).read_text())
    store = OptunaStudyStore(
        storage_url=marker["storage_url"], study_name=_STUDY_NAME
    )
    trial = store._study.ask()

    class _Result:  # a pre-sidecar result object: no `score` to stamp
        terms = {"Count": 0.9}
        n_images = len(_PLATE_NAMES)
        objectives = None
        gap = None
        suspicious = False

    set_trial_user_attrs(trial, params={"0.ignore_zeros": True}, result=_Result())
    store._study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    result = finalize_distributed_study(out)  # gate opens: it consumed a slot

    assert io.param_importance_path(out).is_file()
    payload = json.loads(io.best_params_path(out).read_text())
    assert payload["trial_number"] == _BUDGET - 2
    assert result.winner_trial_number == _BUDGET - 2

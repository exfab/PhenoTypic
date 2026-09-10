"""Publication and manual recovery contracts for distributed Tune studies."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phenotypic._cli._cli_slurm_lifecycle import CancellationResult
from phenotypic.sdk_ import _io_constants as io


class _Store:
    def __init__(self, *, completed: int, winner: object | None) -> None:
        self.trials = [winner] if winner is not None else []
        self._completed = completed
        self._winner = winner

    def terminal_trials(self):
        return list(self.trials)

    def completed_count(self) -> int:
        return self._completed

    def pareto_front(self):
        return []

    def best(self):
        return self._winner


def _install_publication_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: object = 1,
    completed: int = 1,
    winner: object | None = None,
) -> object:
    from phenotypic.tune._tune_cli import _finalize

    resolved_winner = winner or SimpleNamespace(number=3, params={"x": 2})
    store = _Store(completed=completed, winner=resolved_winner)
    monkeypatch.setattr(
        _finalize,
        "_read_run_marker",
        lambda _output: {"n_trials": budget, "study_name": _finalize._STUDY_NAME},
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: object())
    monkeypatch.setattr(
        _finalize,
        "_open_finished_store",
        lambda _spec, _output, _marker: store,
    )
    monkeypatch.setattr(_finalize, "_pipeline_for_trial", lambda _spec, trial: trial)
    return resolved_winner


def _real_optuna_study(
    tmp_path: Path,
    *,
    strategy_trials: int = 3,
    budget_trials: int | None = None,
    completed: int = 3,
) -> Path:
    """Build a real journal study and real data-poor calibration input."""
    from phenotypic import GridImage, ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Budget, Categorical, Evaluator, Knob, SearchSpace
    from phenotypic.tune.score import ReferenceFreeScorer
    from phenotypic.tune.strategy import OptunaConfig
    from phenotypic.tune._spec import TuningSpec
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study._storage import journal_url_for_path
    from phenotypic.tune._study_store import Trial
    from phenotypic.tune._tune_cli import _run

    output = tmp_path / "out"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    pixels = np.zeros((12, 12, 3), dtype=np.uint8)
    pixels[3:9, 3:9] = 255
    GridImage(
        pixels,
        name="plate",
        nrows=2,
        ncols=3,
        bit_depth=8,
    ).rgb.imsave(images_dir / "plate.png")

    storage_url = journal_url_for_path(
        io.tune_cache_journal_path(output.absolute())
    )
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    key="0.ignore_zeros",
                    domain=Categorical(choices=(True, False)),
                ),
            )
        ),
        scorer=ReferenceFreeScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(
            n_trials=strategy_trials,
            storage_url=storage_url,
        ),
        budget=Budget(n_trials=budget_trials),
    )
    io.deliverables_dir(output).mkdir(parents=True)
    io.tuning_spec_path(output).write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )
    _run._write_run_marker(
        output,
        spec,
        storage_url=storage_url,
        images_dir=images_dir,
        slurm=True,
        nrows=2,
        ncols=3,
    )
    store = OptunaStudyStore(
        storage_url=storage_url,
        study_name=_run._STUDY_NAME,
    )
    for number in range(completed):
        store.append(
            Trial(
                number=number,
                params={"0.ignore_zeros": bool(number % 2)},
                score=float(completed - number),
                terms={"cost": float(completed - number)},
                n_images=1,
            )
        )
    return output


@pytest.mark.parametrize("marker_name", [None, "tune", "different-study"])
@pytest.mark.parametrize("backend", ["journal", "sqlite"])
def test_publication_rejects_missing_legacy_or_mismatched_study_identity_before_open(
    tmp_path: Path, marker_name: str | None, backend: str
) -> None:
    """A run marker cannot redirect finalization to another Optuna study."""
    from phenotypic.tune._tune_cli import _finalize

    output = _real_optuna_study(tmp_path)
    marker_path = io.tune_cache_run_marker_path(output)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if backend == "sqlite":
        marker["storage_url"] = f"sqlite:///{tmp_path / 'unopened.db'}"
    if marker_name is None:
        marker.pop("study_name")
    else:
        marker["study_name"] = marker_name
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    before = _tree_file_bytes(output)

    with pytest.raises(RuntimeError, match="study.*identity|study_name"):
        _finalize._publish_distributed_study(output)

    assert _tree_file_bytes(output) == before
    assert not (tmp_path / "unopened.db").exists()


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("marker_name", [None, "tune", "different-study"])
def test_manual_finalize_validates_study_identity_before_lifecycle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
    marker_name: str | None,
) -> None:
    """Manual recovery rejects marker identity before locks or cancellation."""
    from phenotypic._cli._cli_slurm_lifecycle import lifecycle_lock_path
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    marker_path = io.tune_cache_run_marker_path(output)
    marker_path.parent.mkdir(parents=True)
    marker: dict[str, object] = {"n_trials": 1, "generation": "owned"}
    if marker_name is not None:
        marker["study_name"] = marker_name
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    lifecycle_calls: list[str] = []
    cancel_calls: list[str] = []

    def _load_lifecycle(_output):
        lifecycle_calls.append("load")
        return {"generation": "owned", "active": True}

    def _cancel(_output, generation):
        cancel_calls.append(generation)
        return CancellationResult(("1",), (), True)

    monkeypatch.setattr(_finalize, "load_slurm_lifecycle", _load_lifecycle)
    monkeypatch.setattr(_finalize, "cancel_generation", _cancel)
    before = _tree_file_bytes(output)

    with pytest.raises(RuntimeError, match="study.*identity|study_name"):
        _finalize.finalize_distributed_study(output, force=force)

    assert _tree_file_bytes(output) == before
    assert lifecycle_calls == []
    assert cancel_calls == []
    assert not lifecycle_lock_path(output).exists()


@pytest.mark.parametrize("budget", [None, True, 0, "4"])
def test_publication_requires_a_positive_recorded_terminal_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: object
) -> None:
    """Guessing a missing budget can publish while workers are still running."""
    from phenotypic.tune._tune_cli import _finalize

    _install_publication_inputs(monkeypatch, budget=budget)

    with pytest.raises(RuntimeError, match="terminal trial budget"):
        _finalize._publish_distributed_study(tmp_path)


def test_publication_refuses_an_incomplete_terminal_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUNNING/failed rows cannot stand in for completed-or-pruned work."""
    from phenotypic.tune._tune_cli import _finalize

    _install_publication_inputs(monkeypatch, budget=4, completed=3)

    with pytest.raises(RuntimeError, match="3 of 4"):
        _finalize._publish_distributed_study(tmp_path)
    assert not io.best_params_path(tmp_path).exists()


def test_engine_budget_caps_strategy_budget_for_real_finalization(
    tmp_path: Path,
) -> None:
    """Optuna(10) + engine Budget(3) is terminal after three real trials."""
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    output = _real_optuna_study(
        tmp_path,
        strategy_trials=10,
        budget_trials=3,
        completed=3,
    )

    result = finalize_distributed_study(output)

    marker = json.loads(io.tune_cache_run_marker_path(output).read_text())
    assert marker["n_trials"] == 3
    assert result.n_trials == 3
    assert result.best_params_written is True
    assert result.generalization_written is True
    generalization = json.loads(io.generalization_path(output).read_text())
    assert generalization["estimate"] == "calibration_stability"


def test_publication_refuses_a_study_without_a_valid_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drained budget containing only failures must exit nonzero."""
    from phenotypic.tune._tune_cli import _finalize

    _install_publication_inputs(
        monkeypatch, budget=2, completed=2, winner=SimpleNamespace()
    )
    monkeypatch.setattr(
        _finalize, "_headline_winner", lambda _store, **_kwargs: None
    )

    with pytest.raises(RuntimeError, match="valid winner"):
        _finalize._publish_distributed_study(tmp_path)
    assert not io.best_params_path(tmp_path).exists()


def test_publication_refuses_all_nonfinite_multiobjective_history_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty finite front must not fall back to a scalar projected winner."""
    from phenotypic.tune._study_store import JournalStudyStore, Trial
    from phenotypic.tune._tune_cli import _finalize
    from tests.unit.tune.test_run_tuning_pareto import _multi_objective_spec

    output = tmp_path / "out"
    output.mkdir()
    (output / "incumbent.txt").write_bytes(b"unchanged\n")
    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={"0.sigma": 1.0},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": float("nan"), "s1": 0.0},
            ),
            Trial(
                number=1,
                params={"0.sigma": 2.0},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.0, "s1": float("inf")},
            ),
        ]
    )
    spec = _multi_objective_spec(tmp_path)
    monkeypatch.setattr(
        _finalize, "_read_run_marker", lambda _output: {"n_trials": 2, "study_name": _finalize._STUDY_NAME}
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: spec)
    monkeypatch.setattr(
        _finalize,
        "_open_finished_store",
        lambda _spec, _output, _marker: store,
    )
    monkeypatch.setattr(
        _finalize,
        "_finalize_generalization_from_disk",
        lambda *_args, **_kwargs: False,
    )
    before = _tree_file_bytes(output)

    failure = None
    try:
        _finalize._publish_distributed_study(output)
    except RuntimeError as error:
        failure = error

    assert _tree_file_bytes(output) == before
    assert failure is not None
    assert "valid winner" in str(failure)


def test_publication_refuses_all_partial_objective_history_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scorer-required missing sibling must fail before any durable write."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss
    from phenotypic.tune._study_store import JournalStudyStore, Trial
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    output.mkdir()
    (output / "incumbent.txt").write_bytes(b"unchanged\n")
    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={"0.sigma": 1.0},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": 0.1},
            ),
            Trial(
                number=1,
                params={"0.sigma": 2.0},
                score=0.2,
                terms={},
                n_images=1,
                objectives={"s0": 0.2},
            ),
        ]
    )
    spec = SimpleNamespace(
        scorer=SimpleNamespace(
            multi_objective=True, objective_names=lambda: ["s0", "s1"]
        ),
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0)]),
    )
    monkeypatch.setattr(
        _finalize, "_read_run_marker", lambda _output: {"n_trials": 2, "study_name": _finalize._STUDY_NAME}
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: spec)
    monkeypatch.setattr(
        _finalize,
        "_open_finished_store",
        lambda _spec, _output, _marker: store,
    )
    monkeypatch.setattr(
        _finalize,
        "_finalize_generalization_from_disk",
        lambda *_args, **_kwargs: False,
    )
    before = _tree_file_bytes(output)

    failure = None
    try:
        _finalize._publish_distributed_study(output)
    except RuntimeError as error:
        failure = error

    assert _tree_file_bytes(output) == before
    assert failure is not None
    assert "valid winner" in str(failure)


def test_publication_rejects_unsafe_scorer_axis_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distributed publication cannot turn a scorer axis into a path component."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss
    from phenotypic.tune._study_store import JournalStudyStore, Trial
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    output.mkdir()
    (output / "incumbent.txt").write_bytes(b"unchanged\n")
    store = JournalStudyStore([
        Trial(number=0, params={}, score=0.1, terms={}, n_images=1,
              objectives={"safe": 0.1, "../escape": 0.2})
    ])
    spec = SimpleNamespace(
        scorer=SimpleNamespace(
            multi_objective=True, objective_names=lambda: ["safe", "../escape"]
        ),
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0)]),
    )
    monkeypatch.setattr(
        _finalize, "_read_run_marker",
        lambda _output: {"n_trials": 1, "study_name": _finalize._STUDY_NAME},
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: spec)
    monkeypatch.setattr(
        _finalize, "_open_finished_store", lambda _spec, _output, _marker: store
    )
    before = _tree_file_bytes(output)

    with pytest.raises(ValueError, match="safe filename"):
        _finalize._publish_distributed_study(output)

    assert _tree_file_bytes(output) == before


def test_publication_rejects_duplicate_scorer_axes_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed publication must reject a repeated authoritative axis."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss
    from phenotypic.tune._study_store import JournalStudyStore, Trial
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    output.mkdir()
    (output / "incumbent.txt").write_bytes(b"unchanged\n")
    store = JournalStudyStore(
        [
            Trial(
                number=0,
                params={"0.sigma": 1.0},
                score=0.1,
                terms={},
                n_images=1,
                objectives={"s0": 0.1},
            )
        ]
    )
    spec = SimpleNamespace(
        scorer=SimpleNamespace(
            multi_objective=True, objective_names=lambda: ["s0", "s0"]
        ),
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0)]),
    )
    monkeypatch.setattr(
        _finalize, "_read_run_marker", lambda _output: {"n_trials": 1, "study_name": _finalize._STUDY_NAME}
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: spec)
    monkeypatch.setattr(
        _finalize,
        "_open_finished_store",
        lambda _spec, _output, _marker: store,
    )
    monkeypatch.setattr(
        _finalize,
        "_finalize_generalization_from_disk",
        lambda *_args, **_kwargs: False,
    )
    before = _tree_file_bytes(output)

    with pytest.raises(ValueError, match="unique"):
        _finalize._publish_distributed_study(output)

    assert _tree_file_bytes(output) == before


def test_publication_rejects_casefold_scorer_axes_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distributed artifacts cannot alias by Unicode casefold on Windows."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss
    from phenotypic.tune._study_store import JournalStudyStore, Trial
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "casefold-out"
    output.mkdir()
    (output / "incumbent.txt").write_bytes(b"unchanged\n")
    store = JournalStudyStore(
        [Trial(
            number=0, params={"0.sigma": 1.0}, score=0.1, terms={}, n_images=1,
            objectives={"Dice": 0.1, "dice": 0.2},
        )]
    )
    spec = SimpleNamespace(
        scorer=SimpleNamespace(
            multi_objective=True, objective_names=lambda: ["Dice", "dice"]
        ),
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0)]),
    )
    monkeypatch.setattr(
        _finalize, "_read_run_marker",
        lambda _output: {"n_trials": 1, "study_name": _finalize._STUDY_NAME},
    )
    monkeypatch.setattr(_finalize, "_read_resolved_spec", lambda _output: spec)
    monkeypatch.setattr(
        _finalize, "_open_finished_store", lambda _spec, _output, _marker: store
    )
    monkeypatch.setattr(
        _finalize, "_finalize_generalization_from_disk",
        lambda *_args, **_kwargs: False,
    )

    before = _tree_file_bytes(output)

    with pytest.raises(ValueError, match="case-insensitive|casefold|unique"):
        _finalize._publish_distributed_study(output)

    assert _tree_file_bytes(output) == before


def test_read_only_publication_does_not_create_a_missing_journal(
    tmp_path: Path,
) -> None:
    """Observation of a missing fleet store must leave it missing."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Budget, Evaluator, SearchSpace
    from phenotypic.tune.score import ReferenceFreeScorer
    from phenotypic.tune.strategy import OptunaConfig
    from phenotypic.tune._spec import TuningSpec
    from phenotypic.tune._study._storage import journal_url_for_path
    from phenotypic.tune._tune_cli._finalize import _open_finished_store

    journal = tmp_path / "missing.log"
    url = journal_url_for_path(journal.absolute())
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=()),
        scorer=ReferenceFreeScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=1, storage_url=url),
        budget=Budget(n_trials=1),
    )

    with pytest.raises(FileNotFoundError, match="missing.log"):
        _open_finished_store(
            spec, tmp_path, {"storage_url": url, "study_name": "tune_cost_v1"}
        )
    assert not journal.exists()


def test_read_only_publication_does_not_create_a_missing_sqlite_database(
    tmp_path: Path,
) -> None:
    """Constructing RDBStorage before the existence check creates the database."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Budget, Evaluator, SearchSpace
    from phenotypic.tune.score import ReferenceFreeScorer
    from phenotypic.tune.strategy import OptunaConfig
    from phenotypic.tune._spec import TuningSpec
    from phenotypic.tune._tune_cli._finalize import _open_finished_store

    database = tmp_path / "missing.db"
    url = f"sqlite:///{database}"
    spec = TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=()),
        scorer=ReferenceFreeScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=1, storage_url=url),
        budget=Budget(n_trials=1),
    )

    with pytest.raises(FileNotFoundError, match="missing.db"):
        _open_finished_store(
            spec, tmp_path, {"storage_url": url, "study_name": "tune_cost_v1"}
        )
    assert not database.exists()


def _tree_file_bytes(root: Path) -> dict[str, bytes]:
    """Return a stable snapshot of every regular file below ``root``."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_manual_marker(
    output: Path, *, generation: str | None = None
) -> None:
    """Write a current marker for public manual-finalization lifecycle tests."""
    from phenotypic.tune._tune_cli import _finalize

    marker: dict[str, object] = {
        "n_trials": 1,
        "study_name": _finalize._STUDY_NAME,
    }
    if generation is not None:
        marker["generation"] = generation
    marker_path = io.tune_cache_run_marker_path(output)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_publication_requires_best_params_to_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent no-op best-params writer cannot report successful finalization."""
    from phenotypic.tune._tune_cli import _finalize

    _install_publication_inputs(monkeypatch)
    monkeypatch.setattr(_finalize, "_finalize_outputs", lambda *_a: None)
    monkeypatch.setattr(_finalize, "_finalize_pareto_outputs", lambda *_a: None)
    monkeypatch.setattr(
        _finalize,
        "_finalize_generalization_from_disk",
        lambda *_a: False,
    )
    monkeypatch.setattr(_finalize, "_finalize_best_params", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="best_params"):
        _finalize._publish_distributed_study(tmp_path)


@pytest.mark.parametrize(
    "calibration_state",
    ["missing_marker_path", "missing_directory", "empty_load"],
)
def test_missing_calibration_inputs_fail_before_completion_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_state: str,
) -> None:
    """A finalizer cannot close over absent inputs or stale generalization."""
    from phenotypic.tune._tune_cli import _finalize

    _install_publication_inputs(monkeypatch)
    stale_generalization = io.generalization_path(tmp_path)
    stale_generalization.parent.mkdir(parents=True)
    stale_generalization.write_bytes(b"stale generalization\n")
    marker: dict[str, object] = {"n_trials": 1, "study_name": _finalize._STUDY_NAME}
    if calibration_state != "missing_marker_path":
        images_dir = tmp_path / "calibration"
        marker["images_dir"] = str(images_dir)
        if calibration_state == "empty_load":
            images_dir.mkdir()
            monkeypatch.setattr(_finalize, "_load_images", lambda *_a, **_k: [])
    monkeypatch.setattr(_finalize, "_read_run_marker", lambda _output: marker)
    monkeypatch.setattr(_finalize, "_finalize_outputs", lambda *_a: None)
    monkeypatch.setattr(_finalize, "_finalize_pareto_outputs", lambda *_a: None)
    best_writes: list[bool] = []
    monkeypatch.setattr(
        _finalize,
        "_finalize_best_params",
        lambda *_a, **_k: best_writes.append(True),
    )

    with pytest.raises(
        RuntimeError,
        match="calibration|images_dir|image directory|readable images",
    ):
        _finalize._publish_distributed_study(tmp_path)

    assert best_writes == []
    assert not io.best_params_path(tmp_path).exists()
    assert stale_generalization.read_bytes() == b"stale generalization\n"


def _published_bytes(output: Path) -> dict[str, bytes]:
    paths = [io.trials_parquet_path(output), *io.deliverables_dir(output).rglob("*")]
    return {
        str(path.relative_to(output)): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def test_successful_manual_refinalization_is_byte_identical(
    tmp_path: Path,
) -> None:
    """Real journal publication reproduces every artifact byte-for-byte."""
    from phenotypic.tune._tune_cli import _finalize

    output = _real_optuna_study(tmp_path)

    first = _finalize.finalize_distributed_study(output)
    before = _published_bytes(output)
    second = _finalize.finalize_distributed_study(output)
    after = _published_bytes(output)

    assert first == second
    assert first.best_params_written is True
    assert len(before) >= 4
    assert after == before


def test_normal_manual_finalize_refuses_an_active_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual recovery cannot race an automatic publisher or live fleet."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="active", mode="tune")
    _write_manual_marker(output, generation="active")
    monkeypatch.setattr(
        _finalize,
        "_publish_distributed_study",
        lambda _output: pytest.fail("publication ran for an active owner"),
    )

    with pytest.raises(RuntimeError, match="active.*generation"):
        _finalize.finalize_distributed_study(output)


def test_normal_manual_finalize_holds_lifecycle_lock_through_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Releasing after the owner check permits a new launch during writes."""
    from phenotypic._cli._cli_slurm_lifecycle import lifecycle_lock_path
    from phenotypic.sdk_._file_locking import ArtifactLockTimeout, exclusive_path_lock
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    _write_manual_marker(output)
    observed: list[str] = []

    def _publish(_output):
        def _contender() -> None:
            try:
                with exclusive_path_lock(lifecycle_lock_path(output), timeout=0.05):
                    observed.append("acquired")
            except ArtifactLockTimeout:
                observed.append("blocked")

        thread = threading.Thread(target=_contender)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        return object()

    monkeypatch.setattr(_finalize, "_publish_distributed_study", _publish)

    _finalize.finalize_distributed_study(output)

    assert observed == ["blocked"]


@pytest.mark.parametrize(
    "result",
    [
        CancellationResult(("1",), (), False),
        CancellationResult(("1",), ("unknown-token",), True),
    ],
)
def test_forced_finalize_requires_proven_scheduler_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: CancellationResult,
) -> None:
    """Force is a cancellation workflow, not permission to race workers."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="active", mode="tune")
    _write_manual_marker(output, generation="active")
    monkeypatch.setattr(_finalize, "cancel_generation", lambda *_a: result)
    monkeypatch.setattr(
        _finalize,
        "_publish_distributed_study",
        lambda _output: pytest.fail("publication ran without quiescence"),
    )

    with pytest.raises(RuntimeError, match="quiescent|unresolved"):
        _finalize.finalize_distributed_study(output, force=True)


def test_forced_finalize_refuses_a_successor_created_after_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A force command cannot publish over a generation that won the race."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        deactivate_generation,
        initialize_slurm_lifecycle,
        lifecycle_state_path,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="cancelled", mode="tune")
    _write_manual_marker(output, generation="cancelled")

    def _cancel(_output, generation):
        assert deactivate_generation(output, generation)
        initialize_slurm_lifecycle(output, generation="successor", mode="tune")
        return CancellationResult(("1",), (), True)

    monkeypatch.setattr(_finalize, "cancel_generation", _cancel)
    monkeypatch.setattr(
        _finalize,
        "_publish_distributed_study",
        lambda _output: pytest.fail("publication ran over a successor"),
    )

    with pytest.raises(RuntimeError, match="new.*generation|successor"):
        _finalize.finalize_distributed_study(output, force=True)

    payload = json.loads(lifecycle_state_path(output).read_text())
    assert payload["generation"] == "successor"
    assert payload["active"] is True


@pytest.mark.parametrize("authority", ["missing", "corrupt", "mismatched"])
def test_forced_finalize_requires_marker_generation_lifecycle_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    """A generated marker must never turn absent authority into force permission."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        CancellationResult,
        deactivate_generation,
        initialize_slurm_lifecycle,
        lifecycle_state_path,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = _real_optuna_study(tmp_path)
    marker_path = io.tune_cache_run_marker_path(output)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["generation"] = "owned"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    lifecycle_path = lifecycle_state_path(output)
    if authority == "corrupt":
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_bytes(b"{not-json\n")
    elif authority == "mismatched":
        initialize_slurm_lifecycle(
            output, generation="different", mode="tune"
        )

    def _quiesce_recorded_generation(
        target: Path, generation: str
    ) -> CancellationResult:
        assert deactivate_generation(target, generation)
        return CancellationResult(("1",), (), True)

    monkeypatch.setattr(
        _finalize,
        "cancel_generation",
        _quiesce_recorded_generation,
    )
    before = _tree_file_bytes(output)

    with pytest.raises(RuntimeError, match="lifecycle|generation|authority"):
        _finalize.finalize_distributed_study(output, force=True)

    assert _tree_file_bytes(output) == before


@pytest.mark.parametrize("authority", ["missing", "corrupt", "mismatched"])
def test_nonforced_finalize_rejects_bad_authority_before_creating_a_lock(
    tmp_path: Path,
    authority: str,
) -> None:
    """New-style authority rejection must not persist even a lifecycle lock file."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        lifecycle_state_path,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = _real_optuna_study(tmp_path)
    marker_path = io.tune_cache_run_marker_path(output)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["generation"] = "owned"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    lifecycle_path = lifecycle_state_path(output)
    if authority == "corrupt":
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_bytes(b"{not-json\n")
    elif authority == "mismatched":
        initialize_slurm_lifecycle(output, generation="different", mode="tune")
    before = _tree_file_bytes(output)

    with pytest.raises(RuntimeError, match="lifecycle|generation|authority"):
        _finalize.finalize_distributed_study(output)

    assert _tree_file_bytes(output) == before


def test_public_finalize_subcommand_is_not_normalized_as_a_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding finalize without normalization support routes it into run parsing."""
    from phenotypic.tune import __main__ as cli

    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        cli,
        "finalize_distributed_study",
        lambda output, *, force=False: calls.append((Path(output), force)),
    )

    assert cli._normalize_argv(["finalize", str(tmp_path)]) == [
        "finalize",
        str(tmp_path),
    ]
    cli.main(["finalize", str(tmp_path), "--force"])
    assert calls == [(tmp_path, True)]


def test_cli_module_docstring_lists_all_three_subcommands() -> None:
    from phenotypic.tune import __main__ as cli

    assert "Three subcommands:" in (cli.__doc__ or "")


def test_internal_finalizer_cli_dispatches_the_exact_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated batch script must have an executable module target."""
    from phenotypic.tune._tune_cli import _finalize

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        _finalize,
        "finalize_owned_generation",
        lambda output, generation: calls.append((Path(output), generation)),
    )

    _finalize.main(
        ["--output", str(tmp_path / "out"), "--generation", "generation-a"]
    )
    assert calls == [(tmp_path / "out", "generation-a")]


def test_owned_finalizer_rejects_marker_from_another_generation(
    tmp_path: Path,
) -> None:
    """An active finalizer cannot publish inputs recorded by an older launch."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    marker = io.tune_cache_run_marker_path(output)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"generation": "old", "n_trials": 1}), encoding="utf-8"
    )
    initialize_slurm_lifecycle(output, generation="new", mode="tune")

    with pytest.raises(RuntimeError, match="marker.*generation|generation.*marker"):
        _finalize.finalize_owned_generation(output, "new")


def test_run_marker_records_fixed_grid_for_fresh_process_generalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalizer must reload plates with the grid used by the workers."""
    from phenotypic.tune._tune_cli import _run

    monkeypatch.setattr(_run, "is_multi_objective", lambda _scorer: False)
    spec = SimpleNamespace(
        strategy=SimpleNamespace(kind="optuna", n_trials=7), scorer=object()
    )

    _run._write_run_marker(
        tmp_path,
        spec,
        storage_url=f"journal://{tmp_path}/journal.log",
        images_dir=tmp_path / "images",
        slurm=True,
        nrows=8,
        ncols=12,
    )

    marker = json.loads(io.tune_cache_run_marker_path(tmp_path).read_text())
    assert marker["nrows"] == 8
    assert marker["ncols"] == 12

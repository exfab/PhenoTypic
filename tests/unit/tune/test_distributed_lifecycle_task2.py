"""Lifecycle ownership for standalone distributed Tune finalization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phenotypic.sdk_ import _io_constants as io


def _executor(tmp_path: Path, *, generation: str, n_workers: int = 3):
    from phenotypic._execution._slurm import SlurmExecutor

    return SlurmExecutor(
        output_dir=tmp_path / "out",
        spec_path=tmp_path / "spec.json",
        images_dir=tmp_path / "images",
        split_path=tmp_path / "split.json",
        study_name="tune_cost_v1",
        n_workers=n_workers,
        slurm_args={
            "slurm_partition": "gpu",
            "slurm_mem": "12G",
            "slurm_cpus_per_task": 4,
            "slurm_array": "9-20",
        },
        storage_url=f"journal://{tmp_path}/journal.log",
        python_command=["/shared/venv/bin/python"],
        lifecycle_generation=generation,
    )


def test_tune_executor_submits_one_afterany_terminal_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping finalizer/generation kwargs recreates fire-and-forget output loss."""
    from phenotypic._execution import _slurm

    captured: dict[str, object] = {}

    def _dispatchers(**kwargs):
        captured["dispatcher"] = kwargs
        return []

    def _submit(**kwargs):
        captured["submit"] = kwargs
        return ["101", "102"], None

    monkeypatch.setattr(_slurm, "generate_dispatcher_chain", _dispatchers)
    monkeypatch.setattr(_slurm, "submit_drip_feed_start", _submit)

    executor = _executor(tmp_path, generation="generation-a")
    executor.run(lambda item: item, [0, 1, 2])

    submitted = captured["submit"]
    assert isinstance(submitted, dict)
    finalizer = submitted["finalizer_script"]
    assert isinstance(finalizer, Path) and finalizer.is_file()
    assert submitted["continuation_dependency_kind"] == "afterany"
    assert submitted["output_dir"] == tmp_path / "out"
    assert submitted["generation"] == "generation-a"

    dispatched = captured["dispatcher"]
    assert isinstance(dispatched, dict)
    assert dispatched["finalizer_script"] == finalizer
    assert dispatched["generation"] == "generation-a"
    assert dispatched["lifecycle_output_dir"] == tmp_path / "out"


def test_terminal_finalizer_reuses_worker_resources_without_array(
    tmp_path: Path,
) -> None:
    """An array directive would launch multiple publishers for one generation."""
    executor = _executor(tmp_path, generation="generation-a")

    worker = executor.generate_worker_array_script().read_text(encoding="utf-8")
    finalizer = executor.generate_finalizer_script().read_text(encoding="utf-8")

    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --mem=12G",
        "#SBATCH --cpus-per-task=4",
    ):
        assert directive in worker
        assert directive in finalizer
    assert "#SBATCH --array=0-2" in worker
    assert "#SBATCH --array" not in finalizer
    assert "/shared/venv/bin/python" in finalizer
    assert "phenotypic.tune._tune_cli._finalize" in finalizer
    assert "generation-a" in finalizer


def test_hostile_generation_scripts_share_one_contained_digest_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw generation text must never become a worker or dispatcher path."""
    from phenotypic._execution import _slurm
    from phenotypic.sdk_ import slurm_scripts_dir
    from phenotypic.sdk_.slurm import generate_dispatcher_chain

    generations = [
        str(tmp_path / "absolute-generation"),
        "..",
        "../escape",
        "a/b",
        "a/./b",
        "snow/雪-☃",
    ]
    submitted: list[tuple[Path, int | None]] = []

    def _submit(path: Path, *, array_index: int | None = None) -> str:
        submitted.append((path, array_index))
        return "job-1"

    monkeypatch.setattr(_slurm, "submit_script", _submit)
    for index, generation in enumerate(generations):
        case = tmp_path / f"case-{index}"
        executor = _executor(case, generation=generation)
        worker = executor.generate_worker_array_script()
        finalizer = executor.generate_finalizer_script()
        output = case / "out"
        script_root = slurm_scripts_dir(output)
        tune_root = script_root / "tune"
        dispatch_root = script_root / "dispatch"
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=[worker, worker.parent / "next.sh"],
            output_dir=output,
            slurm_args={},
            log_dir=output / "logs",
            generation=generation,
            lifecycle_output_dir=output,
        )

        assert len(dispatchers) == 1
        dispatcher = dispatchers[0]
        assert worker.resolve().is_relative_to(tune_root.resolve())
        assert finalizer.resolve().is_relative_to(tune_root.resolve())
        assert dispatcher.resolve().is_relative_to(dispatch_root.resolve())
        assert worker.parent.parent == tune_root
        assert dispatcher.parent.parent == dispatch_root
        assert finalizer.parent == worker.parent
        assert dispatcher.parent.name == worker.parent.name
        assert len(worker.parent.name) == 64
        assert set(worker.parent.name) <= set("0123456789abcdef")
        assert (
            _executor(case, generation=generation).generate_worker_array_script()
            == worker
        )

        assert executor.reenqueue_dead_worker(worker_index=1) == "job-1"
        assert submitted[-1] == (worker, 1)


def test_distinct_generation_strings_never_alias_script_directories(
    tmp_path: Path,
) -> None:
    """Lossy path normalization must not merge distinct lifecycle owners."""
    collision_pairs = [
        ("a/b", "a/./b"),
        ("a//b", "a/b"),
        ("../escape", "escape"),
        ("é", "e\u0301"),
    ]

    for index, (first, second) in enumerate(collision_pairs):
        case = tmp_path / f"collision-{index}"
        first_path = _executor(
            case, generation=first
        ).generate_worker_array_script()
        second_path = _executor(
            case, generation=second
        ).generate_worker_array_script()

        assert first_path.parent != second_path.parent


def test_stale_submitter_cannot_overwrite_successor_generation_scripts(
    tmp_path: Path,
) -> None:
    """A stale pre-submit writer must never touch its successor's script bytes."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        deactivate_generation,
        initialize_slurm_lifecycle,
    )

    output = tmp_path / "out"
    stale_generation = "same/./path"
    successor_generation = "same/path"
    initialize_slurm_lifecycle(output, generation=stale_generation, mode="tune")
    assert deactivate_generation(output, stale_generation)
    initialize_slurm_lifecycle(
        output, generation=successor_generation, mode="tune"
    )

    successor = _executor(
        tmp_path,
        generation=successor_generation,
        n_workers=4,
    )
    successor_worker = successor.generate_worker_array_script()
    successor_finalizer = successor.generate_finalizer_script()
    before = {
        successor_worker: successor_worker.read_bytes(),
        successor_finalizer: successor_finalizer.read_bytes(),
    }

    stale = _executor(
        tmp_path,
        generation=stale_generation,
        n_workers=2,
    )
    with pytest.raises(RuntimeError, match="inactive|superseded"):
        stale.run(lambda item: item, [0, 1])

    assert successor_finalizer.parent == successor_worker.parent
    assert {path: path.read_bytes() for path in before} == before


def test_owned_finalizer_closes_its_generation_while_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returning after publication without deactivation leaves the run active."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="owned", mode="tune")
    marker = object()
    monkeypatch.setattr(
        _finalize,
        "_publish_distributed_study",
        lambda _output, **_kwargs: marker,
    )

    assert _finalize.finalize_owned_generation(output, "owned") is marker
    state = load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == "owned"
    assert state["active"] is False


def test_stale_finalizer_never_mutates_a_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late finalizer must neither publish nor fail/close its successor."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        lifecycle_state_path,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="stale", mode="tune")
    assert deactivate_generation(output, "stale") is True
    initialize_slurm_lifecycle(output, generation="new", mode="tune")
    before = lifecycle_state_path(output).read_bytes()
    monkeypatch.setattr(
        _finalize,
        "_publish_distributed_study",
        lambda _output: pytest.fail("a stale finalizer reached publication"),
    )

    with pytest.raises(SlurmGenerationInactiveError):
        _finalize.finalize_owned_generation(output, "stale")

    assert lifecycle_state_path(output).read_bytes() == before


def test_owned_publication_failure_marks_only_that_generation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed automatic publisher must not leave its generation active."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )
    from phenotypic.tune._tune_cli import _finalize

    output = tmp_path / "out"
    initialize_slurm_lifecycle(output, generation="owned", mode="tune")

    def _fail(_output, **_kwargs):
        raise RuntimeError("publication broke")

    monkeypatch.setattr(_finalize, "_publish_distributed_study", _fail)

    with pytest.raises(RuntimeError, match="publication broke"):
        _finalize.finalize_owned_generation(output, "owned")

    state = load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == "owned"
    assert state["active"] is False
    assert state["terminal_status"] == "failed"
    assert "publication broke" in state["terminal_error"]


def test_submission_failure_terminalizes_initialized_tune_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initializing before sbatch must not strand an active owner on failure."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.tune._tune_cli import _run

    captured: dict[str, object] = {}

    class _FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, _work, _items):
            raise RuntimeError("sbatch unavailable")

    monkeypatch.setattr(_run, "SlurmExecutor", _FakeExecutor)
    monkeypatch.setattr(
        "phenotypic.tune._study._optuna_store.OptunaStudyStore",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(_run, "objective_directions", lambda _scorer: None)
    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        lambda *, for_slurm: (["/shared/venv/bin/python"], "venv"),
    )

    spec = SimpleNamespace(
        strategy=SimpleNamespace(n_trials=2), scorer=object()
    )
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="sbatch unavailable"):
        _run._submit_slurm_fleet(
            spec,
            output,
            storage_url=f"journal://{tmp_path}/journal.log",
            spec_path=tmp_path / "source-spec.json",
            images_dir=tmp_path / "images",
            split_path=tmp_path / "split.json",
            n_workers=2,
        )

    state = load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == captured["lifecycle_generation"]
    assert state["active"] is False
    assert state["terminal_status"] == "failed"


def test_executor_setup_failure_also_terminalizes_initialized_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every exception after lifecycle initialization must release ownership."""
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle
    from phenotypic.tune._tune_cli import _run

    class _BrokenExecutor:
        def __init__(self, **_kwargs):
            raise RuntimeError("executor setup failed")

    monkeypatch.setattr(_run, "SlurmExecutor", _BrokenExecutor)
    monkeypatch.setattr(
        "phenotypic.tune._study._optuna_store.OptunaStudyStore",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(_run, "objective_directions", lambda _scorer: None)
    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        lambda *, for_slurm: (["/shared/venv/bin/python"], "venv"),
    )
    monkeypatch.setattr(lifecycle, "new_slurm_generation", lambda: "owned")

    spec = SimpleNamespace(strategy=SimpleNamespace(n_trials=2), scorer=object())
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="executor setup failed"):
        _run._submit_slurm_fleet(
            spec,
            output,
            storage_url=f"journal://{tmp_path}/journal.log",
            spec_path=tmp_path / "source-spec.json",
            images_dir=tmp_path / "images",
            split_path=tmp_path / "split.json",
            n_workers=2,
        )

    state = lifecycle.load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == "owned"
    assert state["active"] is False
    assert state["terminal_status"] == "failed"


def test_preclaimed_helper_validation_failure_terminalizes_owned_generation(
    tmp_path: Path,
) -> None:
    """Validation after a direct caller's claim must release that exact owner."""
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle
    from phenotypic.tune._tune_cli import _run

    output = tmp_path / "out"
    lifecycle.initialize_slurm_lifecycle(
        output, generation="owned", mode="tune"
    )

    with pytest.raises(ValueError, match="n-workers"):
        _run._submit_slurm_fleet(
            SimpleNamespace(strategy=SimpleNamespace(n_trials=2)),
            output,
            storage_url=f"journal://{tmp_path}/journal.log",
            spec_path=tmp_path / "spec.json",
            images_dir=tmp_path / "images",
            split_path=tmp_path / "split.json",
            n_workers=0,
            generation="owned",
            shared_state_prepared=True,
        )

    state = lifecycle.load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == "owned"
    assert state["active"] is False
    assert state["terminal_status"] == "failed"


def test_run_tuning_interpreter_failure_terminalizes_claimed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public flow owns cleanup when failure follows shared-state setup."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.tune._study._storage import journal_url_for_path
    from phenotypic.tune._tune_cli import _run
    from tests.unit.tune.test_run_tuning_slurm import _grid_input_spec

    def _missing_interpreter(*, for_slurm: bool):
        assert for_slurm
        raise RuntimeError("interpreter unavailable")

    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        _missing_interpreter,
    )
    monkeypatch.setattr(
        _run,
        "_precreate_shared_optuna_study",
        lambda *_args, **_kwargs: None,
    )
    spec = _grid_input_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="interpreter unavailable"):
        _run.run_tuning(
            spec,
            [],
            output,
            strategy="tpe",
            n_trials=2,
            slurm=True,
            spec_path=spec_path,
            images_dir=tmp_path / "images",
            storage_url=journal_url_for_path(tmp_path / "journal.log"),
        )

    state = load_slurm_lifecycle(output)
    assert state is not None
    assert state["active"] is False
    assert state["terminal_status"] == "failed"


def test_pre_submission_cleanup_cannot_terminalize_a_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A superseded submitter's exception cleanup is generation-fenced."""
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle
    from phenotypic.tune._tune_cli import _run

    output = tmp_path / "out"
    lifecycle.initialize_slurm_lifecycle(
        output, generation="owned", mode="tune"
    )
    before: dict[str, bytes] = {}

    def _supersede_then_fail(*, for_slurm: bool):
        assert for_slurm
        assert lifecycle.deactivate_generation(output, "owned")
        lifecycle.initialize_slurm_lifecycle(
            output, generation="successor", mode="tune"
        )
        before["lifecycle"] = lifecycle.lifecycle_state_path(output).read_bytes()
        raise RuntimeError("interpreter unavailable")

    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        _supersede_then_fail,
    )

    with pytest.raises(RuntimeError, match="interpreter unavailable"):
        _run._submit_slurm_fleet(
            SimpleNamespace(strategy=SimpleNamespace(n_trials=2)),
            output,
            storage_url=f"journal://{tmp_path}/journal.log",
            spec_path=tmp_path / "spec.json",
            images_dir=tmp_path / "images",
            split_path=tmp_path / "split.json",
            n_workers=2,
            generation="owned",
            shared_state_prepared=True,
        )

    assert lifecycle.lifecycle_state_path(output).read_bytes() == before["lifecycle"]


def _install_submit_fakes(monkeypatch: pytest.MonkeyPatch, submitted: list[bool]):
    """Install dependency fakes while keeping lifecycle and marker I/O real."""
    from phenotypic.tune._tune_cli import _run

    class _FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def run(self, _work, _items):
            submitted.append(True)
            return ["101"]

    monkeypatch.setattr(_run, "SlurmExecutor", _FakeExecutor)
    monkeypatch.setattr(
        "phenotypic.tune._study._optuna_store.OptunaStudyStore",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(_run, "objective_directions", lambda _scorer: None)
    monkeypatch.setattr(_run, "is_multi_objective", lambda _scorer: False)
    monkeypatch.setattr(
        "phenotypic._cli._cli_utils.get_python_command",
        lambda *, for_slurm: (["/shared/venv/bin/python"], "venv"),
    )


def _submit_spec():
    return SimpleNamespace(
        strategy=SimpleNamespace(kind="optuna", n_trials=5),
        budget=SimpleNamespace(n_trials=2),
        scorer=object(),
    )


def test_required_generation_marker_failure_prevents_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Slurm attempt cannot launch without its owned finalization record."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.tune._tune_cli import _run

    submitted: list[bool] = []
    _install_submit_fakes(monkeypatch, submitted)
    monkeypatch.setattr(
        _run,
        "_write_run_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("marker filesystem unavailable")
        ),
    )

    with pytest.raises(OSError, match="marker filesystem unavailable"):
        _run._submit_slurm_fleet(
            _submit_spec(),
            tmp_path / "out",
            storage_url=f"journal://{tmp_path}/journal.log",
            spec_path=tmp_path / "spec.json",
            images_dir=tmp_path / "images",
            split_path=tmp_path / "split.json",
            n_workers=2,
        )

    assert submitted == []
    state = load_slurm_lifecycle(tmp_path / "out")
    assert state is not None
    assert state["active"] is False
    assert state["terminal_status"] == "failed"


def test_owned_slurm_marker_records_generation_and_absolute_images_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finalizer inputs must belong to the attempt and survive a new cwd."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.tune._tune_cli import _run

    submitted: list[bool] = []
    _install_submit_fakes(monkeypatch, submitted)
    output = tmp_path / "out"
    images = Path("relative-calibration-images")
    _run._write_run_marker(
        output,
        _submit_spec(),
        storage_url=f"journal://{tmp_path}/journal.log",
        images_dir=images,
        slurm=True,
    )

    _run._submit_slurm_fleet(
        _submit_spec(),
        output,
        storage_url=f"journal://{tmp_path}/journal.log",
        spec_path=tmp_path / "spec.json",
        images_dir=images,
        split_path=tmp_path / "split.json",
        n_workers=2,
    )

    marker = __import__("json").loads(
        io.tune_cache_run_marker_path(output).read_text()
    )
    assert submitted == [True]
    assert marker["generation"]
    assert marker["images_dir"] == str(images.resolve())
    assert marker["n_trials"] == 2
    lifecycle = load_slurm_lifecycle(output)
    assert lifecycle is not None
    assert lifecycle["generation"] == marker["generation"]
    assert lifecycle["active"] is True
    assert "terminal_status" not in lifecycle


def test_stale_submitter_cannot_overwrite_or_fail_successor_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A superseded submitter must lose ownership before marker mutation."""
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle
    from phenotypic.tune._tune_cli import _run

    output = tmp_path / "out"
    images = tmp_path / "images"
    storage_url = f"journal://{tmp_path}/journal.log"
    submitted: list[bool] = []
    _install_submit_fakes(monkeypatch, submitted)
    monkeypatch.setattr(lifecycle, "new_slurm_generation", lambda: "generation-a")
    initialize = lifecycle.initialize_slurm_lifecycle
    before: dict[str, bytes] = {}

    def _initialize_then_supersede(
        output_dir: Path, *, generation: str, mode: str
    ) -> dict[str, object]:
        state = initialize(output_dir, generation=generation, mode=mode)
        assert generation == "generation-a"
        assert lifecycle.deactivate_generation(output_dir, generation)
        initialize(output_dir, generation="generation-b", mode=mode)
        _run._write_run_marker(
            output_dir,
            _submit_spec(),
            storage_url=storage_url,
            images_dir=images,
            slurm=True,
            generation="generation-b",
            required=True,
        )
        before["marker"] = io.tune_cache_run_marker_path(output_dir).read_bytes()
        before["lifecycle"] = lifecycle.lifecycle_state_path(output_dir).read_bytes()
        return state

    monkeypatch.setattr(
        lifecycle, "initialize_slurm_lifecycle", _initialize_then_supersede
    )

    with pytest.raises(lifecycle.SlurmGenerationInactiveError):
        _run._submit_slurm_fleet(
            _submit_spec(),
            output,
            storage_url=storage_url,
            spec_path=tmp_path / "spec.json",
            images_dir=images,
            split_path=tmp_path / "split.json",
            n_workers=2,
        )

    assert submitted == []
    assert io.tune_cache_run_marker_path(output).read_bytes() == before["marker"]
    assert lifecycle.lifecycle_state_path(output).read_bytes() == before["lifecycle"]
    state = lifecycle.load_slurm_lifecycle(output)
    assert state is not None
    assert state["generation"] == "generation-b"
    assert state["active"] is True
    assert "terminal_status" not in state


def test_conflicting_slurm_rerun_claims_before_any_shared_run_mutation(
    tmp_path: Path,
) -> None:
    """Writing setup before the ownership claim corrupts the incumbent attempt."""
    pytest.importorskip("optuna")
    import json

    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        lifecycle_state_path,
    )
    from phenotypic.tune.strategy import OptunaConfig
    from phenotypic.tune._study._optuna_store import OptunaStudyStore
    from phenotypic.tune._study._storage import journal_url_for_path
    from phenotypic.tune._study_store import Trial
    from phenotypic.tune._tune_cli import _run
    from tests.unit.tune.test_run_tuning_slurm import _grid_input_spec

    output = tmp_path / "out"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    source_spec = tmp_path / "source-spec.json"
    journal = tmp_path / "study.log"
    storage_url = journal_url_for_path(journal)
    spec = _grid_input_spec().model_copy(
        update={
            "strategy": OptunaConfig(
                sampler="tpe",
                n_trials=2,
                storage_url=storage_url,
            )
        }
    )
    source_spec.write_text(spec.model_dump_json(), encoding="utf-8")

    _run._resolve_calibration_images(spec, [], output)
    resolved_spec_path = io.tuning_spec_path(output)
    resolved_spec_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_spec_path.write_bytes(b"incumbent resolved spec\n")
    marker_path = io.tune_cache_run_marker_path(output)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "version": 3,
                "generation": "incumbent",
                "storage_url": storage_url,
                "n_trials": 2,
            }
        ),
        encoding="utf-8",
    )
    store = OptunaStudyStore(
        storage_url=storage_url,
        study_name=_run._STUDY_NAME,
    )
    store.append(
        Trial(
            number=0,
            params={"incumbent": True},
            score=0.5,
            terms={"cost": 0.5},
            n_images=1,
        )
    )
    initialize_slurm_lifecycle(
        output, generation="incumbent", mode="tune"
    )

    paths = (
        resolved_spec_path,
        io.tune_cache_split_assignment_path(output),
        marker_path,
        lifecycle_state_path(output),
        journal,
    )
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(RuntimeError, match="active SLURM generation"):
        _run.run_tuning(
            spec,
            [],
            output,
            slurm=True,
            spec_path=source_spec,
            images_dir=images_dir,
            storage_url=storage_url,
        )

    assert {path: path.read_bytes() for path in paths} == before
    reopened = OptunaStudyStore(
        storage_url=storage_url,
        study_name=_run._STUDY_NAME,
        create=False,
    )
    assert [(trial.number, trial.params) for trial in reopened.trials] == [
        (0, {"incumbent": True})
    ]

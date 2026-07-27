import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from dash.development.base_component import Component

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.gui.tune import _callbacks as tune_callbacks
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import create_app
from phenotypic.gui.tune._callbacks import (
    cancel_monitor_run,
    export_monitor_best_pipeline,
    reconcile_run_status,
)
from phenotypic.gui.tune._monitor import run_receipt
from phenotypic.sdk_ import best_params_path, best_pipeline_path, tuning_spec_path
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig
from phenotypic.tune._spec import Budget, TuningSpec


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)


def _component_by_id(layout, component_id: str):
    for component in _walk(layout):
        if getattr(component, "id", None) == component_id:
            return component
    raise AssertionError(f"component {component_id!r} not found")


class _Runner:
    def __init__(self):
        self.stopped = []
        self.running = True
        self.returncode = None
        self.running_checks = []
        self.reaped = []
        self.snapshots = []

    def stop(self, run_id, *, generation: UUID):
        self.stopped.append((run_id, generation))
        return self.running

    def is_running(self, run_id, *, generation: UUID):
        self.running_checks.append((run_id, generation))
        return self.running

    def reap(self, run_id, *, generation: UUID):
        self.reaped.append((run_id, generation))
        return self.returncode

    def snapshot_log(self, run_id, *, generation: UUID, tail: int):
        self.snapshots.append((run_id, generation, tail))
        return []


def _receipt_for(registry: RunRegistry, run_id: str) -> dict[str, str]:
    """Return the exact receipt for one registered test record."""
    record = registry.get(run_id)
    assert record is not None
    receipt = run_receipt(record)
    assert receipt is not None
    return receipt


def _callback_by_name(app, name: str):
    """Return one unwrapped Dash callback by its Python function name."""
    return next(
        callback.__wrapped__
        for spec in app.callback_map.values()
        if (callback := spec.get("callback")) is not None
        and callback.__wrapped__.__name__ == name
    )


def _spec(tmp_path: Path) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    csv.write_text(
        "Metadata_ImageName,Object_Label\n"
        + "\n".join(f"plate,{i}" for i in range(96))
    )
    return TuningSpec(
        pipeline=ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(Knob(key="1.ignore_zeros", domain=Categorical(choices=(True,))),)
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


def test_monitor_view_exposes_switcher_cancel_and_export_slots():
    app = create_app(root=None, url_prefix="/tune/")
    assert _component_by_id(app.layout, ids.TUNE_MONITOR_SWITCHER) is not None
    assert _component_by_id(app.layout, ids.TUNE_MONITOR_CANCEL).disabled is True
    assert _component_by_id(app.layout, ids.TUNE_MONITOR_EXPORT) is not None


def test_local_cancel_stops_runner_and_updates_registry(tmp_path: Path):
    runner = _Runner()
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="local-run",
            mode="local",
            output_dir=tmp_path,
            rel_path="local-run",
            generation=uuid4(),
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "local-run"),
    )

    record = registry.get("local-run")
    assert record is not None
    assert runner.stopped == [("local-run", record.generation)]
    assert "cancelled" in note.lower()
    assert registry.get("local-run").status == "cancelled"  # type: ignore[union-attr]


def test_local_cancel_can_return_confirmation_prompt_without_stopping(
    tmp_path: Path,
):
    runner = _Runner()
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="local-run",
            mode="local",
            output_dir=tmp_path,
            rel_path="local-run",
            generation=uuid4(),
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "local-run"),
        confirmed=False,
    )

    assert runner.stopped == []
    assert "SIGTERM" in note
    assert registry.get("local-run").status == "running"  # type: ignore[union-attr]


def test_stale_tune_receipt_cannot_touch_replacement_generation(
    tmp_path: Path,
) -> None:
    """A stale Monitor page cannot inspect, reap, log, or stop replacement B."""
    runner = _Runner()
    registry = RunRegistry()
    predecessor = RunRecord(
        run_id="same-run",
        mode="local",
        output_dir=tmp_path,
        rel_path="same-run",
        generation=uuid4(),
        status="complete",
    )
    registry.register(predecessor)
    stale_receipt = run_receipt(predecessor)
    assert stale_receipt is not None
    replacement = RunRecord(
        run_id="same-run",
        mode="local",
        output_dir=tmp_path,
        rel_path="same-run",
        generation=uuid4(),
        status="running",
    )
    registry.register(replacement)

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        receipt=stale_receipt,
    )
    status = reconcile_run_status(
        runner=runner,
        registry=registry,
        receipt=stale_receipt,
    )
    app = create_app(
        root=None,
        url_prefix="/tune/",
        registry=registry,
        runner=runner,
    )
    _switcher, cancel_disabled, local_text, _slurm_text = (
        _callback_by_name(app, "_render_monitor_registry")(
            0,
            stale_receipt,
        )
    )

    assert "no longer current" in note
    assert status is None
    assert cancel_disabled is True
    assert local_text == ""
    assert runner.running_checks == []
    assert runner.reaped == []
    assert runner.snapshots == []
    assert runner.stopped == []
    current = registry.get("same-run")
    assert current is replacement
    assert current.status == "running"


def test_slurm_cancel_is_not_supported(tmp_path: Path):
    runner = _Runner()
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="slurm-run",
            mode="slurm",
            output_dir=Path(tmp_path),
            rel_path="slurm-run",
            generation=uuid4(),
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "slurm-run"),
    )

    assert runner.stopped == []
    assert "not supported" in note.lower()
    assert registry.get("slurm-run").status == "running"  # type: ignore[union-attr]


def test_monitor_export_uses_active_registry_run(tmp_path: Path):
    output_dir = tmp_path / "run"
    tuning_spec_path(output_dir).parent.mkdir(parents=True)
    tuning_spec_path(output_dir).write_text(_spec(tmp_path).model_dump_json())
    best_params_path(output_dir).write_text(
        json.dumps(
            {
                "trial_number": 3,
                "score": 0.88,
                "objectives": {},
                "params": {"0.sigma": 3.5},
                "selection": "single_best",
            }
        )
    )
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="local-run",
            mode="local",
            output_dir=output_dir,
            rel_path="local-run",
            generation=uuid4(),
            status="complete",
        )
    )

    written = export_monitor_best_pipeline(
        registry=registry,
        receipt=_receipt_for(registry, "local-run"),
    )

    assert written == best_pipeline_path(output_dir)
    reloaded = ImagePipeline.from_json(written.read_text())
    ops = list(reloaded.get_ops().values())
    assert ops[0].sigma == 3.5


def test_stale_export_cannot_replace_new_generation_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Replacement B between preparation and publish leaves its artifact intact."""
    output_dir = tmp_path / "run"
    tuning_spec_path(output_dir).parent.mkdir(parents=True)
    tuning_spec_path(output_dir).write_text(_spec(tmp_path).model_dump_json())
    best_params_path(output_dir).write_text(
        json.dumps({"params": {"0.sigma": 3.5}})
    )
    target = best_pipeline_path(output_dir)
    replacement_payload = "generation B artifact"
    target.write_text(replacement_payload)

    registry = RunRegistry()
    predecessor = RunRecord(
        run_id="same-run",
        mode="local",
        output_dir=output_dir,
        rel_path="run",
        generation=uuid4(),
        status="complete",
    )
    registry.register(predecessor)
    stale_receipt = run_receipt(predecessor)
    assert stale_receipt is not None
    replacement = RunRecord(
        run_id="same-run",
        mode="local",
        output_dir=output_dir,
        rel_path="run",
        generation=uuid4(),
        status="running",
    )
    real_prepare = tune_callbacks.prepare_best_from_run

    def _prepare_then_replace(path: Path):
        prepared = real_prepare(path)
        registry.register(replacement)
        return prepared

    monkeypatch.setattr(
        tune_callbacks,
        "prepare_best_from_run",
        _prepare_then_replace,
    )

    with pytest.raises(ValueError, match="no longer current"):
        export_monitor_best_pipeline(
            registry=registry,
            receipt=stale_receipt,
        )

    assert registry.get("same-run") is replacement
    assert target.read_text() == replacement_payload
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_local_cancel_reconciles_already_exited_runner(tmp_path: Path):
    runner = _Runner()
    runner.running = False
    runner.returncode = 0
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="local-run",
            mode="local",
            output_dir=tmp_path,
            rel_path="local-run",
            generation=uuid4(),
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "local-run"),
    )

    assert runner.stopped == []
    assert "already exited" in note.lower()
    assert registry.get("local-run").status == "complete"  # type: ignore[union-attr]


def test_slurm_submitter_reap_marks_successful_submit_as_running(tmp_path: Path):
    runner = _Runner()
    runner.running = False
    runner.returncode = 0
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="slurm-run",
            mode="slurm",
            output_dir=tmp_path,
            rel_path="slurm-run",
            generation=uuid4(),
            status="submitting",
        )
    )

    status = reconcile_run_status(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "slurm-run"),
    )

    assert status == "running"
    assert registry.get("slurm-run").status == "running"  # type: ignore[union-attr]


def test_slurm_submitter_reap_marks_failed_submitter_failed(tmp_path: Path):
    runner = _Runner()
    runner.running = False
    runner.returncode = 1
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="slurm-run",
            mode="slurm",
            output_dir=tmp_path,
            rel_path="slurm-run",
            generation=uuid4(),
            status="submitting",
        )
    )

    status = reconcile_run_status(
        runner=runner,
        registry=registry,
        receipt=_receipt_for(registry, "slurm-run"),
    )

    assert status == "failed"
    assert registry.get("slurm-run").status == "failed"  # type: ignore[union-attr]

import json
from pathlib import Path

from dash.development.base_component import Component

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune import create_app
from phenotypic.gui.tune._callbacks import (
    cancel_monitor_run,
    export_monitor_best_pipeline,
    reconcile_run_status,
)
from phenotypic.sdk_ import best_params_path, best_pipeline_path, tuning_spec_path
from phenotypic.tune import Categorical, Evaluator, GridConfig, Knob, QCScorer, SearchSpace
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

    def stop(self, run_id):
        self.stopped.append(run_id)
        return self.running

    def is_running(self, run_id):
        return self.running

    def reap(self, run_id):
        return self.returncode


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
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        run_id="local-run",
    )

    assert runner.stopped == ["local-run"]
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
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        run_id="local-run",
        confirmed=False,
    )

    assert runner.stopped == []
    assert "SIGTERM" in note
    assert registry.get("local-run").status == "running"  # type: ignore[union-attr]


def test_slurm_cancel_is_not_supported(tmp_path: Path):
    runner = _Runner()
    registry = RunRegistry()
    registry.register(
        RunRecord(
            run_id="slurm-run",
            mode="slurm",
            output_dir=Path(tmp_path),
            rel_path="slurm-run",
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        run_id="slurm-run",
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
            status="complete",
        )
    )

    written = export_monitor_best_pipeline(registry=registry, run_id="local-run")

    assert written == best_pipeline_path(output_dir)
    reloaded = ImagePipeline.from_json(written.read_text())
    ops = list(reloaded.get_ops().values())
    assert ops[0].sigma == 3.5


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
            status="running",
        )
    )

    note = cancel_monitor_run(
        runner=runner,
        registry=registry,
        run_id="local-run",
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
            status="submitting",
        )
    )

    status = reconcile_run_status(
        runner=runner,
        registry=registry,
        run_id="slurm-run",
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
            status="submitting",
        )
    )

    status = reconcile_run_status(
        runner=runner,
        registry=registry,
        run_id="slurm-run",
    )

    assert status == "failed"
    assert registry.get("slurm-run").status == "failed"  # type: ignore[union-attr]

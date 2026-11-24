"""The optional ``stage`` field on staged-engine events (Spec 1 §9, OQ1)."""

from phenotypic._cli._cli_update_state import (
    aggregate_stage_state_from_events,
    aggregate_state_from_events,
    append_completion_event,
    append_event,
    parse_event_line,
)


def test_stage_round_trips_through_event_line(tmp_path):
    log = tmp_path / "events.log"
    append_event(log, "ds", "img.tiff", "started", stage="stage2")
    line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    ev = parse_event_line(line)
    assert ev.stage == "stage2"
    assert ev.status == "started"
    # placeholder SLURM slots are present (so stage lands at field 8) but empty
    assert ev.slurm_job_id == ""
    assert ev.slurm_array_task_id == ""


def test_legacy_line_without_stage_parses_stage_none(tmp_path):
    log = tmp_path / "events.log"
    append_event(log, "ds", "img.tiff", "completed")  # no stage, no slurm
    ev = parse_event_line(log.read_text(encoding="utf-8").strip())
    assert ev.stage is None
    assert ev.status == "completed"


def test_slurm_line_without_stage_still_parses(tmp_path):
    log = tmp_path / "events.log"
    append_event(log, "ds", "img.tiff", "started", slurm_job_id="123",
                 slurm_array_task_id="4")
    ev = parse_event_line(log.read_text(encoding="utf-8").strip())
    assert ev.slurm_job_id == "123"
    assert ev.slurm_array_task_id == "4"
    assert ev.stage is None


def test_completion_event_forwards_stage(tmp_path):
    log = tmp_path / "events.log"
    append_completion_event(log, "ds", "img.tiff", "completed", stage="stage3")
    ev = parse_event_line(log.read_text(encoding="utf-8").strip())
    assert ev.stage == "stage3"


def test_intermediate_stage_completion_is_not_overall_complete(tmp_path):
    log = tmp_path / "events.log"
    append_completion_event(log, "ds", "img1.tiff", "completed", stage="stage1")
    state = aggregate_state_from_events(log)
    # stage1 done is NOT overall done — image is in progress, not completed
    assert "img1.tiff" not in state["ds"].completed
    assert "img1.tiff" in state["ds"].started


def test_stage3_completion_is_overall_complete(tmp_path):
    log = tmp_path / "events.log"
    for st in ("stage1", "stage2", "stage3"):
        append_completion_event(log, "ds", "img1.tiff", "completed", stage=st)
    state = aggregate_state_from_events(log)
    assert "img1.tiff" in state["ds"].completed


def test_legacy_non_stage_completion_is_overall_complete(tmp_path):
    # backward compatibility: a plain completion (no stage) is overall done
    log = tmp_path / "events.log"
    append_completion_event(log, "ds", "img1.tiff", "completed")
    state = aggregate_state_from_events(log)
    assert "img1.tiff" in state["ds"].completed


def test_intermediate_stage_completion_clears_prior_failure(tmp_path):
    log = tmp_path / "events.log"
    append_event(log, "ds", "img1.tiff", "failed",
                 error_msg="boom", stage="stage1")
    append_completion_event(log, "ds", "img1.tiff", "completed", stage="stage1")
    state = aggregate_state_from_events(log)
    assert "img1.tiff" not in state["ds"].failed  # retry success cleared it
    assert "img1.tiff" in state["ds"].started


def test_per_stage_aggregation_buckets_by_stage(tmp_path):
    log = tmp_path / "events.log"
    append_completion_event(log, "ds", "img1.tiff", "completed", stage="stage1")
    append_completion_event(log, "ds", "img1.tiff", "completed", stage="stage2")
    per = aggregate_stage_state_from_events(log)
    assert "img1.tiff" in per["ds"]["stage1"].completed
    assert "img1.tiff" in per["ds"]["stage2"].completed
    assert "stage3" not in per["ds"]  # not reached yet


def test_per_stage_aggregation_ignores_legacy_events(tmp_path):
    log = tmp_path / "events.log"
    append_completion_event(log, "ds", "img1.tiff", "completed")  # no stage
    per = aggregate_stage_state_from_events(log)
    assert per == {}


# ---------------------------------------------------------------------------
# #282 P4 T2: a run that embeds on the GPU finishes at Stage 4 (plan review I8)
# ---------------------------------------------------------------------------


def event_log_with(directory, events):
    """An event log holding ``(dataset, image, status, stage)`` rows, in order."""
    log = directory / "events.log"
    for dataset, image, status, stage in events:
        append_completion_event(log, dataset, image, status, stage=stage)
    return log


def test_stage3_completion_is_not_completion_when_the_run_embeds_on_gpu(tmp_path):
    log = event_log_with(tmp_path, [("ds", "a", "completed", "stage3")])
    assert "a" not in aggregate_state_from_events(log, terminal_stage="stage4")["ds"].completed
    assert "a" in aggregate_state_from_events(log, terminal_stage="stage4")["ds"].started
    assert "a" in aggregate_state_from_events(log)["ds"].completed


def test_stage4_completion_is_completion_when_the_run_embeds_on_gpu(tmp_path):
    log = event_log_with(
        tmp_path, [("ds", "a", "completed", "stage3"), ("ds", "a", "completed", "stage4")]
    )
    assert "a" in aggregate_state_from_events(log, terminal_stage="stage4")["ds"].completed


def test_run_terminal_stage():
    from phenotypic._cli._stages import run_terminal_stage

    assert run_terminal_stage({"embed_stage": "gpu"}) == "stage4"
    assert run_terminal_stage({"embed_stage": "inline"}) == "stage3"
    assert run_terminal_stage({}) == run_terminal_stage(None) == "stage3"


def _deferring_state(tmp_path, embed_stage):
    """A saved one-image state whose config records *embed_stage*, and its event log."""
    from datetime import datetime

    from phenotypic._cli._cli_state_management import save_processing_state
    from phenotypic._cli._cli_types import DatasetState, ProcessingState
    from phenotypic.sdk_ import event_log_path

    now = datetime.now()
    state = ProcessingState(
        version="3.0.0",
        pipeline_path=tmp_path / "p.json",
        input_path=tmp_path / "in",
        output_dir=tmp_path,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets={"ds": DatasetState(initial_images={"a"})},
        config={"embed_stage": embed_stage},
    )
    save_processing_state(state, tmp_path)
    append_completion_event(event_log_path(tmp_path), "ds", "a", "completed", stage="stage3")
    return state


def test_the_state_readers_count_completion_at_the_runs_terminal_stage(tmp_path):
    from phenotypic._cli._cli_state_management import (
        load_processing_state,
        recorded_terminal_stage,
        update_state_from_events,
    )

    gpu, inline = tmp_path / "gpu", tmp_path / "inline"
    gpu_state, inline_state = _deferring_state(gpu, "gpu"), _deferring_state(inline, "inline")
    assert "a" not in load_processing_state(gpu).datasets["ds"].completed
    assert "a" not in update_state_from_events(gpu_state, gpu).datasets["ds"].completed
    assert "a" in load_processing_state(inline).datasets["ds"].completed
    assert "a" in update_state_from_events(inline_state, inline).datasets["ds"].completed
    assert recorded_terminal_stage(gpu) == "stage4"
    assert recorded_terminal_stage(inline) == recorded_terminal_stage(tmp_path / "none") == "stage3"


def test_every_event_aggregation_caller_passes_the_terminal_stage():
    """I8: each call of the two aggregators in ``src/`` names ``terminal_stage=``.

    A caller that omits it reads a Stage-3 completion of a GPU run as done.
    """
    import ast
    from pathlib import Path

    import phenotypic

    names = {"aggregate_state_from_events", "aggregate_state_from_events_with_diagnostics"}
    root = Path(phenotypic.__file__).parent
    callers, missing = [], []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name not in names:
                continue
            where = f"{path.relative_to(root)}:{node.lineno}"
            if path.name == "_cli_update_state.py":
                continue  # the one wrapper, which forwards its own parameter
            callers.append(where)
            if not any(keyword.arg == "terminal_stage" for keyword in node.keywords):
                missing.append(where)
    assert len(callers) == 6, callers  # the six callers plan review I8 lists
    assert missing == []

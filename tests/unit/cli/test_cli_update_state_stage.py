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

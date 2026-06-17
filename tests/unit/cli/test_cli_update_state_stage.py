"""The optional ``stage`` field on staged-engine events (Spec 1 §9, OQ1)."""

from phenotypic._cli._cli_update_state import (
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

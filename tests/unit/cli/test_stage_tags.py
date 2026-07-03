"""Unit tests for staged GPU event stage tags."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli._cli_update_state import append_event, parse_event_line
from phenotypic._cli._stages import (
    STAGE_GPU_DETECT,
    STAGE_MEASURE,
    STAGE_PREPROCESS,
    STAGED_TERMINAL_STAGE,
    validate_stage_tag,
)


def test_validate_stage_tag_accepts_known_tags_and_legacy_none() -> None:
    """Known staged-engine tags are returned unchanged; legacy rows use None."""
    assert validate_stage_tag(None) is None
    assert validate_stage_tag("stage1") == STAGE_PREPROCESS
    assert validate_stage_tag("stage2") == STAGE_GPU_DETECT
    assert validate_stage_tag("stage3") == STAGE_MEASURE
    assert STAGED_TERMINAL_STAGE == STAGE_MEASURE


def test_validate_stage_tag_rejects_unknown_tag() -> None:
    """Unknown non-empty stage tags are malformed staged events."""
    with pytest.raises(ValueError, match="Invalid stage tag"):
        validate_stage_tag("stage4")


def test_parse_event_line_keeps_legacy_no_stage_rows() -> None:
    """Old event-log rows without stage fields remain valid."""
    event = parse_event_line("2026-07-03T12:00:00|ds|img.tiff|completed")

    assert event.stage is None
    assert event.status == "completed"


def test_parse_event_line_validates_stage_field() -> None:
    """The parser accepts known stages and rejects unknown non-empty stages."""
    event = parse_event_line(
        "2026-07-03T12:00:00|ds|img.tiff|completed||||stage2"
    )
    assert event.stage == STAGE_GPU_DETECT

    with pytest.raises(ValueError, match="Invalid stage tag"):
        parse_event_line(
            "2026-07-03T12:00:00|ds|img.tiff|completed||||stage4"
        )


def test_append_event_rejects_invalid_stage_without_writing(tmp_path: Path) -> None:
    """Invalid stage tags fail before appending durable event state."""
    event_log = tmp_path / "processing_events.log"

    with pytest.raises(ValueError, match="Invalid stage tag"):
        append_event(event_log, "ds", "img.tiff", "started", stage="stage4")

    assert not event_log.exists()

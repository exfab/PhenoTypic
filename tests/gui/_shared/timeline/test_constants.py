"""Timeline-view shared constants + thumbnail bucket snapping."""
from __future__ import annotations

import pytest

from phenotypic.gui._config import (
    THUMB_SIZE_BUCKETS,
    TIMELINE_FOCUS_MARGIN,
    TIMELINE_TILE_SIZE_DEFAULT,
    TIMELINE_TILE_SIZE_MAX,
    TIMELINE_TILE_SIZE_MIN,
    TIMELINE_TILE_SIZE_STEP,
    snap_thumb_bucket,
    step_timeline_tile_size,
    stepped_timeline_tile_size_from_trigger,
)


def test_focus_margin_is_a_positive_int() -> None:
    # Focus-navigate mount-ring distance in cells (spec §16.3).
    assert isinstance(TIMELINE_FOCUS_MARGIN, int)
    assert TIMELINE_FOCUS_MARGIN >= 1


def test_buckets_are_sorted_ascending_ints() -> None:
    assert THUMB_SIZE_BUCKETS == tuple(sorted(THUMB_SIZE_BUCKETS))
    assert all(isinstance(b, int) for b in THUMB_SIZE_BUCKETS)
    assert THUMB_SIZE_BUCKETS[0] == 64
    assert THUMB_SIZE_BUCKETS[-1] == 256


def test_tile_size_stepper_bounds_mirror_colony() -> None:
    # Mirrors COLONY_TILE_SIZE_* (default 150, step 16, range 64..400).
    assert (
        TIMELINE_TILE_SIZE_DEFAULT,
        TIMELINE_TILE_SIZE_STEP,
        TIMELINE_TILE_SIZE_MIN,
        TIMELINE_TILE_SIZE_MAX,
    ) == (150, 16, 64, 400)


@pytest.mark.parametrize(
    "requested,expected",
    [
        (10, 64),    # below min → smallest bucket
        (64, 64),    # exact
        (65, 96),    # snap up
        (100, 128),  # snap up
        (192, 192),  # exact
        (300, 256),  # above max → largest bucket
    ],
)
def test_snap_thumb_bucket_snaps_up_and_clamps(requested: int, expected: int) -> None:
    assert snap_thumb_bucket(requested) == expected


def test_stepped_timeline_tile_size_from_trigger_picks_direction() -> None:
    # Mirrors the colony helper: minus → step down, anything else → step up.
    # Both timeline surfaces (Browse + Results) funnel their stepper callbacks
    # through this single Dash-free helper.
    up = stepped_timeline_tile_size_from_trigger(
        "plus", TIMELINE_TILE_SIZE_DEFAULT, plus_id="plus", minus_id="minus"
    )
    down = stepped_timeline_tile_size_from_trigger(
        "minus", TIMELINE_TILE_SIZE_DEFAULT, plus_id="plus", minus_id="minus"
    )
    assert up == step_timeline_tile_size(TIMELINE_TILE_SIZE_DEFAULT, 1)
    assert down == step_timeline_tile_size(TIMELINE_TILE_SIZE_DEFAULT, -1)


def test_stepped_timeline_tile_size_from_trigger_unknown_id_steps_up() -> None:
    # An initial-mount echo / unrecognised id steps up rather than down so the
    # size never silently jumps backwards (matches the colony/dim helpers).
    size = stepped_timeline_tile_size_from_trigger(
        None, TIMELINE_TILE_SIZE_DEFAULT, plus_id="plus", minus_id="minus"
    )
    assert size == step_timeline_tile_size(TIMELINE_TILE_SIZE_DEFAULT, 1)

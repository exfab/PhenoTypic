"""Tests for the shared positive SLURM time parser."""

from __future__ import annotations

import pytest

from phenotypic.sdk_.slurm import parse_slurm_time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (10, "00:10:00"),
        ("10", "00:10:00"),
        (90, "01:30:00"),
        ("00:10:00", "00:10:00"),
        ("25:00:00", "25:00:00"),
        ("1-04:00:00", "1-04:00:00"),
    ],
)
def test_parse_slurm_time_accepts_supported_forms(
    value: object, expected: str | None
) -> None:
    assert parse_slurm_time(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        "0",
        "00:00:00",
        "1:00:00",
        "00:60:00",
        "00:00:60",
        "1-24:00:00",
        "1-004:00:00",
        "tomorrow",
        1.5,
        True,
    ],
)
def test_parse_slurm_time_rejects_invalid_or_nonpositive_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        parse_slurm_time(value)

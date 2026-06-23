"""Plate-identity pattern compilation + matching."""
from __future__ import annotations

import pytest

from phenotypic.gui.browse._plate_pattern import (
    PatternError,
    PlateMatch,
    parse_plate_identity,
)


def test_placeholder_extracts_plate_and_time() -> None:
    out = parse_plate_identity(
        ["Exp1_PlateA_t03", "Exp1_PlateB_t10"], "{plate}_t{time}"
    )
    assert out == [
        PlateMatch("Exp1_PlateA_t03", "Exp1_PlateA", "03"),
        PlateMatch("Exp1_PlateB_t10", "Exp1_PlateB", "10"),
    ]


def test_nonmatching_stem_yields_none() -> None:
    out = parse_plate_identity(["junk"], "{plate}_t{time}")
    assert out == [PlateMatch("junk", None, None)]


def test_plate_only_pattern_leaves_time_none() -> None:
    out = parse_plate_identity(["plateA"], "{plate}")
    assert out == [PlateMatch("plateA", "plateA", None)]


def test_missing_plate_token_raises() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], "t{time}")


def test_duplicate_token_raises() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], "{plate}_{plate}")


def test_advanced_regex_requires_named_plate_group() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], r"(.+)", advanced=True)
    out = parse_plate_identity(["A-1"], r"(?P<plate>[A-Z]+)-(?P<time>\d+)", advanced=True)
    assert out == [PlateMatch("A-1", "A", "1")]


def test_invalid_regex_raises_pattern_error() -> None:
    with pytest.raises(PatternError):
        parse_plate_identity(["x"], r"(?P<plate>", advanced=True)

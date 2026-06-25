"""``_parse_key`` grammar coverage — flat / presence / nested (P3-5a).

Exercises the typed parse result (``FlatKey`` / ``PresenceKey`` / ``NestedKey``),
the ``name[i]`` nested disambiguator, the **depth cap of 1** (a second ``[i]``
is rejected), and the malformed-segment error paths.
"""
from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.tune._evaluation._builder import (
    FlatKey,
    NestedKey,
    PresenceKey,
    _parse_key,
)


def _flat_base() -> list:
    base = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    return list(base.get_ops().values())


def _composite_base() -> list:
    base = ImagePipeline(ops=[CompositeDetector()])
    return list(base.get_ops().values())


# --------------------------------------------------------------------------- #
# flat scalar
# --------------------------------------------------------------------------- #
def test_flat_scalar_parses_to_flatkey():
    parsed = _parse_key("0.sigma", _flat_base())
    assert isinstance(parsed, FlatKey)
    assert parsed.position == 0
    assert parsed.field == "sigma"


# --------------------------------------------------------------------------- #
# presence (two- and three-part)
# --------------------------------------------------------------------------- #
def test_bare_presence_parses_to_presencekey_without_class():
    parsed = _parse_key("0.__enabled__", _flat_base())
    assert isinstance(parsed, PresenceKey)
    assert parsed.position == 0
    assert parsed.cls_name is None


def test_classed_presence_parses_to_presencekey_with_class():
    parsed = _parse_key("0.GaussianBlur.__enabled__", _flat_base())
    assert isinstance(parsed, PresenceKey)
    assert parsed.position == 0
    assert parsed.cls_name == "GaussianBlur"


def test_classed_presence_class_mismatch_raises():
    with pytest.raises(ValueError, match="OtsuDetector"):
        _parse_key("0.OtsuDetector.__enabled__", _flat_base())


# --------------------------------------------------------------------------- #
# nested
# --------------------------------------------------------------------------- #
def test_nested_key_parses_to_nestedkey():
    parsed = _parse_key("0.ops[0].ignore_zeros", _composite_base())
    assert isinstance(parsed, NestedKey)
    assert parsed.position == 0
    assert parsed.field == "ops"
    assert parsed.index == 0
    assert parsed.leaf == "ignore_zeros"


def test_nested_key_high_index_parses():
    parsed = _parse_key("0.ops[1].thresh_method", _composite_base())
    assert isinstance(parsed, NestedKey)
    assert parsed.index == 1
    assert parsed.leaf == "thresh_method"


# --------------------------------------------------------------------------- #
# depth cap = 1
# --------------------------------------------------------------------------- #
def test_second_index_segment_rejected_as_depth_error():
    with pytest.raises(ValueError, match="depth"):
        _parse_key("0.ops[0].sub[1].x", _composite_base())


# --------------------------------------------------------------------------- #
# malformed
# --------------------------------------------------------------------------- #
def test_malformed_bracket_raises():
    with pytest.raises(ValueError):
        _parse_key("0.ops[x].ignore_zeros", _composite_base())


def test_nested_missing_leaf_raises():
    with pytest.raises(ValueError):
        _parse_key("0.ops[0]", _composite_base())


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
def test_position_out_of_range_raises():
    with pytest.raises(IndexError):
        _parse_key("9.sigma", _flat_base())

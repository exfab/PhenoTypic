from __future__ import annotations

import pytest

from phenotypic.tune._search_space._targets import (
    Nested,
    Param,
    Presence,
    parse_key,
)


def test_param_key():
    assert Param(op=0, field="sigma").key == "0.sigma"


def test_presence_key_bare_and_classed():
    assert Presence(op=0).key == "0.__enabled__"
    assert Presence(op=0, op_class="GaussianBlur").key == "0.GaussianBlur.__enabled__"


def test_nested_key():
    t = Nested(op=1, field="detectors", index=0, leaf="ignore_zeros")
    assert t.key == "1.detectors[0].ignore_zeros"


def test_targets_are_frozen_and_discriminated():
    t = Param(op=0, field="sigma")
    with pytest.raises(Exception):
        t.op = 5  # frozen
    assert t.kind == "param"
    assert Presence(op=0).kind == "presence"
    assert Nested(op=0, field="r", index=0, leaf="x").kind == "nested"


@pytest.mark.parametrize("key", [
    "0.sigma", "0.__enabled__", "0.GaussianBlur.__enabled__", "1.detectors[0].ignore_zeros",
])
def test_parse_key_round_trips(key):
    assert parse_key(key).key == key          # string-preserving


def test_parse_key_recovers_op_class_only_for_classed_presence():
    assert parse_key("0.GaussianBlur.__enabled__").op_class == "GaussianBlur"
    assert parse_key("0.sigma").op_class is None
    assert parse_key("0.__enabled__").op_class is None


def test_parse_key_rejects_malformed():
    with pytest.raises(ValueError):
        parse_key("notanint.sigma")
    with pytest.raises(ValueError):
        parse_key("0")

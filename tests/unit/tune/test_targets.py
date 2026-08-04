from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune._search_space._targets import (
    Nested,
    Param,
    Presence,
    parse_key,
    with_op_class,
)


def test_param_key():
    assert Param(op=0, field="sigma").key == "0.sigma"


def test_presence_key_bare_and_classed():
    assert Presence(op=0).key == "0.__enabled__"
    assert Presence(op=0, op_class="BlurGauss").key == "0.BlurGauss.__enabled__"


def test_nested_key():
    t = Nested(op=1, field="ops", index=0, leaf="ignore_zeros")
    assert t.key == "1.ops[0].ignore_zeros"


def test_targets_are_frozen_and_discriminated():
    t = Param(op=0, field="sigma")
    with pytest.raises(Exception):
        t.op = 5  # frozen
    assert t.kind == "param"
    assert Presence(op=0).kind == "presence"
    assert Nested(op=0, field="r", index=0, leaf="x").kind == "nested"


@pytest.mark.parametrize("key", [
    "0.sigma", "0.__enabled__", "0.BlurGauss.__enabled__", "1.ops[0].ignore_zeros",
])
def test_parse_key_round_trips(key):
    assert parse_key(key).key == key          # string-preserving


def test_parse_key_recovers_op_class_only_for_classed_presence():
    assert parse_key("0.BlurGauss.__enabled__").op_class == "BlurGauss"
    assert parse_key("0.sigma").op_class is None
    assert parse_key("0.__enabled__").op_class is None


def test_parse_key_rejects_malformed():
    with pytest.raises(ValueError):
        parse_key("notanint.sigma")
    with pytest.raises(ValueError):
        parse_key("0")


def test_with_op_class_fills_from_pipeline():
    ops = list(ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()]).get_ops().values())
    assert with_op_class(Param(op=0, field="sigma"), ops).op_class == "BlurGauss"
    assert with_op_class(Param(op=1, field="ignore_zeros"), ops).op_class == "OtsuDetector"


def test_with_op_class_leaves_out_of_range_untouched():
    assert with_op_class(Param(op=9, field="x"), []).op_class is None


# --- review-fix regressions: parse_key strictness must match build_pipeline ---

def test_parse_key_rejects_depth_2_nested():
    # parse_key must not be more permissive than build_pipeline's depth-1 cap:
    # a leaf carrying a further "[i]" segment is rejected (else a Knob would
    # parse fine but fail at evaluation in build_pipeline).
    with pytest.raises(ValueError, match="depth cap"):
        parse_key("0.f[1].g[2].h")
    with pytest.raises(ValueError, match="depth cap"):
        parse_key("0.f[1].g[2]")


def test_parse_key_accepts_depth_1_nested():
    t = parse_key("0.ops[1].ignore_zeros")
    assert (t.op, t.field, t.index, t.leaf) == (0, "ops", 1, "ignore_zeros")


def test_parse_key_rejects_empty_class_presence():
    with pytest.raises(ValueError, match="empty class segment"):
        parse_key("0..__enabled__")


def test_parent_presence_condition_returns_a_target_parent():
    # When an op opts into presence-wrapping, the nested-knob gate must be a
    # structured Presence *target* (not a raw string) — the nested recursion
    # stamps conditional_on via model_copy (bypassing Knob's str coercion), so a
    # string parent would make Knob.is_active (ptarget.key) raise AttributeError.
    from phenotypic.tune._search_space._infer import _parent_presence_condition

    class _Wrapped:
        _tune_optional = True

    cond = _parent_presence_condition(_Wrapped(), 2)
    assert cond is not None
    (parent, value), = cond
    assert isinstance(parent, Presence)
    assert parent.op == 2 and parent.op_class == "_Wrapped" and value is True
    assert parent.key == "2._Wrapped.__enabled__"

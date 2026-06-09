from __future__ import annotations

import pytest

from phenotypic.tune._search_space import (
    Categorical,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._search_space._targets import Param, Presence


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
        Knob(key="1.size", domain=IntRange(low=4, high=400)),
        Knob(
            key="0.mode",
            domain=Categorical(choices=("reflect", "nearest")),
            conditional_on=(("0.GaussianBlur.__enabled__", True),),
        ),
    ))


def test_knob_defaults():
    k = Knob(key="0.size", domain=IntRange(low=1, high=9))
    assert k.source == "manual"
    assert k.needs_review is False
    assert k.description == ""
    assert k.conditional_on is None


def test_knob_source_is_a_closed_set():
    # source is a KnobSource Literal, not a bare str: a known origin is accepted,
    # an arbitrary string is rejected at construction.
    from pydantic import ValidationError

    assert Knob(
        key="0.size", domain=IntRange(low=1, high=9), source="tune_spec"
    ).source == ("tune_spec")
    with pytest.raises(ValidationError):
        Knob(key="0.size", domain=IntRange(low=1, high=9), source="not-a-real-origin")


def test_searchspace_keys_and_domain_lookup():
    s = _space()
    assert s.keys() == ["0.sigma", "1.size", "0.mode"]
    assert s.domain("1.size") == IntRange(low=4, high=400)
    with pytest.raises(KeyError):
        s.domain("nope")


def test_searchspace_iterates_knobs():
    assert [k.key for k in _space()] == [
        "0.sigma", "1.size", "0.mode",
    ]


def test_searchspace_roundtrip_with_conditional_and_mixed_domains():
    s = _space()
    blob = s.model_dump_json()
    back = SearchSpace.model_validate_json(blob)
    assert back == s
    # conditional_on survives (list<->tuple coercion) and the domain discriminator routes
    cond = next(k for k in back if k.key.endswith(".mode"))
    assert cond.conditional_on == (
        (Presence(op=0, op_class="GaussianBlur"), True),
    )
    assert isinstance(cond.domain, Categorical)


def test_knob_accepts_target_and_string_key_equivalently():
    a = Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0))
    b = Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0))
    assert a.target == b.target
    assert a.key == b.key == "0.sigma"          # .key property reads through


def test_knob_serializes_target_structurally_and_loads_legacy_string():
    k = Knob(key="0.GaussianBlur.__enabled__", domain=Categorical(choices=(True, False)))
    dumped = k.model_dump()
    assert dumped["target"]["kind"] == "presence"
    assert "key" not in dumped                  # structured, not the string
    # legacy string still loads:
    again = Knob.model_validate({"key": "0.GaussianBlur.__enabled__",
                                 "domain": {"kind": "categorical", "choices": [True, False]}})
    assert again.key == "0.GaussianBlur.__enabled__"


def test_search_space_targets_and_keys():
    space = SearchSpace(knobs=(
        Knob(target=Param(op=0, field="sigma"), domain=FloatRange(low=0.5, high=8.0)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))
    assert space.keys() == ["0.sigma", "1.ignore_zeros"]      # via .key property
    assert [t.kind for t in space.targets()] == ["param", "param"]
    assert space.domain("0.sigma").high == 8.0

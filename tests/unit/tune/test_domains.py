from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from phenotypic.tune._search_space._domains import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
)


def test_construction_and_defaults():
    assert Categorical(choices=(True, False)).choices == (True, False)
    r = IntRange(low=2, high=20)
    assert (r.low, r.high, r.step, r.log) == (2, 20, 1, False)
    assert FloatRange(low=1e-3, high=1.0, log=True).log is True
    assert Fixed(value=4.0).value == 4.0


def test_list_choices_coerced_to_tuple():
    assert Categorical(choices=["disk", "square"]).choices == ("disk", "square")


def test_frozen():
    with pytest.raises(ValidationError):
        IntRange(low=2, high=20).low = 3  # type: ignore[misc]


def test_range_validation():
    with pytest.raises(ValidationError):
        IntRange(low=20, high=2)
    with pytest.raises(ValidationError):
        FloatRange(low=2.0, high=1.0)


def test_discriminated_union_roundtrip():
    adapter: TypeAdapter[Domain] = TypeAdapter(Domain)
    for dom in [
        Categorical(choices=(1, 2, 3)),
        IntRange(low=2, high=20, step=2),
        FloatRange(low=0.5, high=8.0, log=True),
        Fixed(value="reflect"),
    ]:
        blob = adapter.dump_json(dom)
        back = adapter.validate_json(blob)
        assert back == dom
        assert json.loads(blob)["kind"] == dom.kind

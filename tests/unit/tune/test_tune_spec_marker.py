"""Tests for the ``TuneSpec`` per-field tuning-metadata marker (P3-1).

``TuneSpec`` mirrors ``_OperationFieldMarker``: a frozen, slotted, non-pydantic
sentinel that rides in an ``Annotated[T, TuneSpec(...)]`` chain, is a complete
no-op at runtime, and is retrievable from ``Model.model_fields[f].metadata``.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel

from phenotypic.tune import TuneSpec


def test_construction_defaults():
    spec = TuneSpec()
    assert spec.low is None
    assert spec.high is None
    assert spec.step is None
    assert spec.log is False
    assert spec.categories is None
    assert spec.tunable is True


def test_positional_low_high():
    spec = TuneSpec(0.5, 5.0, log=True)
    assert spec.low == 0.5
    assert spec.high == 5.0
    assert spec.log is True


def test_tunable_false_sentinel():
    spec = TuneSpec(tunable=False)
    assert spec.tunable is False
    assert spec.low is None and spec.high is None


def test_categories_round_trip():
    spec = TuneSpec(categories=("reflect", "nearest"))
    assert spec.categories == ("reflect", "nearest")
    # Lists are coerced to tuples so the marker stays hashable.
    spec2 = TuneSpec(categories=["disk", "square"])
    assert spec2.categories == ("disk", "square")
    assert isinstance(spec2.categories, tuple)


def test_step():
    spec = TuneSpec(2, 20, step=2)
    assert spec.step == 2


def test_eq_and_hash():
    a = TuneSpec(0.5, 5.0, log=True)
    b = TuneSpec(0.5, 5.0, log=True)
    c = TuneSpec(0.5, 5.0, log=False)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # De-dupes in a set (the Annotated-chain de-dup contract).
    assert len({a, b}) == 1
    # Distinct categories distinguish.
    assert TuneSpec(categories=("a",)) != TuneSpec(categories=("b",))
    assert TuneSpec(categories=("a",)) == TuneSpec(categories=("a",))


def test_not_equal_to_other_types():
    assert TuneSpec() != object()
    assert TuneSpec() != "TuneSpec()"


def test_repr_is_informative():
    spec = TuneSpec(0.5, 5.0, log=True)
    text = repr(spec)
    assert text.startswith("TuneSpec(")
    assert "low=0.5" in text
    assert "high=5.0" in text
    assert "log=True" in text


def test_slotted_no_dict():
    spec = TuneSpec()
    assert not hasattr(spec, "__dict__")


def test_survives_in_model_fields_metadata():
    class Op(BaseModel):
        sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0

    metas = Op.model_fields["sigma"].metadata
    found = [m for m in metas if isinstance(m, TuneSpec)]
    assert len(found) == 1
    assert found[0] == TuneSpec(0.5, 5.0, log=True)


def test_runtime_no_op_construction():
    # The marker never constrains the value — the field is a plain float.
    class Op(BaseModel):
        sigma: Annotated[float, TuneSpec(0.5, 5.0)] = 2.0

    assert Op(sigma=999.0).sigma == 999.0


def test_nested_under_optional():
    # A marker nested under Optional still lives in the annotation tree
    # (the inference layer walks the tree to find it).
    class Op(BaseModel):
        sigma: Annotated[Optional[float], TuneSpec(0.5, 5.0)] = None

    metas = Op.model_fields["sigma"].metadata
    assert any(isinstance(m, TuneSpec) for m in metas)

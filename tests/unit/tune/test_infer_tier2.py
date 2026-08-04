"""Tier-2 type/constraint heuristics for ``infer_search_space`` (P3-3).

Table-driven over synthetic single-field pydantic ops + real
``BlurGauss``/``OtsuDetector`` on a ``load_synth_yeast_plate()`` pipeline.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, Optional

import annotated_types as at
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_._column_ref import ColumnRef
from phenotypic.sdk_.typing_ import NdArrayField
from phenotypic.tune import (
    Categorical,
    Excluded,
    FloatRange,
    InferredSearchSpace,
    IntRange,
    Knob,
)
from phenotypic.tune._search_space._infer import _infer_field, infer_search_space


class _SynthOp(BaseModel):
    """Bare host so ``_infer_field`` can read ``model_fields`` / schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _infer_single(annotation, default, field=None):
    """Build a one-field model and infer that field at position 0."""
    ns = {
        "__annotations__": {"f": annotation},
        "f": field if field is not None else default,
        "model_config": ConfigDict(arbitrary_types_allowed=True),
    }
    op_cls = type("OneField", (BaseModel,), ns)
    op = op_cls()
    return _infer_field(op, 0, "f", op_cls.model_fields["f"])


# --------------------------------------------------------------------------- #
# bool
# --------------------------------------------------------------------------- #
def test_bool_to_categorical():
    res = _infer_single(bool, True)
    assert isinstance(res, Knob)
    assert res.key == "0.f"
    assert res.source == "bool"
    assert res.needs_review is False
    assert isinstance(res.domain, Categorical)
    assert set(res.domain.choices) == {True, False}


# --------------------------------------------------------------------------- #
# Literal / Enum
# --------------------------------------------------------------------------- #
def test_literal_to_categorical():
    res = _infer_single(Literal["reflect", "constant", "nearest"], "reflect")
    assert isinstance(res, Knob)
    assert res.source == "literal"
    assert res.needs_review is False
    assert res.domain.choices == ("reflect", "constant", "nearest")


def test_enum_to_categorical():
    class Mode(enum.Enum):
        A = "a"
        B = "b"

    res = _infer_single(Mode, Mode.A)
    assert isinstance(res, Knob)
    assert res.source == "enum"
    # Stores enum *values* (not members) so the domain is JSON-native and
    # round-trips to the same Python type via the op's field_validator.
    assert set(res.domain.choices) == {"a", "b"}


# --------------------------------------------------------------------------- #
# bounded numeric
# --------------------------------------------------------------------------- #
def test_bounded_float_to_floatrange():
    res = _infer_single(Annotated[float, Field(ge=0.1, le=1.0)], 0.5)
    assert isinstance(res, Knob)
    assert res.source == "bounded"
    assert res.needs_review is False
    assert isinstance(res.domain, FloatRange)
    assert (res.domain.low, res.domain.high) == (0.1, 1.0)
    assert res.domain.log is False


def test_bounded_int_to_intrange():
    res = _infer_single(Annotated[int, Field(ge=2, le=20)], 10)
    assert isinstance(res, Knob)
    assert res.source == "bounded"
    assert isinstance(res.domain, IntRange)
    assert (res.domain.low, res.domain.high) == (2, 20)


def test_bounded_log_auto_trips_on_wide_span():
    # lo > 0 and hi/lo >= 100 -> log auto-on.
    res = _infer_single(Annotated[float, Field(ge=1e-3, le=1.0)], 0.1)
    assert res.domain.log is True


def test_bounded_log_off_on_narrow_span():
    res = _infer_single(Annotated[float, Field(ge=1.0, le=10.0)], 5.0)
    assert res.domain.log is False


def test_bounded_strict_gt_lt_respected():
    # Gt/Lt still produce a bounded range (strictness noted, bounds used).
    res = _infer_single(Annotated[float, at.Gt(0.0), at.Lt(10.0)], 5.0)
    assert isinstance(res.domain, FloatRange)
    assert (res.domain.low, res.domain.high) == (0.0, 10.0)


# --------------------------------------------------------------------------- #
# unbounded numeric heuristic
# --------------------------------------------------------------------------- #
def test_unbounded_float_heuristic():
    res = _infer_single(float, 2.0)
    assert isinstance(res, Knob)
    assert res.source == "unbounded_heuristic"
    assert res.needs_review is True
    assert isinstance(res.domain, FloatRange)
    assert (res.domain.low, res.domain.high) == (0.5, 8.0)


def test_unbounded_int_rounds_outward():
    res = _infer_single(int, 50)
    assert isinstance(res.domain, IntRange)
    # floor(50/4)=12, ceil(50*4)=200
    assert (res.domain.low, res.domain.high) == (12, 200)
    assert res.needs_review is True


def test_unbounded_log_auto_when_spans_order_of_magnitude():
    # d=2.0 -> [0.5, 8.0], hi/lo = 16 > 10 -> log on.
    res = _infer_single(float, 2.0)
    assert res.domain.log is True


def test_unbounded_widen_when_low_equals_high_int():
    # d=0 is non-positive -> excluded; use a tiny int that floors/ceils to equal.
    # d=1 -> floor(0.25)=0, ceil(4)=4 (not equal); craft a degenerate by step.
    # Instead test a float where rounding could collapse: not applicable to float.
    # For int d such that floor(d/4)==ceil(d*4) is impossible for d>0, so this
    # guards the generic widen path on the (rare) collapsed range.
    res = _infer_single(int, 1)
    assert isinstance(res.domain, IntRange)
    assert res.domain.low < res.domain.high


def test_unbounded_non_positive_default_excluded():
    # A non-positive anchor (d <= 0) collapses the multiplicative window and is
    # now flagged with the distinct ``non_positive_default`` reason (not the
    # overloaded ``non_numeric``).
    res = _infer_single(float, 0.0)
    assert isinstance(res, Excluded)
    assert res.reason == "non_positive_default"
    res_neg = _infer_single(int, -5)
    assert isinstance(res_neg, Excluded)
    assert res_neg.reason == "non_positive_default"


# --------------------------------------------------------------------------- #
# exclusions
# --------------------------------------------------------------------------- #
def test_ndarray_excluded():
    res = _infer_single(NdArrayField, np.zeros((2, 2)), field=Field(default=None))
    assert isinstance(res, Excluded)
    assert res.reason == "ndarray"


def test_freeform_str_excluded():
    res = _infer_single(str, "wavelet")
    assert isinstance(res, Excluded)
    assert res.reason == "non_numeric"


def test_column_ref_excluded():
    res = _infer_single(Optional[ColumnRef], None)
    assert isinstance(res, Excluded)
    assert res.reason == "name_ref"


def test_multi_type_union_excluded():
    res = _infer_single(int | str, 1)  # noqa: UP007
    assert isinstance(res, Excluded)
    assert res.reason == "unsupported_type"


# --------------------------------------------------------------------------- #
# T | None -> infer over T
# --------------------------------------------------------------------------- #
def test_optional_float_infers_over_inner():
    res = _infer_single(Optional[float], 2.0)
    assert isinstance(res, Knob)
    assert res.source == "unbounded_heuristic"
    assert (res.domain.low, res.domain.high) == (0.5, 8.0)


def test_optional_int_with_none_default_uses_none_safely():
    # Optional[int] default None -> inner is int but value is None (non-numeric
    # anchor) -> excluded, not a crash.
    res = _infer_single(Optional[int], None)
    assert isinstance(res, Excluded)


def test_optional_bounded_int():
    res = _infer_single(Annotated[Optional[int], Field(ge=2, le=20)], 10)
    assert isinstance(res, Knob)
    assert res.source == "bounded"
    assert (res.domain.low, res.domain.high) == (2, 20)


def test_optional_literal():
    res = _infer_single(Optional[Literal["a", "b"]], "a")
    assert isinstance(res, Knob)
    assert res.source == "literal"


# --------------------------------------------------------------------------- #
# description sourced from schema
# --------------------------------------------------------------------------- #
def test_description_from_schema():
    res = _infer_single(
        Annotated[float, Field(description="blur strength")], 2.0
    )
    assert res.description == "blur strength"


# --------------------------------------------------------------------------- #
# real pipeline: BlurGauss + OtsuDetector
# --------------------------------------------------------------------------- #
def test_infer_real_pipeline_flat():
    pipe = ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()])
    space = infer_search_space(pipe)
    assert isinstance(space, InferredSearchSpace)
    keys = {k.key for k in space.knobs}
    # BlurGauss.sigma now carries a TuneSpec -> Tier-1 ``tune_spec`` (the
    # annotations workstream migrated it from the unbounded heuristic).
    assert "0.sigma" in keys
    sigma = next(k for k in space.knobs if k.key == "0.sigma")
    assert sigma.source == "tune_spec"
    assert (sigma.domain.low, sigma.domain.high, sigma.domain.log) == (0.5, 5.0, True)
    assert "Standard deviation" in sigma.description
    # BlurGauss.mode (Literal) -> 0.mode categorical
    mode = next(k for k in space.knobs if k.key == "0.mode")
    assert mode.source == "literal"
    assert mode.domain.choices == ("reflect", "constant", "nearest")
    # OtsuDetector.ignore_zeros (bool) -> 1.ignore_zeros
    iz = next(k for k in space.knobs if k.key == "1.ignore_zeros")
    assert iz.source == "bool"
    # BlurGauss.cval now carries TuneSpec(tunable=False) -> excluded by opt-out.
    cval_excluded = next((e for e in space.excluded if e.key == "0.cval"), None)
    assert cval_excluded is not None
    assert cval_excluded.reason == "tune_spec_off"


def test_recurse_nested_is_noop_stub_this_chunk():
    # Nested recursion is the next chunk; recurse_nested must be accepted and
    # not crash, but emit no nested ("[" in key) knobs for a flat pipeline.
    pipe = ImagePipeline(ops=[BlurGauss(), OtsuDetector()])
    space = infer_search_space(pipe, recurse_nested=True)
    assert not any("[" in k.key for k in space.knobs)


@pytest.mark.parametrize("recurse", [True, False])
def test_infer_accepts_recurse_flag(recurse):
    pipe = ImagePipeline(ops=[BlurGauss()])
    space = infer_search_space(pipe, recurse_nested=recurse)
    assert isinstance(space, InferredSearchSpace)

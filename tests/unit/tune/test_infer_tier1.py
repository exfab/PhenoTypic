"""Tier-1 ``TuneSpec`` override + the ``⊆`` invariant (P3-4)."""
from __future__ import annotations

from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from phenotypic.tune import (
    Categorical,
    Excluded,
    FloatRange,
    IntRange,
    Knob,
    TuneSpec,
)
from phenotypic.tune._search_space._infer import _infer_field


def _infer_single(annotation, default):
    ns = {
        "__annotations__": {"f": annotation},
        "f": default,
        "model_config": ConfigDict(arbitrary_types_allowed=True),
    }
    op_cls = type("OneField", (BaseModel,), ns)
    op = op_cls()
    return _infer_field(op, 0, "f", op_cls.model_fields["f"])


# --------------------------------------------------------------------------- #
# TuneSpec wins over the heuristic
# --------------------------------------------------------------------------- #
def test_tune_spec_float_overrides_unbounded_heuristic():
    res = _infer_single(Annotated[float, TuneSpec(0.5, 5.0, log=True)], 2.0)
    assert isinstance(res, Knob)
    assert res.source == "tune_spec"
    assert res.needs_review is False
    assert isinstance(res.domain, FloatRange)
    assert (res.domain.low, res.domain.high, res.domain.log) == (0.5, 5.0, True)


def test_tune_spec_int_with_step():
    res = _infer_single(Annotated[int, TuneSpec(2, 20, step=2)], 10)
    assert isinstance(res.domain, IntRange)
    assert (res.domain.low, res.domain.high, res.domain.step) == (2, 20, 2)
    assert res.source == "tune_spec"


def test_tune_spec_tunable_false_excludes():
    res = _infer_single(Annotated[float, TuneSpec(tunable=False)], 4.0)
    assert isinstance(res, Excluded)
    assert res.reason == "tune_spec_off"


def test_tune_spec_categories_to_categorical():
    res = _infer_single(
        Annotated[Literal["reflect", "constant", "nearest"],
                  TuneSpec(categories=("reflect", "nearest"))],
        "reflect",
    )
    assert isinstance(res, Knob)
    assert res.source == "tune_spec"
    assert isinstance(res.domain, Categorical)
    assert res.domain.choices == ("reflect", "nearest")


def test_tune_spec_nested_under_optional_resolved():
    res = _infer_single(
        Annotated[Optional[float], TuneSpec(0.5, 5.0)], None
    )
    assert isinstance(res, Knob)
    assert res.source == "tune_spec"
    assert (res.domain.low, res.domain.high) == (0.5, 5.0)


def test_tune_spec_overrides_bool():
    # Even a bool field is overridden by an explicit categories TuneSpec.
    res = _infer_single(
        Annotated[bool, TuneSpec(categories=(True,))], True
    )
    assert isinstance(res, Knob)
    assert res.source == "tune_spec"
    assert res.domain.choices == (True,)


# --------------------------------------------------------------------------- #
# ⊆ invariant: TuneSpec must lie within co-located Field bounds
# --------------------------------------------------------------------------- #
def test_subset_invariant_low_below_ge_raises():
    with pytest.raises(ValueError, match=r"⊆|within|bound"):
        _infer_single(
            Annotated[float, Field(ge=1.0, le=10.0), TuneSpec(0.5, 5.0)], 2.0
        )


def test_subset_invariant_high_above_le_raises():
    with pytest.raises(ValueError, match=r"⊆|within|bound"):
        _infer_single(
            Annotated[float, Field(ge=0.0, le=4.0), TuneSpec(0.5, 5.0)], 2.0
        )


def test_subset_invariant_strict_gt_respected():
    # Field(gt=0.5) means low must be > 0.5; a TuneSpec low == 0.5 escapes.
    with pytest.raises(ValueError, match=r"⊆|within|bound"):
        _infer_single(
            Annotated[float, Field(gt=0.5, le=10.0), TuneSpec(0.5, 5.0)], 2.0
        )


def test_subset_invariant_within_bounds_ok():
    res = _infer_single(
        Annotated[float, Field(ge=0.1, le=10.0), TuneSpec(0.5, 5.0)], 2.0
    )
    assert isinstance(res, Knob)
    assert res.source == "tune_spec"
    assert (res.domain.low, res.domain.high) == (0.5, 5.0)


def test_subset_invariant_error_names_the_key():
    with pytest.raises(ValueError, match="0.f"):
        _infer_single(
            Annotated[float, Field(ge=1.0, le=10.0), TuneSpec(0.5, 5.0)], 2.0
        )


def test_categories_tune_spec_skips_subset_check():
    # A categories-only TuneSpec carries no numeric range, so the ⊆ numeric
    # check does not apply (and must not crash on the bound).
    res = _infer_single(
        Annotated[Literal["a", "b"], TuneSpec(categories=("a",))], "a"
    )
    assert isinstance(res, Knob)
    assert res.domain.choices == ("a",)


def test_inferred_targets_carry_op_class():
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.tune import infer_search_space

    pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    proposal = infer_search_space(pipe)
    sigma = next(
        k for k in proposal.knobs if k.target.op == 0 and k.target.key == "0.sigma"
    )
    assert sigma.target.op_class == "GaussianBlur"
    assert all(k.target.op_class for k in proposal.knobs)  # every knob stamped

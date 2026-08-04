"""Phase 3: ``infer_search_space`` stamps ``op_class`` on conditional parents.

When ``infer_search_space`` finalizes knobs it fills ``op_class`` on each knob's
own target (posture C). It now *also* stamps the ``conditional_on`` parent
targets, so a presence-conditional knob's gate resolves against the correct
parent op once presence-wrapping is enabled.

This is latent in v1 — no op sets ``_tune_optional``, so inference emits
``conditional_on=None`` for every knob — hence the helper ``_stamp_op_classes``
is exercised directly with a hand-built conditional knob.
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune import Categorical, Knob, infer_search_space
from phenotypic.tune._search_space._infer import _stamp_op_classes
from phenotypic.tune._search_space._targets import Param, Presence


def _ops() -> list:
    return [BlurGauss(), OtsuDetector()]


def test_stamp_fills_op_class_on_conditional_parent():
    # A knob on op 1 gated on a presence-parent at op 0, both initially unstamped.
    knob = Knob(
        target=Param(op=1, field="ignore_zeros"),
        domain=Categorical(choices=(True, False)),
        conditional_on=((Presence(op=0), True),),
    )
    stamped = _stamp_op_classes(knob, _ops())

    # The knob's own target is stamped (existing behavior) ...
    assert stamped.target.op_class == "OtsuDetector"
    # ... and the conditional parent target is now stamped too (the fix).
    assert stamped.conditional_on is not None
    parent_target, parent_value = stamped.conditional_on[0]
    assert parent_target.op_class == "BlurGauss"
    assert parent_value is True


def test_stamp_preserves_parent_value_and_count():
    knob = Knob(
        target=Param(op=1, field="ignore_zeros"),
        domain=Categorical(choices=(True, False)),
        conditional_on=(
            (Presence(op=0), True),
            (Param(op=0, field="sigma"), 2.0),
        ),
    )
    stamped = _stamp_op_classes(knob, _ops())
    assert stamped.conditional_on is not None
    assert len(stamped.conditional_on) == 2
    # Both parents are stamped against the op at their position (both op 0).
    for parent_target, _value in stamped.conditional_on:
        assert parent_target.op_class == "BlurGauss"
    # Values carry through unchanged.
    assert [v for _t, v in stamped.conditional_on] == [True, 2.0]


def test_stamp_leaves_unconditional_knob_untouched():
    knob = Knob(
        target=Param(op=0, field="sigma"),
        domain=Categorical(choices=(1.0, 2.0)),
    )
    stamped = _stamp_op_classes(knob, _ops())
    assert stamped.conditional_on is None
    # The target is still stamped (the conditional path is purely additive).
    assert stamped.target.op_class == "BlurGauss"


def test_stamp_parent_out_of_range_left_untouched():
    # ``with_op_class`` returns an out-of-range target unchanged (the TuningSpec
    # validator reports the range error later); the parent must not raise here.
    knob = Knob(
        target=Param(op=0, field="sigma"),
        domain=Categorical(choices=(1.0, 2.0)),
        conditional_on=((Presence(op=99), True),),
    )
    stamped = _stamp_op_classes(knob, _ops())
    assert stamped.conditional_on is not None
    assert stamped.conditional_on[0][0].op_class is None


def test_inference_keeps_conditional_on_none_in_v1():
    # End-to-end: v1 never presence-wraps, so every inferred knob's
    # conditional_on stays None (the stamping is a no-op on the None branch).
    pipe = ImagePipeline(ops=[BlurGauss(), OtsuDetector()])
    space = infer_search_space(pipe)
    assert all(k.conditional_on is None for k in space.knobs)
    # But every knob's own target is op_class-stamped.
    assert all(k.target.op_class is not None for k in space.knobs)

"""Nested-op overlay in ``build_pipeline`` (P3-5b).

Real nesting host: ``CompositeDetector.ops: List[OperationField | None]``.
A ``NestedKey`` (``"<pos>.ops[<i>].<leaf>"``) overlays a scalar field on
the leaf op occupying slot ``i``, rebuilds that leaf through the key-tagged
backstop, splices it back into the parent's list, and reconstructs the parent —
leaving sibling slots, sibling ops, and the base pipeline untouched.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.detect import (
    CompositeDetector,
    OtsuDetector,
    RoundPeaksDetector,
)
from phenotypic.enhance import GaussianBlur
from phenotypic.tune._evaluation._builder import build_pipeline


def _composite_base() -> ImagePipeline:
    """A pipeline whose sole op nests two detectors in a list field."""
    return ImagePipeline(ops=[
        CompositeDetector(ops=[
            OtsuDetector(ignore_zeros=False),       # ops[0]
            RoundPeaksDetector(),                   # ops[1]
        ]),
    ])


def test_nested_overlay_rebuilds_leaf_and_leaves_base_untouched():
    base = _composite_base()
    candidate = build_pipeline(base, {"0.ops[0].ignore_zeros": True})

    comp = candidate.get_ops()["CompositeDetector"]
    assert comp.ops[0].ignore_zeros is True
    # sibling slot is untouched
    assert type(comp.ops[1]).__name__ == "RoundPeaksDetector"
    # base is unmutated
    base_comp = base.get_ops()["CompositeDetector"]
    assert base_comp.ops[0].ignore_zeros is False


def test_nested_overlay_preserves_sibling_leaf_fields():
    base = _composite_base()
    candidate = build_pipeline(base, {"0.ops[0].ignore_borders": True})
    comp = candidate.get_ops()["CompositeDetector"]
    # only the addressed field changed; ignore_zeros stays at its base value
    assert comp.ops[0].ignore_borders is True
    assert comp.ops[0].ignore_zeros is False


def test_multiple_nested_overlays_on_same_field_fold_into_one_rebuild():
    base = _composite_base()
    candidate = build_pipeline(base, {
        "0.ops[0].ignore_zeros": True,
        "0.ops[1].split_merged": True,
    })
    comp = candidate.get_ops()["CompositeDetector"]
    assert comp.ops[0].ignore_zeros is True
    assert comp.ops[1].split_merged is True


def test_nested_overlay_combined_with_parent_scalar_field():
    base = _composite_base()
    candidate = build_pipeline(base, {
        "0.mode": "union",
        "0.ops[0].ignore_zeros": True,
    })
    comp = candidate.get_ops()["CompositeDetector"]
    assert comp.mode == "union"
    assert comp.ops[0].ignore_zeros is True


def test_nested_overlay_alongside_flat_op_in_pipeline():
    base = ImagePipeline(ops=[
        GaussianBlur(sigma=2.0),                    # position 0
        CompositeDetector(ops=[               # position 1
            OtsuDetector(ignore_zeros=False),
            RoundPeaksDetector(),
        ]),
    ])
    candidate = build_pipeline(base, {
        "0.sigma": 4.0,
        "1.ops[0].ignore_zeros": True,
    })
    ops = candidate.get_ops()
    assert ops["GaussianBlur"].sigma == 4.0
    assert ops["CompositeDetector"].ops[0].ignore_zeros is True


def test_nested_index_out_of_range_raises():
    base = _composite_base()
    with pytest.raises(IndexError):
        build_pipeline(base, {"0.ops[5].ignore_zeros": True})


def test_nested_overlay_on_none_slot_raises():
    base = ImagePipeline(ops=[
        CompositeDetector(ops=[OtsuDetector(), None]),
    ])
    with pytest.raises(ValueError, match="empty"):
        build_pipeline(base, {"0.ops[1].ignore_zeros": True})


def test_untouched_none_slot_passes_through():
    """A ``None`` slot that is not targeted survives the rebuild unchanged."""
    base = ImagePipeline(ops=[
        CompositeDetector(ops=[OtsuDetector(ignore_zeros=False), None]),
    ])
    candidate = build_pipeline(base, {"0.ops[0].ignore_zeros": True})
    comp = candidate.get_ops()["CompositeDetector"]
    assert comp.ops[0].ignore_zeros is True
    assert comp.ops[1] is None


def test_nested_leaf_validation_error_names_the_knob_key():
    """A bad nested value surfaces a ValidationError tagged with its knob key."""
    base = ImagePipeline(ops=[
        CompositeDetector(ops=[
            RoundPeaksDetector(),                   # ops[0]
            OtsuDetector(),
        ]),
    ])
    # thresh_method is a closed Literal set; an out-of-set value must fail
    # reconstruction of the leaf, tagged with the nested knob key.
    with pytest.raises(ValidationError, match=r"ops\[0\]\.thresh_method"):
        build_pipeline(base, {"0.ops[0].thresh_method": "not_a_method"})

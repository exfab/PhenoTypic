"""One-level nested-op recursion in ``infer_search_space`` (P3-5c).

Real host: ``CompositeDetector.detectors: List[OperationField | None]``. With
``recurse_nested=True`` (the default), inference emits ``pos.field[i].leaf``
knobs for each non-``None`` leaf op's scalar fields — depth-capped at 1 (a
nested op's *own* operation-valued fields are excluded). ``recurse_nested=False``
yields the flat-only proposal (no nested knobs).
"""
from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import (
    CompositeDetector,
    FilamentousFungiDetector,
    InoculumDetector,
    OtsuDetector,
    RoundPeaksDetector,
)
from phenotypic.enhance import GaussianBlur
from phenotypic.tune import Knob, infer_search_space


def _composite_pipe() -> ImagePipeline:
    return ImagePipeline(ops=[
        CompositeDetector(detectors=[
            OtsuDetector(ignore_zeros=False),   # detectors[0] — two bool fields
            RoundPeaksDetector(),               # detectors[1]
        ]),
    ])


def _knob(space, key: str) -> Knob:
    return next(k for k in space.knobs if k.key == key)


def _keys(space) -> set[str]:
    return {k.key for k in space.knobs}


def test_nested_recursion_emits_depth1_leaf_keys():
    space = infer_search_space(_composite_pipe())
    keys = _keys(space)
    # OtsuDetector leaf fields at detectors[0]
    assert "0.detectors[0].ignore_zeros" in keys
    assert "0.detectors[0].ignore_borders" in keys
    # RoundPeaksDetector leaf bool/enum fields at detectors[1]
    assert "0.detectors[1].split_merged" in keys
    assert "0.detectors[1].thresh_method" in keys


def test_nested_leaf_provenance_and_description_from_nested_class():
    space = infer_search_space(_composite_pipe())
    knob = _knob(space, "0.detectors[0].ignore_zeros")
    # a bool field resolves to the ``bool`` source
    assert knob.source == "bool"
    # description is sourced from the *nested* class's schema (OtsuDetector)
    assert isinstance(knob.description, str)


def test_nested_conditional_on_is_none_without_presence_wrapping():
    """v1: nested ops are never presence-wrapped, so conditional_on stays None."""
    space = infer_search_space(_composite_pipe())
    for knob in space.knobs:
        if knob.key.startswith("0.detectors["):
            assert knob.conditional_on is None


def test_depth_cap_excludes_nested_ops_own_operation_fields():
    """A nested op's own OperationField children are not recursed (depth cap 1)."""
    space = infer_search_space(_composite_pipe())
    # No key descends two ``[i]`` levels and no nested 'detectors' of a nested op.
    for key in _keys(space):
        # at most one ``[`` index segment in any emitted key
        assert key.count("[") <= 1


def test_recurse_nested_false_yields_flat_only():
    space = infer_search_space(_composite_pipe(), recurse_nested=False)
    keys = _keys(space)
    # no nested keys at all
    assert not any("[" in k for k in keys)
    # the parent's own scalar field is still inferred flat
    assert "0.min_overlap_ratio" in keys or "0.mode" in keys


def test_nested_recursion_additive_alongside_flat_op():
    pipe = ImagePipeline(ops=[
        GaussianBlur(sigma=2.0),                # position 0 — flat
        CompositeDetector(detectors=[           # position 1 — nested host
            OtsuDetector(ignore_zeros=False),
            RoundPeaksDetector(),
        ]),
    ])
    space = infer_search_space(pipe)
    keys = _keys(space)
    # flat knob for the GaussianBlur op survives unchanged
    assert "0.sigma" in keys
    # nested keys are prefixed with the parent's position (1)
    assert "1.detectors[0].ignore_zeros" in keys


def test_none_slot_is_skipped_in_recursion():
    pipe = ImagePipeline(ops=[
        CompositeDetector(detectors=[OtsuDetector(ignore_zeros=False), None]),
    ])
    space = infer_search_space(pipe)
    keys = _keys(space)
    assert "0.detectors[0].ignore_zeros" in keys
    # the None slot produces no knobs
    assert not any(k.startswith("0.detectors[1]") for k in keys)


def test_single_operation_field_is_excluded_not_emitted_as_unparsable_key():
    pipe = ImagePipeline(
        ops=[FilamentousFungiDetector(inoculum_detector=InoculumDetector())]
    )

    space = infer_search_space(pipe)
    keys = _keys(space)

    assert not any(key.startswith("0.inoculum_detector.") for key in keys)
    excluded = {e.key: e.reason for e in space.excluded}
    assert excluded["0.inoculum_detector"] == "unsupported_type"

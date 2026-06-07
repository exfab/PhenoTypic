"""Tests for ``InferredSearchSpace`` + ``Excluded`` value-models (P3-2)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from phenotypic.tune import (
    Categorical,
    Excluded,
    FloatRange,
    InferredSearchSpace,
    IntRange,
    Knob,
    SearchSpace,
)


def _knob(key: str, **kw) -> Knob:
    return Knob(key=key, domain=FloatRange(low=0.5, high=8.0), **kw)


def test_excluded_construction():
    ex = Excluded(key="0.kernel", reason="ndarray", field_type="ndarray")
    assert ex.key == "0.kernel"
    assert ex.reason == "ndarray"
    assert ex.field_type == "ndarray"


def test_excluded_frozen():
    ex = Excluded(key="0.kernel", reason="path", field_type="Path")
    with pytest.raises(ValidationError):
        ex.key = "x"  # type: ignore[misc]


def test_excluded_rejects_unknown_reason():
    with pytest.raises(ValidationError):
        Excluded(key="0.x", reason="not_a_reason", field_type="str")


def test_excluded_reason_members():
    # Every documented ExcludeReason is accepted.
    for reason in (
        "ndarray",
        "path",
        "name_ref",
        "non_numeric",
        "non_positive_default",
        "tune_spec_off",
        "unsupported_type",
    ):
        ex = Excluded(key="0.x", reason=reason, field_type="str")
        assert ex.reason == reason


def test_inferred_construction_and_counts():
    space = InferredSearchSpace(
        knobs=(
            _knob("0.sigma", source="unbounded_heuristic", needs_review=True),
            _knob("1.cutoff", source="bounded"),
        ),
        excluded=(
            Excluded(key="0.kernel", reason="ndarray", field_type="ndarray"),
        ),
    )
    assert space.n_knobs == 2
    assert space.n_excluded == 1
    assert space.n_needs_review == 1


def test_needs_review_true_when_any_knob_flagged():
    space = InferredSearchSpace(
        knobs=(_knob("0.sigma", needs_review=True),),
        excluded=(),
    )
    assert space.needs_review is True


def test_needs_review_false_when_clean():
    space = InferredSearchSpace(
        knobs=(
            Knob(
                key="0.flag",
                domain=Categorical(choices=(True, False)),
                source="bool",
            ),
        ),
        excluded=(),
    )
    assert space.needs_review is False


def test_needs_review_true_for_inference_blind_exclusion():
    # A field excluded for an inference-blind reason (non_numeric / unsupported)
    # raises the review flag even when no knob is flagged.
    space = InferredSearchSpace(
        knobs=(
            Knob(
                key="0.flag",
                domain=Categorical(choices=(True, False)),
                source="bool",
            ),
        ),
        excluded=(
            Excluded(key="0.scale", reason="non_numeric", field_type="float"),
        ),
    )
    assert space.needs_review is True


def test_needs_review_true_for_non_positive_default_exclusion():
    # ``non_positive_default`` (a numeric anchor <= 0 collapsing the [d/f, d·f]
    # window) is inference-blind — it must raise the review flag just like
    # ``non_numeric`` did before it was split out.
    space = InferredSearchSpace(
        knobs=(),
        excluded=(
            Excluded(
                key="0.k", reason="non_positive_default", field_type="float"
            ),
        ),
    )
    assert space.needs_review is True


def test_needs_review_false_for_benign_exclusion():
    # ndarray / path / tune_spec_off exclusions are deliberate, not blind —
    # they do not raise the review flag.
    space = InferredSearchSpace(
        knobs=(),
        excluded=(
            Excluded(key="0.kernel", reason="ndarray", field_type="ndarray"),
            Excluded(key="0.path", reason="path", field_type="Path"),
            Excluded(key="0.off", reason="tune_spec_off", field_type="float"),
        ),
    )
    assert space.needs_review is False


def test_to_search_space():
    inferred = InferredSearchSpace(
        knobs=(
            _knob("0.sigma", source="unbounded_heuristic", needs_review=True),
            Knob(key="1.min_size", domain=IntRange(low=12, high=200), source="bounded"),
        ),
        excluded=(
            Excluded(key="0.kernel", reason="ndarray", field_type="ndarray"),
        ),
    )
    space = inferred.to_search_space()
    assert isinstance(space, SearchSpace)
    assert space.keys() == ["0.sigma", "1.min_size"]
    # Excluded data never leaks into the optimizer-facing object.
    assert space.knobs[0].source == "unbounded_heuristic"


def test_json_round_trip():
    inferred = InferredSearchSpace(
        knobs=(
            _knob("0.sigma", source="unbounded_heuristic", needs_review=True),
            Knob(
                key="1.mode",
                domain=Categorical(choices=("reflect", "nearest")),
                source="literal",
            ),
        ),
        excluded=(
            Excluded(key="0.kernel", reason="ndarray", field_type="ndarray"),
            Excluded(key="0.scale", reason="non_numeric", field_type="float"),
        ),
    )
    dumped = inferred.model_dump_json()
    restored = InferredSearchSpace.model_validate_json(dumped)
    assert restored == inferred
    assert restored.n_knobs == 2
    assert restored.n_excluded == 2
    assert restored.needs_review is True


def test_frozen():
    inferred = InferredSearchSpace(knobs=(), excluded=())
    with pytest.raises(ValidationError):
        inferred.knobs = (_knob("0.x"),)  # type: ignore[misc]

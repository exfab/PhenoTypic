"""Wave-0 contract: the apply-time ``⊆`` backstop for validator-enforced bounds.

The ``⊆`` inference check (``test_annotation_subset_invariant``) is **blind** to
bounds enforced in a ``field_validator`` rather than ``Field(ge=, le=)`` — those
live in imperative code, not ``model_fields[name].metadata``. So a ``TuneSpec``
whose window exceeds a *validator*-enforced bound passes inference, and the real
guard is **apply time**: when ``build_pipeline`` reconstructs the leaf op with the
sampled value, the op's own ``pydantic.ValidationError`` fires.

This test pins that backstop's *contract*: ``build_pipeline`` must let the leaf
op's ``ValidationError`` surface, **wrapped** so the message names the offending
knob key **and** the op class (no new exception type — it stays a
``pydantic.ValidationError`` / ``ValueError`` subclass).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pytest
from pydantic import ValidationError, field_validator

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageEnhancer
from phenotypic.tune import TuneSpec, build_pipeline

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class _ValidatorBoundedEnhancer(ImageEnhancer):
    """A synthetic op whose ``TuneSpec`` exceeds a *validator*-enforced bound.

    ``strength``'s search window (``TuneSpec(0.0, 10.0)``) escapes the validator
    guard (``strength <= 5``) — and there is **no** mirrored ``Field`` bound, so
    inference's ``⊆`` check cannot see the conflict. A sampled value above ``5``
    must therefore be caught only at apply time.
    """

    strength: Annotated[float, TuneSpec(0.0, 10.0)] = 1.0

    @field_validator("strength")
    @classmethod
    def _check_strength(cls, value: float) -> float:
        if value > 5.0:
            raise ValueError("strength must be <= 5 (validator-enforced)")
        return value

    def _operate(self, image: "Image") -> "Image":  # pragma: no cover - never reached
        return image


def _base_pipeline() -> ImagePipeline:
    return ImagePipeline(ops=[_ValidatorBoundedEnhancer(strength=1.0)])


def test_in_bound_value_builds_cleanly():
    """A value inside the validator bound rebuilds without error."""
    built = build_pipeline(_base_pipeline(), {"0.strength": 3.0})
    op = list(built.get_ops().values())[0]
    assert op.strength == 3.0


def test_out_of_bound_value_raises_validation_error():
    """A sampled value past the validator bound surfaces a ``ValidationError``."""
    with pytest.raises(ValidationError):
        build_pipeline(_base_pipeline(), {"0.strength": 9.0})


def test_wrapped_message_names_knob_key_and_op_class():
    """The wrapped error names both the knob key and the op class for triage."""
    with pytest.raises(ValidationError) as excinfo:
        build_pipeline(_base_pipeline(), {"0.strength": 9.0})
    message = str(excinfo.value)
    assert "0.strength" in message
    assert "_ValidatorBoundedEnhancer" in message

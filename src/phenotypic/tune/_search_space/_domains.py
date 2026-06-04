"""Search-space domain types — a frozen pydantic discriminated union.

A tunable parameter's domain is one of ``Categorical`` / ``IntRange`` /
``FloatRange`` / ``Fixed``; each carries a ``kind`` literal so a ``Knob``'s
``domain`` field serializes and deserializes to the concrete type via the
``Domain`` discriminated union.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Closed set of domain discriminator tags (reused by the ``Domain`` union).
DomainKind = Literal["categorical", "int_range", "float_range", "fixed"]


class _DomainBase(BaseModel):
    """Shared config for every domain value-model (frozen, no extra fields)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Categorical(_DomainBase):
    """A finite set of choices (bools, enum/literal members, strings, ...).

    Args:
        choices: The non-empty tuple of allowed values. Lists are coerced to
            tuples so the model stays hashable and frozen.
    """

    kind: Literal["categorical"] = "categorical"
    choices: tuple[Any, ...]

    @model_validator(mode="after")
    def _non_empty(self) -> "Categorical":
        if len(self.choices) == 0:
            raise ValueError("Categorical requires at least one choice")
        return self


class IntRange(_DomainBase):
    """An integer range ``[low, high]`` with an optional step / log scale.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; must be ``>= low``.
        step: Stride between sampled integers (default ``1``).
        log: Whether to sample on a logarithmic scale (default ``False``).
    """

    kind: Literal["int_range"] = "int_range"
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "IntRange":
        if self.high < self.low:
            raise ValueError(f"IntRange high ({self.high}) < low ({self.low})")
        return self


class FloatRange(_DomainBase):
    """A float range ``[low, high]`` with an optional log scale.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; must be ``>= low``.
        log: Whether to sample on a logarithmic scale (default ``False``).
    """

    kind: Literal["float_range"] = "float_range"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatRange":
        if self.high < self.low:
            raise ValueError(f"FloatRange high ({self.high}) < low ({self.low})")
        return self


class Fixed(_DomainBase):
    """A pinned (non-tunable / frozen) value.

    Args:
        value: The single value this knob is pinned to.
    """

    kind: Literal["fixed"] = "fixed"
    value: Any


#: The discriminated union a ``Knob``'s ``domain`` field uses.
Domain = Annotated[
    Union[Categorical, IntRange, FloatRange, Fixed],
    Field(discriminator="kind"),
]

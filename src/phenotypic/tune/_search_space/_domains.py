"""Search-space domain types — a frozen pydantic discriminated union.

A tunable parameter's domain is one of ``Categorical`` / ``IntRange`` /
``FloatRange`` / ``Fixed``; each carries a ``kind`` literal so a ``Knob``'s
``domain`` field serializes and deserializes to the concrete type via the
``Domain`` discriminated union.
"""
from __future__ import annotations

from decimal import Decimal
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

    def values(self) -> list[int]:
        """The discrete integers in ``[low, high]`` stepped by :attr:`step`.

        The enumerable grid an ``IntRange`` exposes to the grid and random
        strategies (``range(low, high + 1, step)`` — ``high`` inclusive). The
        log scale is a *sampling* hint and does not change the enumerated set.

        Returns:
            The stepped integers from ``low`` to ``high`` inclusive.
        """
        return list(range(self.low, self.high + 1, self.step))


class FloatRange(_DomainBase):
    """A float range ``[low, high]`` with optional grid step / log scale.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; must be ``>= low``.
        step: Optional strict stride for grid enumeration. ``None`` means the
            range is continuous and not grid-enumerable. When the stride does
            not land on ``high``, ``high`` is appended exactly.
        log: Whether to sample on a logarithmic scale (default ``False``).
    """

    kind: Literal["float_range"] = "float_range"
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatRange":
        if self.high < self.low:
            raise ValueError(f"FloatRange high ({self.high}) < low ({self.low})")
        if self.step is not None and self.step <= 0:
            raise ValueError("FloatRange step must be positive when set")
        return self

    def values(self) -> list[float]:
        """The deterministic strict-stride grid values for a stepped range.

        Values start at ``low`` and advance by repeated ``step`` addition.
        ``high`` is appended exactly when the stride does not land on it.

        Returns:
            The inclusive float values from ``low`` to ``high``.

        Raises:
            ValueError: When this range is continuous (``step is None``).
        """
        if self.step is None:
            raise ValueError("continuous FloatRange has no enumerable values")
        if self.low == self.high:
            return [self.low]
        low = Decimal(str(self.low))
        high = Decimal(str(self.high))
        step = Decimal(str(self.step))
        values = [self.low]
        value = low + step
        while value < high:
            values.append(float(value))
            value += step
        if values[-1] != self.high:
            values.append(self.high)
        return values


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

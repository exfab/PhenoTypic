"""The ``TuneSpec`` per-field tuning-metadata marker.

``TuneSpec`` is an ``Annotated`` extra that mirrors the existing field-marker
pattern (``_OperationFieldMarker`` in ``tools_/typing_.py``): a frozen, slotted,
**non-pydantic** sentinel that is a complete no-op at runtime and is read *only*
by ``infer_search_space``, via ``op.model_fields[name].metadata`` (where pydantic
v2 stores ``Annotated`` extras).

It rides in an ``Annotated[T, TuneSpec(...)]`` chain::

    class GaussianBlur(ImageEnhancer):
        sigma:    Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
        truncate: Annotated[float, TuneSpec(tunable=False)]      = 4.0

At runtime ``sigma`` is still a plain ``float``; ``GaussianBlur(sigma=999.0)``
constructs exactly as before. The marker is the **search** domain, never the
**valid** domain — validity stays the job of pydantic ``Field(ge=, le=)``.
``__eq__``/``__hash__`` (over the full field tuple) let a duplicate in an
``Annotated`` chain de-dupe.
"""
from __future__ import annotations

from typing import Optional


class TuneSpec:
    """Per-field tuning-search metadata (Tier-1 inference override).

    A precise, opt-in marker the operation author attaches to a field; when
    present it is authoritative over the Tier-2 type heuristics. A
    ``TuneSpec(tunable=False)`` is the explicit "never tune this" escape hatch.

    Args:
        low: Inclusive lower search bound (positional). ``None`` leaves the
            bound to Tier-2 inference / the co-located ``Field`` constraint.
        high: Inclusive upper search bound (positional). ``None`` as ``low``.
        step: Discretization stride — integer step or quantized float.
            ``None`` means continuous (or unit-step for integers).
        log: Whether to sample on a logarithmic scale (default ``False``).
        categories: Override / subset the auto-derived categorical choices.
            A list is coerced to a tuple so the marker stays hashable.
        tunable: ``False`` excludes the field from tuning outright and
            short-circuits Tier-2 (default ``True``).

    Note:
        This is **not** a pydantic model — it is a plain slotted sentinel so it
        can sit in an ``Annotated`` chain without pydantic trying to validate
        it. It carries ``__eq__``/``__hash__`` over the full field tuple so a
        duplicate in an ``Annotated`` chain de-dupes (PEP 593 chain semantics).
    """

    __slots__ = ("low", "high", "step", "log", "categories", "tunable")

    low: Optional[float]
    high: Optional[float]
    step: Optional[float]
    log: bool
    categories: Optional[tuple]
    tunable: bool

    def __init__(
        self,
        low: Optional[float] = None,
        high: Optional[float] = None,
        *,
        step: Optional[float] = None,
        log: bool = False,
        categories: Optional[tuple] = None,
        tunable: bool = True,
    ) -> None:
        self.low = low
        self.high = high
        self.step = step
        self.log = log
        self.categories = tuple(categories) if categories is not None else None
        self.tunable = tunable

    def _as_tuple(self) -> tuple:
        return (
            self.low,
            self.high,
            self.step,
            self.log,
            self.categories,
            self.tunable,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TuneSpec):
            return NotImplemented
        return self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        return hash(self._as_tuple())

    def __repr__(self) -> str:
        return (
            f"TuneSpec(low={self.low!r}, high={self.high!r}, "
            f"step={self.step!r}, log={self.log!r}, "
            f"categories={self.categories!r}, tunable={self.tunable!r})"
        )

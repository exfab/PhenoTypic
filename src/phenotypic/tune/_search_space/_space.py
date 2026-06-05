"""The optimizer-facing search space: knobs + their domains."""
from __future__ import annotations

from typing import Any, Iterator, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from ._domains import Domain

#: Provenance of a knob's domain — a closed set (never a bare ``str``).
#: ``"manual"`` is the hand-authored default; the remaining tags are assigned by
#: ``infer_search_space`` (Phase 3) to record how each domain was derived. Phase 3
#: owns widening this alias if inference introduces further origins.
KnobSource = Literal[
    "manual",
    "tune_spec",
    "bool",
    "enum",
    "literal",
    "bounded",
    "unbounded_heuristic",
    "presence_optin",
]


class Knob(BaseModel):
    """One tunable parameter: a key, a domain, and optional provenance.

    Args:
        key: Position-index path identifying the parameter, e.g.
            ``"1.detectors[0].ignore_zeros"`` (the ``N.`` prefix is the
            operation's position in the pipeline).
        domain: The value space to search over (the ``Domain`` discriminated
            union — ``Categorical`` / ``IntRange`` / ``FloatRange`` / ``Fixed``).
        conditional_on: Parent presence conditions that gate this knob; the knob
            is active only when each ``(key, value)`` pair holds, e.g.
            ``(("0.GaussianBlur.__enabled__", True),)`` — define-by-run
            conditional nesting. ``None`` means unconditional.
        source: Provenance of the knob (a closed ``KnobSource`` set). Defaults to
            ``"manual"`` for hand-authored spaces; ``infer_search_space`` (Phase 3)
            populates it with the inference origin.
        needs_review: Whether a human should confirm this knob before tuning
            (set by inference for shaky guesses); defaults to ``False``.
        description: Human-readable description, auto-sourced from the owning
            class's ``model_json_schema()`` during inference; ``""`` by default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    domain: Domain
    conditional_on: Optional[tuple[tuple[str, Any], ...]] = None
    source: KnobSource = "manual"
    needs_review: bool = False
    description: str = ""

    def is_active(self, chosen: Mapping[str, Any]) -> bool:
        """Whether this knob's parent presence conditions hold in ``chosen``.

        An unconditional knob (``conditional_on is None``) is always active; a
        conditional knob is active only when **every** ``(parent_key, value)``
        pair in :attr:`conditional_on` matches what was already chosen this trial
        (define-by-run conditional nesting). The single predicate every strategy
        (grid / random / Optuna) gates conditional knobs by.

        Args:
            chosen: The parameter values already assigned this trial
                (``{key: value}``); a missing parent key never matches.

        Returns:
            ``True`` if the knob should be assigned given ``chosen``.
        """
        if self.conditional_on is None:
            return True
        return all(chosen.get(pkey) == pval for pkey, pval in self.conditional_on)


class SearchSpace(BaseModel):
    """The clean, optimizer-facing collection of tunable knobs.

    Args:
        knobs: The ordered tuple of ``Knob`` instances the optimizer searches.

    Note:
        ``__iter__`` is overridden to yield ``Knob`` instances (not pydantic's
        default ``(field_name, value)`` pairs), so ``dict(space)`` does **not**
        produce a model dict — use ``model_dump()`` for serialization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knobs: tuple[Knob, ...]

    def keys(self) -> list[str]:
        """Return the knob keys in declaration order."""
        return [k.key for k in self.knobs]

    def domain(self, key: str) -> Domain:
        """Return the domain for ``key``; raise ``KeyError`` if absent."""
        for k in self.knobs:
            if k.key == key:
                return k.domain
        raise KeyError(key)

    def __iter__(self) -> Iterator[Knob]:  # type: ignore[override]
        return iter(self.knobs)

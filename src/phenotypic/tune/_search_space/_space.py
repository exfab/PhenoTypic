"""The optimizer-facing search space: knobs + their domains."""
from __future__ import annotations

from typing import Any, Iterator, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from ._domains import Domain
from ._targets import KnobTarget, parse_key

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
    """One tunable parameter: a target, a domain, and optional provenance.

    Args:
        target: The structured ``KnobTarget`` identifying the parameter — a
            ``Param`` / ``Presence`` / ``Nested`` whose ``.key`` renders the
            canonical position-index path (e.g. ``"1.ops[0].ignore_zeros"``,
            where the ``N.`` prefix is the operation's pipeline position). A
            legacy string ``key="0.sigma"`` is still accepted and coerced to a
            target (string-preserving round-trip), so older call sites and frozen
            ``tuning_spec.json`` blobs keep working without migration.
        domain: The value space to search over (the ``Domain`` discriminated
            union — ``Categorical`` / ``IntRange`` / ``FloatRange`` / ``Fixed``).
        conditional_on: Parent presence conditions that gate this knob; the knob
            is active only when each ``(target, value)`` pair holds, e.g.
            ``((Presence(op=0, op_class="GaussianBlur"), True),)`` — define-by-run
            conditional nesting. A string parent (``"0.GaussianBlur.__enabled__"``)
            is coerced to a target. ``None`` means unconditional.
        source: Provenance of the knob (a closed ``KnobSource`` set). Defaults to
            ``"manual"`` for hand-authored spaces; ``infer_search_space`` (Phase 3)
            populates it with the inference origin.
        needs_review: Whether a human should confirm this knob before tuning
            (set by inference for shaky guesses); defaults to ``False``.
        description: Human-readable description, auto-sourced from the owning
            class's ``model_json_schema()`` during inference; ``""`` by default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: KnobTarget
    domain: Domain
    conditional_on: Optional[tuple[tuple[KnobTarget, Any], ...]] = None
    source: KnobSource = "manual"
    needs_review: bool = False
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_strings(cls, data: Any) -> Any:
        """Accept a legacy string ``key=`` and string ``conditional_on`` parents.

        ``key="0.sigma"`` is coerced to ``target=parse_key("0.sigma")`` and
        removed from the input (so ``extra="forbid"`` does not trip). A
        ``conditional_on`` entry whose parent is a string is parsed the same way;
        a dict (structured JSON) or a target instance passes through to the
        discriminated-union validator unchanged.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "key" in data and "target" not in data:
            data["target"] = parse_key(data.pop("key"))
        cond = data.get("conditional_on")
        if cond is not None:
            data["conditional_on"] = tuple(
                (parse_key(parent) if isinstance(parent, str) else parent, value)
                for parent, value in cond
            )
        return data

    @property
    def key(self) -> str:
        """The canonical key string of this knob's target (back-compat read)."""
        return self.target.key

    def is_active(self, chosen: Mapping[str, Any]) -> bool:
        """Whether this knob's parent presence conditions hold in ``chosen``.

        An unconditional knob (``conditional_on is None``) is always active; a
        conditional knob is active only when **every** ``(parent_target, value)``
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
        return all(chosen.get(ptarget.key) == pval for ptarget, pval in self.conditional_on)


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

    def targets(self) -> list[KnobTarget]:
        """Return the knob targets in declaration order."""
        return [k.target for k in self.knobs]

    def domain(self, key: str) -> Domain:
        """Return the domain for ``key``; raise ``KeyError`` if absent."""
        for k in self.knobs:
            if k.key == key:
                return k.domain
        raise KeyError(key)

    def __iter__(self) -> Iterator[Knob]:  # type: ignore[override]
        return iter(self.knobs)

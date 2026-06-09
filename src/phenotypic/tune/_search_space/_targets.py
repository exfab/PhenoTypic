"""Typed parameter-reference targets for search-space knobs.

A ``Knob`` addresses a pipeline parameter by a structured ``KnobTarget`` — one
of ``Param`` (a flat field on a top-level op), ``Presence`` (an op on/off
toggle), or ``Nested`` (a depth-1 nested-op leaf). Each renders the canonical
``.key`` string the engine already consumes (``build_pipeline`` is untouched),
so targets are a typed authoring/serialization layer over the existing keys.

The public surface lives in the ``phenotypic.tune.targets`` subpackage; these
classes are accessed as ``targets.Param(...)``. This module imports neither
``_space`` nor ``_infer`` (it sits below both, so there is no import cycle).
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

#: The presence-toggle dunder segment (an op on/off flag in the key grammar).
_ENABLED = "__enabled__"


class _TargetBase(BaseModel):
    """Shared config for every target value-model (frozen, no extra fields)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Param(_TargetBase):
    """A flat scalar field on the top-level op at ``op`` — key ``"<op>.<field>"``.

    Args:
        op: The op's position index in the pipeline.
        field: The scalar field name on that op.
        op_class: Optional class-name cross-check — when set, the ``TuningSpec``
            validator asserts the op at ``op`` is this class (posture C: always
            populated by discovery / inference).
    """

    kind: Literal["param"] = "param"
    op: int
    field: str
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical ``"<op>.<field>"`` key string."""
        return f"{self.op}.{self.field}"


class Presence(_TargetBase):
    """An op on/off toggle — key ``"<op>[.<Class>].__enabled__"``.

    Args:
        op: The op's position index.
        op_class: The op's class name; when set, renders the classed key form
            and is cross-checked by the ``TuningSpec`` validator.
    """

    kind: Literal["presence"] = "presence"
    op: int
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical presence key (classed when ``op_class`` is set)."""
        if self.op_class:
            return f"{self.op}.{self.op_class}.{_ENABLED}"
        return f"{self.op}.{_ENABLED}"


class Nested(_TargetBase):
    """A depth-1 nested-op leaf — key ``"<op>.<field>[<index>].<leaf>"``.

    Args:
        op: The parent (top-level) op's position index.
        field: The parent's operation-valued list field.
        index: The slot in that list.
        leaf: The scalar field on the nested op.
        op_class: Optional class-name cross-check of the *parent* op at ``op``.
    """

    kind: Literal["nested"] = "nested"
    op: int
    field: str
    index: int
    leaf: str
    op_class: Optional[str] = None

    @property
    def key(self) -> str:
        """The canonical ``"<op>.<field>[<index>].<leaf>"`` key string."""
        return f"{self.op}.{self.field}[{self.index}].{self.leaf}"


#: The discriminated union a ``Knob.target`` holds.
KnobTarget = Annotated[Union[Param, Presence, Nested], Field(discriminator="kind")]


def parse_key(key: str) -> "KnobTarget":
    """Parse a canonical key string into a structured ``KnobTarget``.

    The pipeline-free inverse of ``KnobTarget.key`` — a purely *structural*
    parse; op-range / field-existence checks happen later against the pipeline
    in the ``TuningSpec`` validator. ``op_class`` is recovered only from the
    classed presence form (``"0.GaussianBlur.__enabled__"``); flat and nested
    keys do not encode a class, so their ``op_class`` is ``None``.

    Args:
        key: A canonical key, e.g. ``"0.sigma"`` /
            ``"0.GaussianBlur.__enabled__"`` / ``"0.refiners[1].min_size"``.

    Returns:
        The matching ``Param`` / ``Presence`` / ``Nested``.

    Raises:
        ValueError: When the first segment is not an int position, or the key
            matches none of the three grammars.
    """
    parts = key.split(".")
    try:
        op = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"key {key!r} does not start with an int position") from exc

    # Nested: a "<field>[<index>]" segment selects the nested grammar.
    for i, segment in enumerate(parts[1:], start=1):
        if "[" in segment and segment.endswith("]"):
            field, _, idx = segment[:-1].partition("[")
            leaf = ".".join(parts[i + 1:])
            if not field or not idx.isdigit() or not leaf:
                raise ValueError(f"key {key!r} has a malformed nested segment")
            if "[" in leaf:
                # Mirror build_pipeline's canonical depth-1 cap: the leaf must
                # not carry a further "[i]" segment (e.g. "0.f[1].g[2].h").
                raise ValueError(
                    f"key {key!r} exceeds the nesting depth cap of 1 "
                    "(the leaf must not carry a further '[i]' segment)"
                )
            return Nested(op=op, field=field, index=int(idx), leaf=leaf)

    # Presence: trailing "__enabled__" (classed three-part or bare two-part).
    if parts[-1] == _ENABLED:
        if len(parts) == 3:
            if not parts[1]:
                # "0..__enabled__" — an empty class segment would render the bare
                # key, a non-round-tripping parse; reject it like build_pipeline.
                raise ValueError(f"presence key {key!r} has an empty class segment")
            return Presence(op=op, op_class=parts[1])
        if len(parts) == 2:
            return Presence(op=op)
        raise ValueError(f"presence key {key!r} is malformed")

    # Flat: "<op>.<field>".
    if len(parts) == 2:
        return Param(op=op, field=parts[1])
    raise ValueError(f"key {key!r} is not a recognised flat/presence/nested key")


def with_op_class(target: "KnobTarget", ordered_ops: list) -> "KnobTarget":
    """Return ``target`` with ``op_class`` filled from the op at ``target.op``.

    Posture C: discovery / inference always stamp ``op_class`` so every
    programmatic target is wrong-op cross-checked. An out-of-range ``op`` is
    returned untouched (the ``TuningSpec`` validator reports the range error).

    Args:
        target: The target to enrich.
        ordered_ops: The pipeline's ops in position order.

    Returns:
        A copy with ``op_class = type(ordered_ops[target.op]).__name__``, or the
        unchanged ``target`` when ``op`` is out of range.
    """
    if not 0 <= target.op < len(ordered_ops):
        return target
    return target.model_copy(
        update={"op_class": type(ordered_ops[target.op]).__name__}
    )

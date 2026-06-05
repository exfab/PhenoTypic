"""``infer_search_space`` — mine a pipeline's pydantic fields into a proposal.

This module implements the **flat, single-op** Tier-2 dispatch (the §4 heuristic
table of ``search-space-inference.md``) plus the Tier-1 ``TuneSpec`` override
(§3), and **one-level nested-op recursion** (§6): when ``recurse_nested=True``
(the default), an ``OperationField`` field — a single nested op or a list of
them — is recursed exactly one level, emitting ``"<pos>.<field>[<i>].<leaf>"``
(or ``"<pos>.<field>.<leaf>"`` for a single op) knobs. The recursion is strictly
additive; ``recurse_nested=False`` yields the flat-only proposal.

The inference core, :func:`_infer_field`, maps one operation field to either a
:class:`~phenotypic.tune.Knob` (a tunable domain) or an
:class:`~phenotypic.tune.Excluded` record (a field that could not / should not be
tuned). :func:`infer_search_space` walks a live ``ImagePipeline``'s ops in
position order, builds the flat ``"<pos>.<field>"`` keys, and assembles the
:class:`~phenotypic.tune.InferredSearchSpace` proposal.

Example:
    ``GaussianBlur.sigma`` carries a ``TuneSpec`` (the annotations workstream), so
    it resolves via Tier-1; ``ChanVeseDetector.mu`` is still un-annotated and
    exercises the Tier-2 unbounded heuristic ``[d/4, d·4]``.

    >>> from phenotypic import ImagePipeline
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import ChanVeseDetector
    >>> from phenotypic.tune import infer_search_space
    >>> pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), ChanVeseDetector()])
    >>> sigma = next(k for k in infer_search_space(pipe).knobs if k.key == "0.sigma")
    >>> (sigma.source, sigma.domain.low, sigma.domain.high)
    ('tune_spec', 0.5, 5.0)
    >>> mu = next(k for k in infer_search_space(pipe).knobs if k.key == "1.mu")
    >>> mu.source
    'unbounded_heuristic'
    >>> infer_search_space(pipe).needs_review
    True
"""
from __future__ import annotations

import enum
import math
import types
import typing
from pathlib import Path
from typing import Any, Final, Literal, Union, get_args, get_origin

import annotated_types as at
import numpy as np

from ._domains import Categorical, FloatRange, IntRange
from ._inferred import Excluded, ExcludeReason, InferredSearchSpace
from ._space import Knob
from ._targets import parse_key, with_op_class
from ._tune_spec import TuneSpec

#: Multiplicative half-window for the unbounded heuristic: ``[d/f, d·f]``.
_DEFAULT_UNBOUNDED_FACTOR: Final[float] = 4.0
#: Span ratio (high/low) above which a range auto-trips ``log=True`` — "spans
#: more than one order of magnitude" for the unbounded heuristic.
_LOG_SPAN_THRESHOLD: Final[float] = 10.0
#: Span ratio above which a *bounded* range auto-trips ``log=True`` (~100×).
_BOUNDED_LOG_SPAN_THRESHOLD: Final[float] = 100.0


def _is_union_origin(origin: Any) -> bool:
    """Return ``True`` for both ``typing.Union`` and PEP 604 unions."""
    return origin is Union or origin is types.UnionType


def _walk_metadata(annotation: Any) -> list[Any]:
    """Collect ``Annotated`` extras from anywhere in the annotation tree.

    pydantic surfaces only the *outermost* ``Annotated`` extras in
    ``model_fields[name].metadata``; a marker nested under ``Optional`` (the
    natural ``Optional[Annotated[float, TuneSpec(...)]]``) would be missed. This
    walks the tree — every ``__metadata__`` chain and ``get_args`` branch.

    Args:
        annotation: A (possibly wrapped / nested) type annotation.

    Returns:
        All ``Annotated`` extra objects found at any depth.
    """
    found: list[Any] = []
    found.extend(getattr(annotation, "__metadata__", ()))
    for arg in get_args(annotation):
        found.extend(_walk_metadata(arg))
    return found


def _has_marker(metadata: list[Any], name: str) -> bool:
    """Return ``True`` if any metadata object has class ``name``.

    Markers (``TuneSpec`` / ``_ColumnRefMarker`` / ``_OperationFieldMarker``)
    are matched by class *name* rather than ``isinstance`` so this module never
    has to import the GUI / tools markers — it reads only their presence.
    """
    return any(type(m).__name__ == name for m in metadata)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, is_multi_union)`` for a ``T | None`` annotation.

    For ``T | None`` (a union whose only non-``None`` member is ``T``) returns
    ``(T, False)``. For a multi-type union with no ``None`` collapse target
    returns ``(annotation, True)`` flagging it unsupported. Non-unions pass
    through as ``(annotation, False)``.
    """
    origin = get_origin(annotation)
    if not _is_union_origin(origin):
        return annotation, False
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0], False
    # A multi-type union (A | B, neither resolving to a single non-None T).
    return annotation, True


def _core_type(annotation: Any) -> Any:
    """Strip an outer ``Annotated`` wrapper, returning the bare type."""
    if get_origin(annotation) is typing.Annotated:
        return get_args(annotation)[0]
    return annotation


def _to_float(value: Any) -> float:
    """Coerce an ``annotated_types`` bound (a ``SupportsX`` value) to ``float``."""
    return float(value)


def _numeric_bounds(metadata: list[Any]) -> tuple[float | None, float | None]:
    """Extract ``(low, high)`` from ``annotated_types`` constraints.

    Reads ``Ge``/``Gt`` for the lower bound and ``Le``/``Lt`` for the upper,
    and unwraps an ``Interval``. Strictness is recorded only implicitly — the
    bound value itself is used as the (inclusive) search edge.

    Args:
        metadata: The field's ``annotated_types`` constraint objects.

    Returns:
        ``(low, high)`` where each is ``None`` when that side is unbounded.
    """
    low: float | None = None
    high: float | None = None
    for m in metadata:
        if isinstance(m, at.Interval):
            if m.ge is not None:
                low = _to_float(m.ge)
            if m.gt is not None:
                low = _to_float(m.gt)
            if m.le is not None:
                high = _to_float(m.le)
            if m.lt is not None:
                high = _to_float(m.lt)
        elif isinstance(m, at.Ge):
            low = _to_float(m.ge)
        elif isinstance(m, at.Gt):
            low = _to_float(m.gt)
        elif isinstance(m, at.Le):
            high = _to_float(m.le)
        elif isinstance(m, at.Lt):
            high = _to_float(m.lt)
    return low, high


def _schema_description(op: Any, field_name: str) -> str:
    """Return the field's docstring-derived description (``""`` if absent)."""
    try:
        props = type(op).model_json_schema().get("properties", {})
    except Exception:  # pragma: no cover - schema generation is robust here
        return ""
    return props.get(field_name, {}).get("description", "") or ""


def _bounded_knob(
    key: str,
    core: Any,
    low: float,
    high: float,
    *,
    description: str,
) -> Knob:
    """Build a ``bounded`` knob from explicit Field bounds."""
    is_int = core is int
    log = low > 0 and (high / low) >= _BOUNDED_LOG_SPAN_THRESHOLD
    if is_int:
        domain: Any = IntRange(low=int(low), high=int(high), log=log)
    else:
        domain = FloatRange(low=low, high=high, log=log)
    return Knob(
        target=parse_key(key),
        domain=domain,
        source="bounded",
        needs_review=False,
        description=description,
    )


def _unbounded_knob_or_excluded(
    key: str,
    core: Any,
    value: Any,
    *,
    factor: float,
    description: str,
    field_type: str,
) -> Knob | Excluded:
    """Apply the ``[d/f, d·f]`` heuristic, or exclude a non-positive anchor."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return Excluded(key=key, reason="non_numeric", field_type=field_type)
    d = float(value)
    if d <= 0:
        return Excluded(key=key, reason="non_numeric", field_type=field_type)

    low = d / factor
    high = d * factor
    is_int = core is int
    if is_int:
        ilow = math.floor(low)
        ihigh = math.ceil(high)
        if ilow == ihigh:  # widen a collapsed range so low < high
            ilow -= 1
            ihigh += 1
        log = ilow > 0 and (ihigh / ilow) > _LOG_SPAN_THRESHOLD
        domain: Any = IntRange(low=ilow, high=ihigh, log=log)
    else:
        log = low > 0 and (high / low) > _LOG_SPAN_THRESHOLD
        domain = FloatRange(low=low, high=high, log=log)
    return Knob(
        target=parse_key(key),
        domain=domain,
        source="unbounded_heuristic",
        needs_review=True,
        description=description,
    )


def _find_tune_spec(metadata: list[Any]) -> TuneSpec | None:
    """Return the first ``TuneSpec`` in ``metadata`` (or ``None``)."""
    for m in metadata:
        if isinstance(m, TuneSpec):
            return m
    return None


def _assert_subset(
    key: str, spec: TuneSpec, metadata: list[Any]
) -> None:
    """Enforce ``TuneSpec[low, high] ⊆ [ge/gt, le/lt]`` at inference time.

    Reads the four ``annotated_types`` bounds pydantic emits into the field
    metadata (``Ge``/``Gt``/``Le``/``Lt``) and respects strictness: ``low > gt``
    / ``low >= ge`` on the lower edge, ``high < lt`` / ``high <= le`` on the
    upper. A ``TuneSpec`` that escapes a co-located ``Field`` constraint raises
    a clear ``ValueError`` here (you cannot search where the value is invalid).

    Caveat — **validator-enforced bounds are invisible.** This project usually
    enforces numeric bounds in a ``field_validator``, not ``Field(ge=, le=)``;
    those bounds live in imperative code, never in ``model_fields[name]
    .metadata``, so this check cannot see them. A ``TuneSpec`` exceeding a
    *validator*-enforced bound passes inference and only fails later at apply /
    trial time — that apply-time backstop is the real guard.

    Args:
        key: The field's root-relative key (for the error message).
        spec: The ``TuneSpec`` carrying ``low``/``high``.
        metadata: The combined field + annotation-tree metadata.

    Raises:
        ValueError: If the ``TuneSpec`` range escapes the Field bounds.
    """
    if spec.low is None and spec.high is None:
        return
    # Expand any Interval into its component bounds so a single loop handles all.
    bounds: list[Any] = []
    for m in metadata:
        if isinstance(m, at.Interval):
            bounds.extend(_interval_to_bounds(m))
        else:
            bounds.append(m)

    for m in bounds:
        if isinstance(m, at.Ge) and spec.low is not None and spec.low < _to_float(m.ge):
            raise ValueError(
                f"TuneSpec on {key!r} is not ⊆ its Field bound: low "
                f"{spec.low} < ge {_to_float(m.ge)}"
            )
        if isinstance(m, at.Gt) and spec.low is not None and spec.low <= _to_float(m.gt):
            raise ValueError(
                f"TuneSpec on {key!r} is not ⊆ its Field bound: low "
                f"{spec.low} <= gt {_to_float(m.gt)} (strict)"
            )
        if isinstance(m, at.Le) and spec.high is not None and spec.high > _to_float(m.le):
            raise ValueError(
                f"TuneSpec on {key!r} is not ⊆ its Field bound: high "
                f"{spec.high} > le {_to_float(m.le)}"
            )
        if isinstance(m, at.Lt) and spec.high is not None and spec.high >= _to_float(m.lt):
            raise ValueError(
                f"TuneSpec on {key!r} is not ⊆ its Field bound: high "
                f"{spec.high} >= lt {_to_float(m.lt)} (strict)"
            )


def _interval_to_bounds(interval: at.Interval) -> list[Any]:
    """Expand an ``Interval`` into the equivalent ``Ge``/``Gt``/``Le``/``Lt`` set."""
    out: list[Any] = []
    if interval.ge is not None:
        out.append(at.Ge(interval.ge))
    if interval.gt is not None:
        out.append(at.Gt(interval.gt))
    if interval.le is not None:
        out.append(at.Le(interval.le))
    if interval.lt is not None:
        out.append(at.Lt(interval.lt))
    return out


def _resolve_tune_spec(
    key: str,
    spec: TuneSpec,
    core: Any,
    metadata: list[Any],
    *,
    description: str,
) -> Knob | Excluded:
    """Build a Tier-1 ``Knob`` (or ``Excluded``) from an explicit ``TuneSpec``.

    Args:
        key: The field's root-relative key.
        spec: The ``TuneSpec`` marker.
        core: The bare (Optional/Annotated-stripped) field type.
        metadata: The combined field + annotation-tree metadata (for the ⊆ check).
        description: The field's docstring-derived description.

    Returns:
        A ``Knob`` (``source="tune_spec"``, ``needs_review=False``), or an
        ``Excluded(reason="tune_spec_off")`` when ``tunable=False``.

    Raises:
        ValueError: If the ``TuneSpec`` range escapes a co-located Field bound.
    """
    if not spec.tunable:
        return Excluded(
            key=key, reason="tune_spec_off", field_type=_annotation_str(core)
        )

    # categories override -> Categorical (no numeric ⊆ check applies).
    if spec.categories is not None:
        return Knob(
            target=parse_key(key),
            domain=Categorical(choices=tuple(spec.categories)),
            source="tune_spec",
            needs_review=False,
            description=description,
        )

    _assert_subset(key, spec, metadata)

    if spec.low is None or spec.high is None:
        # An incomplete numeric TuneSpec (no full range, no categories) can't
        # form a domain — surface it rather than fabricate one.
        return Excluded(
            key=key, reason="non_numeric", field_type=_annotation_str(core)
        )

    if core is int:
        step = int(spec.step) if spec.step is not None else 1
        domain: Any = IntRange(
            low=int(spec.low), high=int(spec.high), step=step, log=spec.log
        )
    else:
        domain = FloatRange(low=spec.low, high=spec.high, log=spec.log)
    return Knob(
        target=parse_key(key),
        domain=domain,
        source="tune_spec",
        needs_review=False,
        description=description,
    )


def _infer_field(
    op: Any,
    position: int,
    field_name: str,
    field_info: Any,
    *,
    factor: float = _DEFAULT_UNBOUNDED_FACTOR,
) -> Knob | Excluded:
    """Map one operation field to a ``Knob`` or an ``Excluded`` record.

    A Tier-1 ``TuneSpec`` (when present) wins over the heuristics (added in
    P3-4); this is the Tier-2 type/constraint dispatch. The flat key is
    ``"<position>.<field_name>"``.

    Args:
        op: The operation instance (for the current value + schema description).
        position: The op's position in the pipeline (key prefix).
        field_name: The pydantic field name.
        field_info: ``type(op).model_fields[field_name]``.
        factor: Multiplicative half-window for the unbounded heuristic.

    Returns:
        A ``Knob`` (tunable) or an ``Excluded`` record.
    """
    key = f"{position}.{field_name}"
    annotation = field_info.annotation
    description = _schema_description(op, field_name)
    field_type = _annotation_str(annotation)
    value = getattr(op, field_name, field_info.default)

    # pydantic lifts Field/annotated_types constraints onto ``field_info.metadata``
    # and unwraps the outer ``Annotated``; markers nested deeper in the tree
    # (TuneSpec/ColumnRef under Optional) are found by walking the annotation.
    metadata = list(field_info.metadata) + _walk_metadata(annotation)

    # Name-ref marker (ColumnRef) — an open set, excluded.
    if _has_marker(metadata, "_ColumnRefMarker"):
        return Excluded(key=key, reason="name_ref", field_type=field_type)
    # Operation-valued field — handled by nested recursion (next chunk), not a
    # scalar knob here.
    if _has_marker(metadata, "_OperationFieldMarker"):
        return Excluded(key=key, reason="unsupported_type", field_type=field_type)

    # T | None -> infer over T; multi-type union -> unsupported.
    inner, is_multi_union = _strip_optional(annotation)
    if is_multi_union:
        return Excluded(key=key, reason="unsupported_type", field_type=field_type)

    core_inner = _core_type(_strip_optional(_core_type(inner))[0])

    # Tier 1: an explicit ``TuneSpec`` wins over the Tier-2 heuristics.
    spec = _find_tune_spec(metadata)
    if spec is not None:
        return _resolve_tune_spec(
            key, spec, core_inner, metadata, description=description
        )

    return _dispatch_core(
        key,
        core_inner,
        value,
        metadata,
        factor=factor,
        description=description,
        field_type=field_type,
    )


def _dispatch_core(
    key: str,
    core: Any,
    value: Any,
    metadata: list[Any],
    *,
    factor: float,
    description: str,
    field_type: str,
) -> Knob | Excluded:
    """Tier-2 type dispatch over the bare (Optional/Annotated-stripped) type."""
    # ndarray (raw or NdArrayField) — not scalar-tunable.
    if core is np.ndarray:
        return Excluded(key=key, reason="ndarray", field_type=field_type)
    # Paths — open set.
    if isinstance(core, type) and issubclass(core, Path):
        return Excluded(key=key, reason="path", field_type=field_type)

    # bool (check before int — bool is an int subclass).
    if core is bool:
        return Knob(
            target=parse_key(key),
            domain=Categorical(choices=(True, False)),
            source="bool",
            needs_review=False,
            description=description,
        )

    # Literal[...] -> Categorical(literal members).
    if get_origin(core) is Literal:
        return Knob(
            target=parse_key(key),
            domain=Categorical(choices=tuple(get_args(core))),
            source="literal",
            needs_review=False,
            description=description,
        )

    # Enum -> Categorical(members).
    if isinstance(core, type) and issubclass(core, enum.Enum):
        return Knob(
            target=parse_key(key),
            domain=Categorical(choices=tuple(core)),
            source="enum",
            needs_review=False,
            description=description,
        )

    # Numeric (int / float).
    if core in (int, float):
        low, high = _numeric_bounds(metadata)
        if low is not None and high is not None:
            return _bounded_knob(
                key, core, low, high, description=description
            )
        return _unbounded_knob_or_excluded(
            key,
            core,
            value,
            factor=factor,
            description=description,
            field_type=field_type,
        )

    # Free-form str (and anything else non-numeric / unrecognised).
    if core is str:
        # Open set — excluded. ``non_numeric`` is the only fitting closed-set
        # reason; it conservatively raises the proposal review flag.
        return Excluded(key=key, reason="non_numeric", field_type=field_type)

    return Excluded(key=key, reason="unsupported_type", field_type=field_type)


def _annotation_str(annotation: Any) -> str:
    """Render an annotation for the ``Excluded.field_type`` display field."""
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation)


def _is_recursable_op(value: Any) -> bool:
    """Return ``True`` for a leaf operation worth recursing into (not a pipeline).

    A nested ``OperationField`` value may be a single leaf operation, a nested
    ``ImagePipeline``, or ``None``. We recurse only into a leaf op — one that
    exposes ``model_fields`` but is **not** itself a pipeline (a pipeline has
    ``get_ops``, and recursing into its ops would be a different, deferred
    naming scheme). ``None`` slots and pipelines are skipped.
    """
    return (
        value is not None
        and hasattr(type(value), "model_fields")
        and not hasattr(value, "get_ops")
    )


def _reparent_key(child_key: str, prefix: str) -> str:
    """Rewrite a depth-0 child key ``"0.<leaf>"`` as ``"<prefix>.<leaf>"``.

    ``_infer_field`` always builds keys as ``"<position>.<field>"`` with the
    position it is handed. When recursing, we infer each nested leaf field at a
    throwaway position ``0`` and then splice the real nested prefix
    (``"1.detectors[0]"``) in front of the leaf name — so the canonical nested
    key becomes ``"1.detectors[0].<leaf>"``.
    """
    _, _, leaf = child_key.partition(".")
    return f"{prefix}.{leaf}"


def _recurse_into_op(
    leaf_op: Any,
    prefix: str,
    *,
    factor: float,
    conditional_on: tuple[tuple[str, Any], ...] | None,
) -> tuple[list[Knob], list[Excluded]]:
    """Infer one nested op's scalar fields (depth cap = 1).

    Applies the same Tier-1 → Tier-2 dispatch to each of ``leaf_op``'s fields,
    re-parents the resulting keys under ``prefix`` (``"<pos>.<field>[<i>]"`` or
    ``"<pos>.<field>"``), and attaches ``conditional_on`` to every emitted knob.
    The nested op's **own** operation-valued fields are excluded here — depth is
    capped at one level (no chaining).

    Args:
        leaf_op: The nested operation instance to recurse into.
        prefix: The root-relative path of the nested slot (e.g.
            ``"1.detectors[0]"``).
        factor: Multiplicative half-window for the unbounded heuristic.
        conditional_on: Parent-presence gate to stamp on each nested knob (a
            ``((key, value),)`` tuple), or ``None`` when the parent is not
            presence-wrapped.

    Returns:
        ``(knobs, excluded)`` for the nested op's scalar fields.
    """
    knobs: list[Knob] = []
    excluded: list[Excluded] = []
    for field_name, field_info in type(leaf_op).model_fields.items():
        result = _infer_field(leaf_op, 0, field_name, field_info, factor=factor)
        new_key = _reparent_key(result.key, prefix)
        if isinstance(result, Knob):
            knobs.append(
                result.model_copy(
                    update={
                        "target": parse_key(new_key),
                        "conditional_on": conditional_on,
                    }
                )
            )
        else:
            excluded.append(result.model_copy(update={"key": new_key}))
    return knobs, excluded


def _infer_nested_field(
    op: Any,
    position: int,
    field_name: str,
    *,
    factor: float,
) -> tuple[list[Knob], list[Excluded]]:
    """Recurse one level into an operation-valued field's live value.

    Reads ``op.<field_name>`` (a single nested op or a ``list`` of them) and
    recurses via :func:`_recurse_into_op`. List members are indexed
    (``<field>[<i>]``) and ``None`` slots / nested pipelines are skipped; a
    single op recurses under the bare field path (``<field>``).

    ``conditional_on`` ties a nested knob to the parent's ``__enabled__`` **only
    when the parent op is presence-wrapped** (``type(op)._tune_optional`` is
    truthy). In v1 nested ops are never presence-wrapped and top-level presence
    wrapping is not emitted by inference, so this resolves to ``None`` — kept
    explicit so a future presence layer needs no change here.

    Args:
        op: The parent operation instance owning the nested field.
        position: The parent's position in the pipeline.
        field_name: The operation-valued field name.
        factor: Multiplicative half-window for the unbounded heuristic.

    Returns:
        ``(knobs, excluded)`` from the one-level recursion.
    """
    value = getattr(op, field_name, None)
    conditional_on = _parent_presence_condition(op, position)

    knobs: list[Knob] = []
    excluded: list[Excluded] = []
    if isinstance(value, list):
        for index, member in enumerate(value):
            if not _is_recursable_op(member):
                continue  # skip None slots and nested pipelines (depth cap)
            prefix = f"{position}.{field_name}[{index}]"
            k, e = _recurse_into_op(
                member, prefix, factor=factor, conditional_on=conditional_on
            )
            knobs.extend(k)
            excluded.extend(e)
    elif _is_recursable_op(value):
        prefix = f"{position}.{field_name}"
        k, e = _recurse_into_op(
            value, prefix, factor=factor, conditional_on=conditional_on
        )
        knobs.extend(k)
        excluded.extend(e)
    return knobs, excluded


def _parent_presence_condition(
    op: Any, position: int
) -> tuple[tuple[str, Any], ...] | None:
    """Return the nested knob's ``conditional_on`` gate, or ``None``.

    A nested knob is gated on its parent's ``__enabled__`` toggle **only** when
    the parent op is presence-wrapped (``type(op)._tune_optional``). The live
    ``Knob.conditional_on`` type is ``tuple[tuple[str, Any], ...]`` (not the
    ``dict`` of the design sketch). In v1 no op sets ``_tune_optional`` and
    inference emits no top-level presence knob, so this returns ``None``.
    """
    if getattr(type(op), "_tune_optional", False):
        cls_name = type(op).__name__
        return ((f"{position}.{cls_name}.__enabled__", True),)
    return None


def infer_search_space(
    pipeline: Any,
    *,
    unbounded_factor: float = _DEFAULT_UNBOUNDED_FACTOR,
    recurse_nested: bool = True,
) -> InferredSearchSpace:
    """Mine a pipeline into a generous, reviewable ``InferredSearchSpace``.

    Walks the pipeline's ops in position order and infers each scalar field via
    the Tier-2 type heuristics (Tier-1 ``TuneSpec`` override added in P3-4).
    Knob keys are flat ``"<position>.<field>"`` paths.

    Args:
        pipeline: A live ``ImagePipeline`` whose ``get_ops()`` yields the ops in
            insertion order.
        unbounded_factor: Multiplicative half-window ``[d/f, d·f]`` for the
            unbounded numeric heuristic (default ``4.0`` → 16× span).
        recurse_nested: Recurse **exactly one level** into operation-valued
            (``OperationField``) fields — a single nested op or a list of them.
            On by default; ``False`` yields the flat-only proposal (no nested
            knobs), byte-identical to the pre-recursion space. The recursion is
            strictly additive: a pipeline with no nested ops is unchanged.

    Returns:
        The ``InferredSearchSpace`` proposal.
    """
    ops = list(pipeline.get_ops().values())
    knobs: list[Knob] = []
    excluded: list[Excluded] = []
    for position, op in enumerate(ops):
        for field_name, field_info in type(op).model_fields.items():
            result = _infer_field(
                op, position, field_name, field_info, factor=unbounded_factor
            )
            if isinstance(result, Knob):
                knobs.append(result)
            else:
                excluded.append(result)
            # One-level recursion into operation-valued fields. The flat pass
            # above records the field as ``Excluded(unsupported_type)``; the
            # nested leaves are emitted here (additive — keeps the marker record).
            if recurse_nested and _field_holds_operation(field_info):
                n_knobs, n_excluded = _infer_nested_field(
                    op, position, field_name, factor=unbounded_factor
                )
                knobs.extend(n_knobs)
                excluded.extend(n_excluded)
    knobs = [
        k.model_copy(update={"target": with_op_class(k.target, ops)}) for k in knobs
    ]
    return InferredSearchSpace(knobs=tuple(knobs), excluded=tuple(excluded))


def _field_holds_operation(field_info: Any) -> bool:
    """Return ``True`` if a field carries the ``_OperationFieldMarker``.

    Detected by walking the annotation tree (the marker may sit under
    ``Optional`` / inside a ``list``), matching by class *name* so this module
    never imports the ``tools_`` marker.
    """
    metadata = list(field_info.metadata) + _walk_metadata(field_info.annotation)
    return _has_marker(metadata, "_OperationFieldMarker")


__all__ = ["infer_search_space", "_infer_field", "ExcludeReason"]

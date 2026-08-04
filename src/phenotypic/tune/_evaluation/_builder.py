"""Turn a sampled parameter combo into a runnable ``ImagePipeline``.

The combo is a flat ``{root-relative-key: value}`` mapping (the same keys a
``SearchSpace`` knob carries; see master §5). ``build_pipeline`` clones the base
pipeline, overlays each key onto the op it addresses by **fresh reconstruction**
(full validation — byte-compatible with the legacy sweep's
``operation_class(**merged)``), and drops ops toggled off via ``__enabled__``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Union, cast

from pydantic import ValidationError
from pydantic_core import InitErrorDetails

from phenotypic import ImagePipeline
from phenotypic.sdk_._class_aliases import canonical_class_name

#: Matches a nested list-index segment ``name[i]`` (e.g. ``ops[0]``) and
#: captures the field name and the integer index. The depth cap is 1: a key may
#: carry **exactly one** such segment (a second ``[i]`` is a depth error).
_NESTED_SEGMENT_RE = re.compile(r"^(?P<field>[^\[\]]+)\[(?P<index>\d+)\]$")
#: Matches any segment that *opens* a bracket — used to detect a malformed or
#: second ``[i]`` segment after the first nested segment has been consumed.
_BRACKET_SEGMENT_RE = re.compile(r"\[")


@dataclass(frozen=True)
class FlatKey:
    """A scalar field on the op at ``position``: ``"<pos>.<field>"``."""

    position: int
    field: str


@dataclass(frozen=True)
class PresenceKey:
    """A presence toggle: ``"<pos>.__enabled__"`` or ``"<pos>.<Class>.__enabled__"``.

    ``cls_name`` is the class segment from the three-part form (validated against
    the op actually at ``position``), or ``None`` for the bare two-part form.
    """

    position: int
    cls_name: str | None


@dataclass(frozen=True)
class NestedKey:
    """A one-level nested-op field: ``"<pos>.<field>[<index>].<leaf>"``.

    Addresses ``ordered_ops[position].<field>[index].<leaf>`` — a scalar field on
    the operation occupying slot ``index`` of the parent's operation-valued list
    field. The depth cap is 1: ``leaf`` may not itself carry a ``[i]`` segment.
    """

    position: int
    field: str
    index: int
    leaf: str


#: The union of typed parse results ``_parse_key`` may return.
ParsedKey = Union[FlatKey, PresenceKey, NestedKey]

_FILAMENTOUS_SCENE_PARENTS = {"max_colony_radius_px", "min_branch_width_px"}
_FILAMENTOUS_DERIVED_FIELDS = {
    "gauss_sigma",
    "tile_size",
    "tile_overlap",
    "pct_min_wavelength",
    "mad_window",
    "path_dilation_radius",
    "snr_margin",
    "coherence_window_radius",
}


def _parse_key(key: str, ordered_ops: list) -> ParsedKey:
    """Resolve a combo key to a typed ``FlatKey`` / ``PresenceKey`` / ``NestedKey``.

    The grammar is additive over Phase 1: flat ``"<pos>.<field>"`` and presence
    ``"<pos>[.<Class>].__enabled__"`` keys parse exactly as before (projecting
    onto the same ``(position, field)`` / ``(position, "__enabled__")`` shape);
    a segment matching ``name[i]`` switches the key into the nested grammar.

    Disambiguation and limits:
        * A segment matching ``name[i]`` (e.g. ``ops[0]``) marks a nested
          key; the **depth cap is 1** — a second ``[i]`` segment raises a
          "depth" ``ValueError``.
        * A malformed bracket segment (``name[x]``, ``name[]``) or a nested key
          missing its leaf raises a clear ``ValueError``.

    Args:
        key: A root-relative combo key.
        ordered_ops: The base pipeline's ops in order (for bounds + class checks).

    Returns:
        A ``FlatKey``, ``PresenceKey``, or ``NestedKey``.

    Raises:
        IndexError: If the position is out of range.
        ValueError: If a presence key's class name does not match the op there,
            or for a malformed / over-deep nested key.
    """
    parts = key.split(".")
    position = int(parts[0])
    if not 0 <= position < len(ordered_ops):
        raise IndexError(
            f"combo key {key!r} targets position {position}, but the base "
            f"pipeline has {len(ordered_ops)} op(s)"
        )

    # Locate the first ``name[i]`` segment (if any) — its presence selects the
    # nested grammar. The remaining segments after it must contain no further
    # bracket (depth cap = 1).
    for seg_index, segment in enumerate(parts[1:], start=1):
        nested = _NESTED_SEGMENT_RE.match(segment)
        if nested is not None:
            return _parse_nested_key(key, position, parts, seg_index, nested)

    if parts[-1] == "__enabled__":
        if len(parts) == 3:
            expected_cls = parts[1]
            actual_cls = type(ordered_ops[position]).__name__
            if actual_cls != canonical_class_name(expected_cls):
                raise ValueError(
                    f"presence key {key!r} targets class {expected_cls!r}, but "
                    f"position {position} holds a {actual_cls!r}"
                )
            return PresenceKey(position=position, cls_name=expected_cls)
        return PresenceKey(position=position, cls_name=None)

    if len(parts) == 2:
        # Guard against a malformed bracket that escaped the nested regex
        # (e.g. ``"0.ops[x]"``): an unmatched ``[`` is never a flat field.
        if _BRACKET_SEGMENT_RE.search(parts[1]):
            raise ValueError(
                f"combo key {key!r} has a malformed nested segment {parts[1]!r} "
                "(expected ``field[<int>]``)"
            )
        return FlatKey(position=position, field=parts[1])

    # A 3+ part key with no ``name[i]`` and not a classed presence is malformed
    # (e.g. a stray bracket that did not match, or an unexpected dotted path).
    raise ValueError(
        f"combo key {key!r} is not a recognised flat, presence, or nested key"
    )


def _parse_nested_key(
    key: str,
    position: int,
    parts: list[str],
    seg_index: int,
    nested: re.Match[str],
) -> NestedKey:
    """Build a ``NestedKey`` from a matched ``name[i]`` segment (depth cap = 1).

    Args:
        key: The full combo key (for error messages).
        position: The already-parsed op position.
        parts: The dot-split key segments.
        seg_index: Index into ``parts`` of the matched ``name[i]`` segment.
        nested: The ``_NESTED_SEGMENT_RE`` match for that segment.

    Returns:
        The parsed ``NestedKey``.

    Raises:
        ValueError: For a second ``[i]`` (depth error) or a missing leaf.
    """
    leaf_parts = parts[seg_index + 1:]
    if not leaf_parts:
        raise ValueError(
            f"nested key {key!r} is missing its leaf field "
            "(expected ``<pos>.<field>[<int>].<leaf>``)"
        )
    # Depth cap = 1: no segment after the first ``[i]`` may open another bracket.
    for leaf_segment in leaf_parts:
        if _BRACKET_SEGMENT_RE.search(leaf_segment):
            raise ValueError(
                f"nested key {key!r} exceeds the nesting depth cap of 1 "
                f"(a second ``[i]`` segment {leaf_segment!r} is not supported)"
            )
    leaf = ".".join(leaf_parts)
    return NestedKey(
        position=position,
        field=nested.group("field"),
        index=int(nested.group("index")),
        leaf=leaf,
    )


def _rebuild_op(op: Any, overrides: dict[str, Any]) -> Any:
    """Return a fresh op of the same type with ``overrides`` applied.

    Reconstructs through the constructor (re-running validators) rather than
    mutating in place, so the result serializes byte-identically to a freshly
    constructed op — operations are immutable/keyword-only.

    Args:
        op: The base operation instance.
        overrides: Field name → new value.

    Returns:
        A new operation instance.
    """
    fields = {name: getattr(op, name) for name in type(op).model_fields}
    fields.update(overrides)
    return type(op)(**fields)


def _filamentous_auto_values(op: Any) -> dict[str, Any]:
    """Return old auto-derived values for a ``FilamentousFungiDetector``."""
    r = op.max_colony_radius_px
    w = op.min_branch_width_px
    mad_window = int(round(op._MAD_WINDOW_PER_W * w)) + 1
    if mad_window % 2 == 0:
        mad_window += 1
    return {
        "gauss_sigma": op._GAUSS_SIGMA_PER_R * r,
        "tile_size": int(round(op._TILE_SIZE_PER_R * r)),
        "tile_overlap": int(round(op._TILE_OVERLAP_PER_R * r)),
        "pct_min_wavelength": op._WAVELENGTH_PER_W * w,
        "mad_window": mad_window,
        "path_dilation_radius": max(1, int(round(op._PATH_DILATION_PER_W * w))),
        "snr_margin": max(2, int(round(op._SNR_MARGIN_PER_W * w))),
        "coherence_window_radius": int(round(op._COHERENCE_RADIUS_PER_W * w)),
    }


def _reset_filamentous_auto_derived_fields(
    op: Any, overrides: dict[str, Any]
) -> dict[str, Any]:
    """Reset auto-derived detector fields when their scene parent is tuned."""
    if type(op).__name__ != "FilamentousFungiDetector":
        return overrides
    if not (_FILAMENTOUS_SCENE_PARENTS & set(overrides)):
        return overrides

    auto_values = _filamentous_auto_values(op)
    updated = dict(overrides)
    for field in _FILAMENTOUS_DERIVED_FIELDS:
        if field in updated:
            continue
        if getattr(op, field) == auto_values[field]:
            updated[field] = None
    return updated


def _rebuild_op_or_raise_with_keys(
    op: Any, overrides: dict[str, Any], keys: list[str]
) -> Any:
    """Reconstruct ``op`` with ``overrides``; on failure, name the knob keys.

    The leaf op's own ``field_validator`` / ``Field`` bounds fire here during
    fresh reconstruction. This is the **apply-time ⊆ backstop**: the ``⊆``
    inference check is blind to *validator*-enforced bounds (they live in
    imperative code, not ``model_fields[name].metadata``), so an out-of-bound
    sampled value is only caught at this reconstruction site.

    A failing reconstruction re-raises the op's ``pydantic.ValidationError``
    **wrapped** so the message names the offending knob key(s) and the op class
    — no new exception type, the result is still a ``ValidationError`` (a
    ``ValueError`` subclass) carrying the original per-field errors.

    Args:
        op: The base operation instance being overlaid.
        overrides: Field name → sampled value for this position.
        keys: The root-relative knob keys (``"<pos>.<field>"``) for ``overrides``.

    Returns:
        A freshly reconstructed operation instance.

    Raises:
        ValidationError: Wrapped to prepend the knob key + op class.
    """
    try:
        return _rebuild_op(op, overrides)
    except ValidationError as exc:
        cls_name = type(op).__name__
        prefix = f"{', '.join(keys)} [{cls_name}]"
        augmented: list[dict[str, Any]] = []
        for err in exc.errors(include_url=False):
            ctx = dict(err.get("ctx") or {})
            ctx["error"] = f"{prefix}: {err['msg']}"
            augmented.append(
                {
                    "type": "value_error",
                    "loc": err["loc"],
                    "input": err.get("input"),
                    "ctx": ctx,
                }
            )
        raise ValidationError.from_exception_data(
            f"{prefix} (tuning overlay)",
            cast("list[InitErrorDetails]", augmented),
        ) from exc


def _rebuild_nested_field(
    parent: Any,
    position: int,
    field: str,
    slot_overrides: dict[int, dict[str, Any]],
) -> list[Any]:
    """Return a new value for ``parent.<field>`` with nested leaf overrides applied.

    For each ``(index, {leaf: value})`` entry, reads the live leaf op at
    ``parent.<field>[index]``, asserts it is present (loud error on a ``None``
    slot or out-of-range index), and rebuilds it through the **same** key-tagged
    backstop (:func:`_rebuild_op_or_raise_with_keys`) so a leaf-validator failure
    still surfaces with the offending knob key + op class. Untouched slots
    (including ``None`` slots) pass through unchanged.

    Depth cap = 1: leaf overrides are scalar fields on the nested op; the nested
    op's own operation-valued fields are never recursed into here.

    Args:
        parent: The parent operation instance owning the list field.
        position: The parent's position in the pipeline (for key tagging).
        field: The parent's operation-list field name (e.g. ``"ops"``).
        slot_overrides: ``{index: {leaf_field: value}}`` for this field.

    Returns:
        A fresh list to splice into the parent's ``field`` (the parent itself is
        reconstructed by the caller via :func:`_rebuild_op_or_raise_with_keys`).

    Raises:
        IndexError: If a nested index is out of range.
        ValueError: If the addressed slot is ``None`` (an empty/unfilled slot).
        ValidationError: When a nested leaf value violates the leaf op's own
            bounds — wrapped to name the knob key + leaf op class.
    """
    current = getattr(parent, field, None)
    if not isinstance(current, list):
        raise ValueError(
            f"nested key targets {type(parent).__name__}.{field}, which is not a "
            f"list field (got {type(current).__name__})"
        )
    new_list = list(current)
    for index, leaf_overrides in slot_overrides.items():
        if not 0 <= index < len(new_list):
            raise IndexError(
                f"nested key targets {position}.{field}[{index}], but that field "
                f"holds {len(new_list)} slot(s)"
            )
        leaf_op = new_list[index]
        if leaf_op is None:
            raise ValueError(
                f"nested key targets {position}.{field}[{index}], but that slot "
                "is empty (None) — there is no operation to tune there"
            )
        keys = [
            f"{position}.{field}[{index}].{leaf}" for leaf in leaf_overrides
        ]
        new_list[index] = _rebuild_op_or_raise_with_keys(
            leaf_op, leaf_overrides, keys
        )
    return new_list


def build_pipeline(base: ImagePipeline, params: dict[str, Any]) -> ImagePipeline:
    """Clone ``base``, overlay ``params``, and drop ``__enabled__=False`` ops.

    Args:
        base: The base pipeline embedded in the ``TuningSpec``.
        params: A flat combo (``{root-relative-key: value}``) from a strategy.

    Returns:
        A new ``ImagePipeline`` carrying the base's measurements/post/qc with the
        tuned operations.

    Raises:
        IndexError / ValueError: Propagated from key parsing (out-of-range
            position/index, class mismatch, empty nested slot, malformed key).
        ValidationError: When a sampled value violates the leaf op's own bounds
            (the apply-time ``⊆`` backstop), wrapped to name the knob key + op
            class — for both flat and nested leaves.
    """
    candidate = base.model_copy(deep=True)  # preserves meas/post/qc; isolates ops from base
    ordered_ops = list(candidate.get_ops().values())

    overrides: dict[int, dict[str, Any]] = {}
    enabled: dict[int, bool] = {}
    # Nested overrides keyed by (position, field) → {index: {leaf: value}}.
    nested: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    for key, value in params.items():
        parsed = _parse_key(key, ordered_ops)
        if isinstance(parsed, PresenceKey):
            enabled[parsed.position] = bool(value)
        elif isinstance(parsed, FlatKey):
            overrides.setdefault(parsed.position, {})[parsed.field] = value
        else:  # NestedKey
            (
                nested.setdefault((parsed.position, parsed.field), {})
                .setdefault(parsed.index, {})[parsed.leaf]
            ) = value

    new_ops: list[Any] = []
    for position, op in enumerate(ordered_ops):
        if not enabled.get(position, True):
            continue  # presence toggled off → drop the op
        op_overrides = dict(overrides.get(position, {}))
        # Fold any nested-field overrides for this op into a single rebuild.
        nested_for_op = {
            field: slot_map
            for (pos, field), slot_map in nested.items()
            if pos == position
        }
        for field, slot_map in nested_for_op.items():
            op_overrides[field] = _rebuild_nested_field(
                op, position, field, slot_map
            )
        op_overrides = _reset_filamentous_auto_derived_fields(op, op_overrides)
        if not op_overrides:
            # un-overridden ops come from `candidate` (the deep copy), never `base`
            new_ops.append(op)
            continue
        keys = [f"{position}.{field}" for field in op_overrides]
        new_ops.append(_rebuild_op_or_raise_with_keys(op, op_overrides, keys))

    candidate.set_ops(new_ops)
    return candidate

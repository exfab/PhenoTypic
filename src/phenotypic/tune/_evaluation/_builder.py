"""Turn a sampled parameter combo into a runnable ``ImagePipeline``.

The combo is a flat ``{root-relative-key: value}`` mapping (the same keys a
``SearchSpace`` knob carries; see master §5). ``build_pipeline`` clones the base
pipeline, overlays each key onto the op it addresses by **fresh reconstruction**
(full validation — byte-compatible with the legacy sweep's
``operation_class(**merged)``), and drops ops toggled off via ``__enabled__``.
"""
from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError
from pydantic_core import InitErrorDetails

from phenotypic import ImagePipeline


def _parse_key(key: str, ordered_ops: list) -> tuple[int, str]:
    """Resolve a combo key to ``(position, field)``.

    ``field`` is ``"__enabled__"`` for a presence toggle, otherwise a scalar
    field name. Presence keys carry the class name (``"0.GaussianBlur.__enabled__"``)
    which is validated against the op actually at that position.

    Args:
        key: A root-relative combo key.
        ordered_ops: The base pipeline's ops in order (for bounds + class checks).

    Returns:
        ``(position, field)``.

    Raises:
        IndexError: If the position is out of range.
        ValueError: If a presence key's class name does not match the op there.
        NotImplementedError: For nested keys (Phase 3).
    """
    parts = key.split(".")
    position = int(parts[0])
    if not 0 <= position < len(ordered_ops):
        raise IndexError(
            f"combo key {key!r} targets position {position}, but the base "
            f"pipeline has {len(ordered_ops)} op(s)"
        )

    if parts[-1] == "__enabled__":
        if len(parts) == 3:
            expected_cls = parts[1]
            actual_cls = type(ordered_ops[position]).__name__
            if actual_cls != expected_cls:
                raise ValueError(
                    f"presence key {key!r} targets class {expected_cls!r}, but "
                    f"position {position} holds a {actual_cls!r}"
                )
        return position, "__enabled__"

    if len(parts) == 2:
        return position, parts[1]

    raise NotImplementedError(
        f"nested overlay key {key!r} is not supported in Phase 1 "
        "(nested-op tuning lands with Phase 3 search-space inference)"
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


def build_pipeline(base: ImagePipeline, params: dict[str, Any]) -> ImagePipeline:
    """Clone ``base``, overlay ``params``, and drop ``__enabled__=False`` ops.

    Args:
        base: The base pipeline embedded in the ``TuningSpec``.
        params: A flat combo (``{root-relative-key: value}``) from a strategy.

    Returns:
        A new ``ImagePipeline`` carrying the base's measurements/post/qc with the
        tuned operations.

    Raises:
        IndexError / ValueError / NotImplementedError: Propagated from key parsing.
        ValidationError: When a sampled value violates the leaf op's own bounds
            (the apply-time ``⊆`` backstop), wrapped to name the knob key + op
            class.
    """
    candidate = base.model_copy(deep=True)  # preserves meas/post/qc; isolates ops from base
    ordered_ops = list(candidate.get_ops().values())

    overrides: dict[int, dict[str, Any]] = {}
    enabled: dict[int, bool] = {}
    for key, value in params.items():
        position, field = _parse_key(key, ordered_ops)
        if field == "__enabled__":
            enabled[position] = bool(value)
        else:
            overrides.setdefault(position, {})[field] = value

    new_ops: list[Any] = []
    for position, op in enumerate(ordered_ops):
        if not enabled.get(position, True):
            continue  # presence toggled off → drop the op
        op_overrides = overrides.get(position)
        if not op_overrides:
            # un-overridden ops come from `candidate` (the deep copy), never `base`
            new_ops.append(op)
            continue
        keys = [f"{position}.{field}" for field in op_overrides]
        new_ops.append(_rebuild_op_or_raise_with_keys(op, op_overrides, keys))

    candidate.set_ops(new_ops)
    return candidate

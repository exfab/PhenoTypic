"""Turn a sampled parameter combo into a runnable ``ImagePipeline``.

The combo is a flat ``{root-relative-key: value}`` mapping (the same keys a
``SearchSpace`` knob carries; see master §5). ``build_pipeline`` clones the base
pipeline, overlays each key onto the op it addresses by **fresh reconstruction**
(full validation — byte-compatible with the legacy sweep's
``operation_class(**merged)``), and drops ops toggled off via ``__enabled__``.
"""
from __future__ import annotations

from typing import Any

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

    new_ops = []
    for position, op in enumerate(ordered_ops):
        if not enabled.get(position, True):
            continue  # presence toggled off → drop the op
        op_overrides = overrides.get(position)
        # un-overridden ops come from `candidate` (the deep copy), never `base`
        new_ops.append(_rebuild_op(op, op_overrides) if op_overrides else op)

    candidate.set_ops(new_ops)
    return candidate

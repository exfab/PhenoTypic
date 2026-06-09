"""Discovery catalog: the structured set of tunable parameters in a pipeline.

``pipeline_targets`` re-surfaces ``infer_search_space``'s mining as per-parameter
descriptors (each target ``op_class``-stamped) for the GUI 6c form and the MCP
"what can I tune?" tool — the agent *selects* a target rather than authoring a
key. Imports ``_infer`` (and is NOT imported by ``_space``), so the
``_space -> _targets`` edge stays cycle-free.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ._domains import Categorical, Domain, FloatRange, IntRange
from ._infer import infer_search_space
from ._targets import KnobTarget, Nested, Param, Presence

#: The value-type of a tunable parameter (named ``value_type`` — not ``kind`` —
#: to avoid colliding with the target union's ``kind`` discriminator).
ValueType = Literal["float", "int", "bool", "categorical"]


class TunableParam(BaseModel):
    """One tunable parameter a pipeline exposes (a discovery descriptor).

    Args:
        target: The structured ``KnobTarget`` to reference it by (``op_class``
            always set).
        op_class: The class of the op at ``target.op``.
        value_type: The parameter's value type.
        default: The field's current value on the pipeline.
        suggested_domain: The inferred domain, or ``None`` if inference excluded
            it.
        description: The field's docstring / ``TuneSpec`` text.
        needs_review: Whether inference flagged the suggested domain for review.
    """

    model_config = ConfigDict(frozen=True)

    target: KnobTarget
    op_class: str
    value_type: ValueType
    default: Any
    suggested_domain: Optional[Domain]
    description: str
    needs_review: bool


def _value_type(domain: Domain) -> ValueType:
    if isinstance(domain, FloatRange):
        return "float"
    if isinstance(domain, IntRange):
        return "int"
    if isinstance(domain, Categorical) and all(isinstance(c, bool) for c in domain.choices):
        return "bool"
    return "categorical"


def _current_value(target: KnobTarget, ordered_ops: list) -> Any:
    op = ordered_ops[target.op]
    if isinstance(target, Param):
        return getattr(op, target.field, None)
    if isinstance(target, Presence):
        return True  # an op present in the base pipeline is enabled
    if isinstance(target, Nested):
        nested = getattr(op, target.field, None)
        if isinstance(nested, list) and 0 <= target.index < len(nested) and nested[target.index] is not None:
            return getattr(nested[target.index], target.leaf, None)
    return None


def pipeline_targets(pipeline: Any) -> list[TunableParam]:
    """The structured catalog of tunable parameters in ``pipeline``.

    Built on ``infer_search_space`` (each knob's target already ``op_class``-
    stamped); each knob becomes a ``TunableParam`` carrying the field's current
    value, inferred domain, value type, description, and review flag.

    Args:
        pipeline: A live ``ImagePipeline``.

    Returns:
        One ``TunableParam`` per inferred knob, in proposal order.
    """
    proposal = infer_search_space(pipeline)
    ordered_ops = list(pipeline.get_ops().values())
    return [
        TunableParam(
            target=knob.target,
            op_class=knob.target.op_class or type(ordered_ops[knob.target.op]).__name__,
            value_type=_value_type(knob.domain),
            default=_current_value(knob.target, ordered_ops),
            suggested_domain=knob.domain,
            description=knob.description,
            needs_review=knob.needs_review,
        )
        for knob in proposal.knobs
    ]

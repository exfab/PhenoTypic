"""Phase 3: enum categoricals store member *values*, not members.

The search-space inference enum branch (``_dispatch_core``'s ``"enum"`` case)
stores ``choices=tuple(m.value for m in core)`` so a ``Categorical`` domain stays
JSON-native: ``model_dump(mode="json")`` → ``model_validate`` round-trips to the
same Python type. Storing the enum *member* instead would serialize to a bare
string and reload as ``str``, silently changing the domain's element type. The
build path re-applies a chosen value through the op constructor, whose
``field_validator`` coerces the value back to the enum.
"""
from __future__ import annotations

import enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from phenotypic.tune import (
    Categorical,
    Knob,
    SearchSpace,
)
from phenotypic.tune._search_space._infer import _infer_field
from phenotypic.tune._search_space._targets import Param


class _Mode(enum.Enum):
    """A string-valued enum standing in for a real operation parameter."""

    FAST = "fast"
    PRECISE = "precise"


def _infer_enum_knob(annotation, default) -> Knob:
    """Infer a single enum-typed field at position 0 and return the knob."""
    ns = {
        "__annotations__": {"f": annotation},
        "f": default,
        "model_config": ConfigDict(arbitrary_types_allowed=True),
    }
    op_cls = type("OneEnumField", (BaseModel,), ns)
    result = _infer_field(op_cls(), 0, "f", op_cls.model_fields["f"])
    assert isinstance(result, Knob)
    return result


def test_enum_branch_stores_values_not_members():
    knob = _infer_enum_knob(_Mode, _Mode.FAST)
    assert knob.source == "enum"
    assert isinstance(knob.domain, Categorical)
    # Values, not members — every choice is a plain string.
    assert set(knob.domain.choices) == {"fast", "precise"}
    assert all(isinstance(c, str) for c in knob.domain.choices)
    assert not any(isinstance(c, enum.Enum) for c in knob.domain.choices)


def test_optional_enum_still_stores_values():
    knob = _infer_enum_knob(Optional[_Mode], _Mode.PRECISE)
    assert knob.source == "enum"
    assert set(knob.domain.choices) == {"fast", "precise"}


def test_searchspace_enum_categorical_survives_json_roundtrip():
    # Wrap the inferred enum knob in a SearchSpace and round-trip via JSON.
    knob = _infer_enum_knob(_Mode, _Mode.FAST).model_copy(
        update={"target": Param(op=0, field="mode")}
    )
    space = SearchSpace(knobs=(knob,))

    blob = space.model_dump_json()
    # The serialized form carries the raw string values (JSON-native), not a
    # Python ``repr`` of an enum member.
    assert '"fast"' in blob
    assert '"precise"' in blob
    assert "_Mode" not in blob

    restored = SearchSpace.model_validate_json(blob)
    restored_choices = restored.knobs[0].domain.choices
    # Values intact and still plain strings (not coerced into anything lossy).
    assert set(restored_choices) == {"fast", "precise"}
    assert all(isinstance(c, str) for c in restored_choices)


def test_chosen_value_rebuilds_the_op_via_field_validator():
    """The build path applies a chosen string and the op coerces it back.

    Mirrors how a real operation rebuilds under ``build_pipeline``: a freshly
    constructed op runs its ``field_validator``, which coerces the JSON-native
    string the categorical stored back into the enum type. Storing the value
    (not the member) is what makes the round-tripped choice a valid constructor
    input here.
    """

    class _Op(BaseModel):
        mode: Annotated[_Mode, ...] = _Mode.FAST

        @field_validator("mode", mode="before")
        @classmethod
        def _coerce_mode(cls, value: object) -> _Mode:
            return value if isinstance(value, _Mode) else _Mode(value)

    knob = _infer_enum_knob(_Mode, _Mode.FAST)
    chosen = sorted(knob.domain.choices)[0]  # "fast" — a bare string
    rebuilt = _Op(mode=chosen)
    assert rebuilt.mode is _Mode.FAST
    assert isinstance(rebuilt.mode, _Mode)

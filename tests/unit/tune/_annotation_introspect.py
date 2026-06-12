"""Shared annotation-introspection helpers for the operation-annotation gates.

The three Wave-0 contract tests (``test_annotation_subset_invariant``,
``test_annotation_coverage``, ``test_annotation_back_compat``) all need to walk
the ``detect/`` + ``enhance/`` ``__all__`` classes and classify each field's
annotation tree. This module centralises that walk so the tests agree on the
**denominator** (which fields are numeric-tunable) and on how a ``TuneSpec`` /
``Field`` bound is detected — exactly mirroring ``infer_search_space``'s own
metadata reading (``field_info.metadata`` + an annotation-tree walk for markers
nested under ``Optional``).
"""
from __future__ import annotations

import enum
import types
import typing
from typing import Any, Iterator, Literal, Union, get_args, get_origin

import annotated_types as at
import numpy as np

import phenotypic.correction as _correction
import phenotypic.detect as _detect
import phenotypic.enhance as _enhance
import phenotypic.grid as _grid
import phenotypic.refine as _refine
from phenotypic.tune import TuneSpec

#: The operation families the annotations workstream covers. v1 shipped
#: ``detect`` + ``enhance``; the ``refine`` / ``grid`` / ``correction`` pass
#: (DEFERRED-WORK.md §3) brought the remaining families into the denominator.
ANNOTATED_MODULES = (_detect, _enhance, _refine, _grid, _correction)


def iter_annotated_classes() -> Iterator[type]:
    """Yield every operation class in the in-scope families' ``__all__``.

    A family's ``__all__`` may legitimately export non-operation symbols — e.g.
    ``phenotypic.grid`` exports ``CenteredAutoGridFinderFallbackWarning`` (a plain
    ``UserWarning`` subclass) so the GUI/``from_json`` can discover the warning
    category. Only pydantic operation models carry ``model_fields``, so restrict
    the denominator to those and skip the rest.
    """
    for module in ANNOTATED_MODULES:
        for name in module.__all__:
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, "model_fields"):
                yield obj


def walk_metadata(annotation: Any) -> list[Any]:
    """Collect ``Annotated`` extras from anywhere in the annotation tree.

    pydantic surfaces only the *outermost* ``Annotated`` extras in
    ``model_fields[name].metadata``; a marker nested under ``Optional`` would be
    missed. This walks the whole tree (matching ``_infer._walk_metadata``).
    """
    found: list[Any] = list(getattr(annotation, "__metadata__", ()))
    for arg in get_args(annotation):
        found.extend(walk_metadata(arg))
    return found


def field_metadata(field_info: Any) -> list[Any]:
    """The combined field + annotation-tree metadata for one field."""
    return list(field_info.metadata) + walk_metadata(field_info.annotation)


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def _strip_annotated(annotation: Any) -> Any:
    if get_origin(annotation) is typing.Annotated:
        return get_args(annotation)[0]
    return annotation


def core_type(annotation: Any) -> Any:
    """Return the bare type after stripping ``Annotated`` and ``Optional``.

    Multi-type unions (``A | B`` with neither resolving to a single non-``None``
    member) return the sentinel string ``"MULTI_UNION"`` so callers can exclude
    them — matching ``infer_search_space``'s unsupported-union handling.
    """
    annotation = _strip_annotated(annotation)
    if _is_union(get_origin(annotation)):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
        else:
            return "MULTI_UNION"
    return _strip_annotated(annotation)


def _has_marker(metadata: list[Any], name: str) -> bool:
    return any(type(m).__name__ == name for m in metadata)


def is_numeric_tunable(field_info: Any) -> bool:
    """Return ``True`` when a field is part of the coverage denominator.

    The denominator is **numeric** (``int`` / ``float``) tunable fields, with
    these explicit exclusions (per the workstream decision):

    - closed-set ``Literal`` / ``Enum`` / ``bool`` fields,
    - multi-type unions (e.g. ``str | float``),
    - ``OperationField`` / ``NdArrayField`` / container / ``ClassVar`` fields,
    - free-form ``str`` / path fields.
    """
    annotation = field_info.annotation
    metadata = field_metadata(field_info)
    core = core_type(annotation)

    if core == "MULTI_UNION":
        return False
    if _has_marker(metadata, "_OperationFieldMarker"):
        return False
    if _has_marker(metadata, "_NdArrayMarker"):
        return False
    if core is np.ndarray:
        return False
    if core is bool:
        return False
    if get_origin(core) is Literal:
        return False
    if isinstance(core, type) and issubclass(core, enum.Enum):
        return False
    return core in (int, float)


def tune_spec_of(field_info: Any) -> TuneSpec | None:
    """Return the field's ``TuneSpec`` marker (or ``None``)."""
    for m in field_metadata(field_info):
        if isinstance(m, TuneSpec):
            return m
    return None


def has_field_bound(field_info: Any) -> bool:
    """Return ``True`` if the field carries an ``annotated_types`` bound."""
    return any(
        isinstance(m, (at.Ge, at.Gt, at.Le, at.Lt, at.Interval))
        for m in field_metadata(field_info)
    )


def is_covered(field_info: Any) -> bool:
    """A field is *covered* by a ``TuneSpec``, a ``Field`` bound, or both.

    ``TuneSpec(tunable=False)`` counts as covered (an explicit "never tune this"
    decision), exactly like a ``TuneSpec`` with a search window.
    """
    return tune_spec_of(field_info) is not None or has_field_bound(field_info)


def field_key(cls: type, field_name: str) -> str:
    """The canonical ``"<ClassName>.<field>"`` key used by the gates."""
    return f"{cls.__name__}.{field_name}"


def iter_numeric_tunable_fields() -> Iterator[tuple[type, str, Any]]:
    """Yield ``(cls, field_name, field_info)`` for the whole denominator."""
    for cls in iter_annotated_classes():
        for field_name, field_info in cls.model_fields.items():
            if is_numeric_tunable(field_info):
                yield cls, field_name, field_info

"""Adapt scorer pydantic models to the shared operation param form."""
from __future__ import annotations

from typing import Type

from phenotypic.gui._operation_registry import OperationInfo, OperationRegistry


def scorer_operation_info(scorer_cls: Type) -> OperationInfo:
    """Build an :class:`OperationInfo` for a scorer's editable params."""
    registry = OperationRegistry()
    params = registry._extract_parameters(scorer_cls)
    return OperationInfo(
        cls=scorer_cls,
        name=scorer_cls.__name__,
        category="scorer",
        module=scorer_cls.__module__,
        docstring=scorer_cls.__doc__ or "",
        parameters=params,
    )

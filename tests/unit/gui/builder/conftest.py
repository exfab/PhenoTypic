"""Shared fixtures + helpers for the Phase 1 builder DAG test suite.

Hosts the small primitives that multiple ``test_*.py`` modules in this
directory need:

* :class:`_FakeRegistry` — a minimal registry stand-in honouring the
  ``.get(name)`` contract used by the validation module.
* :func:`_make_param` — a :class:`ParamInfo` builder with only the
  keyword arguments the validation rules care about.
* :func:`_make_op_info` — a :class:`OperationInfo` builder with a stub
  class object, used by every test module that seeds a fake registry.
* :func:`empty_registry` (pytest fixture) — monkeypatches
  ``phenotypic.gui.builder._validation.get_registry`` so registry-driven
  rules (Rule 3 / Rule 7) test against a stable, isolated surface.

Defined here (not in individual test modules) so the validation suite
and the recovery suite share one source of truth — see Phase 1
"reuse" review for the original divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from phenotypic.gui._operation_registry import OperationInfo, ParamInfo


@dataclass
class _FakeRegistry:
    """Minimal registry stand-in honouring the ``.get(name)`` contract."""

    ops: Dict[str, OperationInfo] = field(default_factory=dict)

    def get(self, name: str) -> Optional[OperationInfo]:
        return self.ops.get(name)


def _make_param(
    name: str,
    *,
    has_default: bool,
    is_operation: bool = False,
    is_pipeline: bool = False,
    is_list: bool = False,
    default: Any = None,
) -> ParamInfo:
    """Construct a ``ParamInfo`` with as few keyword arguments as possible.

    Args:
        name: Parameter name.
        has_default: Whether the parameter has a default; gates Rule 3.
        is_operation: ``True`` if the type hint resolves to an
            ``ImageOperation`` subclass.
        is_pipeline: ``True`` if the type hint accepts ``ImagePipeline``.
        is_list: ``True`` for list-typed parameters.
        default: Default value to record on ``ParamInfo.default`` — the
            registry coerces missing defaults to ``None``, so this is the
            value field reads even when ``has_default`` is ``False``.

    Returns:
        Fully-populated :class:`ParamInfo` instance.
    """

    return ParamInfo(
        name=name,
        type_hint=Any,
        default=default,
        has_default=has_default,
        is_operation=is_operation,
        is_pipeline=is_pipeline,
        is_optional=False,
        is_list=is_list,
    )


def _make_op_info(
    cls_name: str,
    parameters: Optional[Dict[str, ParamInfo]] = None,
    *,
    category: str = "Enhancer",
) -> OperationInfo:
    """Build an :class:`OperationInfo` shell with the right shape for tests.

    Used by ``test_validation.py`` and ``test_dispatch.py`` (both seed a
    fake registry with stub operations). The class object is a throwaway
    ``_StubCls`` whose ``__name__`` matches *cls_name* so any code path
    that calls ``info.cls.__name__`` keeps working.

    Args:
        cls_name: Registry key (``OperationInfo.name``).
        parameters: Parameter dict — defaults to an empty dict so callers
            that only care about the class identity can omit it.
        category: Operation category bucket (defaults to ``"Enhancer"``
            because most validation/dispatch tests seed enhancers).

    Returns:
        Fully-populated :class:`OperationInfo` instance.
    """

    class _StubCls:
        pass

    _StubCls.__name__ = cls_name
    return OperationInfo(
        cls=_StubCls,
        name=cls_name,
        category=category,
        module="tests.fake",
        docstring="",
        parameters=parameters or {},
    )


@pytest.fixture
def empty_registry(monkeypatch):
    """Monkeypatch the validation module's ``get_registry`` symbol.

    Returns a fresh :class:`_FakeRegistry` so each test can mutate
    ``registry.ops`` independently.  The validation module always reads
    ``get_registry()`` at the top of ``_validate_scope`` so a single
    patch suffices.
    """

    reg = _FakeRegistry()
    monkeypatch.setattr(
        "phenotypic.gui.builder._validation.get_registry", lambda: reg
    )
    return reg

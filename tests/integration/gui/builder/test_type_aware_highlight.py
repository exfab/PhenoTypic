"""Tests for the per-port ``accepts`` resolution in
``build_canvas_elements_dag``.

Spec §5.5 documents the algorithm — given a parameter's annotation,
emit the list of registry class names whose ``cls`` is a subclass of
the annotated type (with PEP 604 / typing constructs handled
recursively).  The clientside ``wire_drawing.js`` reads this list on
dragstart to glow / dim ports during a wire-drag gesture.

This test suite uses a stub :class:`OperationRegistry` so the assertions
don't depend on the live phenotypic registry — every fixture annotation
form is tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Type, Union

import pytest

from phenotypic.abc_ import ImageOperation
from phenotypic.gui._operation_registry import OperationInfo, ParamInfo


def _resolve_accepts(
    type_hint: Any,
    *,
    is_operation: bool = False,
    is_pipeline: bool = False,
    is_list: bool = False,
    has_default: bool = True,
    registry_ops: List[OperationInfo] = (),
    monkeypatch: pytest.MonkeyPatch,
) -> List[str]:
    """Drive ``_resolve_dag_accepts`` against a stub registry.

    Args:
        type_hint: The annotation under test.
        is_operation: Mimics ``ParamInfo.is_operation``.
        is_pipeline: Mimics ``ParamInfo.is_pipeline``.
        is_list: Mimics ``ParamInfo.is_list``.
        has_default: Mimics ``ParamInfo.has_default``.
        registry_ops: ``OperationInfo`` records the stub registry exposes
            via ``get_categories()`` / ``get_by_category()``.
        monkeypatch: Pytest's monkeypatching fixture.

    Returns:
        Sorted list of accepted class names.
    """

    from phenotypic.gui.builder import _layout

    @dataclass
    class _StubRegistry:
        ops: List[OperationInfo] = field(default_factory=list)

        def get_categories(self) -> List[str]:
            return sorted({op.category for op in self.ops})

        def get_by_category(self, category: str) -> List[OperationInfo]:
            return [op for op in self.ops if op.category == category]

    stub = _StubRegistry(list(registry_ops))
    monkeypatch.setattr(
        "phenotypic.gui._operation_registry.get_registry",
        lambda: stub,
    )

    param_info = ParamInfo(
        name="port",
        type_hint=type_hint,
        default=None,
        has_default=has_default,
        is_operation=is_operation,
        is_pipeline=is_pipeline,
        is_optional=False,
        is_list=is_list,
    )
    return _layout._resolve_dag_accepts(param_info, stub)


class _DetectorBase(ImageOperation):
    """Stub detector base class used as a target for accepts resolution."""


class _ConcreteDetectorA(_DetectorBase):
    pass


class _ConcreteDetectorB(_DetectorBase):
    pass


class _UnrelatedClass:
    pass


def _info(name: str, cls: Type[Any], category: str = "Detector") -> OperationInfo:
    return OperationInfo(
        cls=cls,
        name=name,
        category=category,
        module=cls.__module__,
        docstring=None,
        parameters={},
    )


def test_plain_class_annotation_accepts_subclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    """``T`` accepts every registry class whose ``cls`` is a subclass of ``T``."""

    ops = [
        _info("ConcreteDetectorA", _ConcreteDetectorA),
        _info("ConcreteDetectorB", _ConcreteDetectorB),
        _info("Unrelated", _UnrelatedClass, category="Unrelated"),
    ]
    accepts = _resolve_accepts(
        _DetectorBase,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts
    assert "ConcreteDetectorB" in accepts
    assert "Unrelated" not in accepts


def test_annotated_unwraps_to_underlying_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Annotated[T, ...]`` resolves identically to ``T``."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    annotated_type = Annotated[_DetectorBase, "metadata"]
    accepts = _resolve_accepts(
        annotated_type,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts


def test_union_of_two_classes_accepts_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Union[A, B]`` accepts every subclass of A or B."""

    ops = [
        _info("ConcreteDetectorA", _ConcreteDetectorA),
        _info("ConcreteDetectorB", _ConcreteDetectorB),
    ]
    union_type = Union[_ConcreteDetectorA, _ConcreteDetectorB]
    accepts = _resolve_accepts(
        union_type,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts
    assert "ConcreteDetectorB" in accepts


def test_pep604_union_accepts_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """PEP 604 ``A | B`` resolves like ``Union[A, B]``."""

    ops = [
        _info("ConcreteDetectorA", _ConcreteDetectorA),
        _info("ConcreteDetectorB", _ConcreteDetectorB),
    ]
    union_type = _ConcreteDetectorA | _ConcreteDetectorB
    accepts = _resolve_accepts(
        union_type,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts
    assert "ConcreteDetectorB" in accepts


def test_optional_drops_none_from_union(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Optional[T]`` resolves like ``T`` (None is dropped before unioning)."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    optional_type = Optional[_DetectorBase]
    accepts = _resolve_accepts(
        optional_type,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts


def test_list_of_t_resolves_like_scalar_t(monkeypatch: pytest.MonkeyPatch) -> None:
    """``List[T]`` accepts the same classes as scalar ``T`` (list-ness via flag)."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    list_type = List[_DetectorBase]
    accepts = _resolve_accepts(
        list_type,
        is_operation=True,
        is_list=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts


def test_lowercase_list_t_resolves_like_capital_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """``list[T]`` (PEP 585) accepts the same classes as ``List[T]``."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    list_type = list[_DetectorBase]
    accepts = _resolve_accepts(
        list_type,
        is_operation=True,
        is_list=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert "ConcreteDetectorA" in accepts


def test_is_operation_and_is_pipeline_emits_full_op_set_plus_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both flags True → full registry + ``ImagePipeline`` sentinel."""

    ops = [
        _info("ConcreteDetectorA", _ConcreteDetectorA),
        _info("ConcreteDetectorB", _ConcreteDetectorB),
        _info("Unrelated", _UnrelatedClass, category="Unrelated"),
    ]
    accepts = _resolve_accepts(
        Any,
        is_operation=True,
        is_pipeline=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    # Every registered op + the pipeline sentinel
    assert "ConcreteDetectorA" in accepts
    assert "ConcreteDetectorB" in accepts
    assert "Unrelated" in accepts
    assert "ImagePipeline" in accepts


def test_is_pipeline_only_emits_pipeline_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_pipeline=True`` alone yields just ``["ImagePipeline"]``."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    accepts = _resolve_accepts(
        Any,
        is_operation=False,
        is_pipeline=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert accepts == ["ImagePipeline"]


def test_forward_reference_resolves_to_empty_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unresolved type hints produce ``accepts: []`` (advisory surface)."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    # Strings + bare forward references can't be resolved to a class.
    accepts = _resolve_accepts(
        "UnknownClass",  # forward ref string
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert accepts == []


def test_non_aux_param_returns_empty_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ColumnRef`` / scalar non-op params: ``accepts`` is empty."""

    ops = [_info("ConcreteDetectorA", _ConcreteDetectorA)]
    # Neither is_operation nor is_pipeline → not an aux port.
    accepts = _resolve_accepts(
        str,
        is_operation=False,
        is_pipeline=False,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert accepts == []


def test_accepts_list_is_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output is deterministic (sorted by class name)."""

    ops = [
        _info("ZDetector", _ConcreteDetectorB),
        _info("ADetector", _ConcreteDetectorA),
    ]
    accepts = _resolve_accepts(
        _DetectorBase,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert accepts == sorted(accepts)


def test_accepts_list_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate registry entries don't produce duplicate accepts entries."""

    ops = [
        _info("ConcreteDetectorA", _ConcreteDetectorA),
        _info("ConcreteDetectorA", _ConcreteDetectorA, category="DuplicateCat"),
    ]
    accepts = _resolve_accepts(
        _DetectorBase,
        is_operation=True,
        registry_ops=ops,
        monkeypatch=monkeypatch,
    )
    assert accepts.count("ConcreteDetectorA") == 1

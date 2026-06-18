"""Type-level markers for column-name parameters.

These markers attach metadata to ``Annotated[T, ...]`` annotations on
analyzer / model parameters so the GUI can render the field as a
dropdown populated from the live ``measurements.parquet`` schema rather
than as a free-text input.

At runtime the annotated value is still a plain :class:`str` or
``list[str]`` — the marker is purely informational and only surfaces
through ``typing.get_type_hints(..., include_extras=True)``.

Usage::

    from phenotypic.sdk_ import ColumnRef, ColumnRefList

    class MyAnalyzer(SetAnalyzer):
        def __init__(self, on: ColumnRef, groupby: ColumnRefList): ...
"""
from __future__ import annotations

from typing import Annotated, Literal

ColumnSource = Literal["measurements", "master_measurements"]


class _ColumnRefMarker:
    """Sentinel attached to ``Annotated[T, ...]`` for column-name params.

    Carries the source file the GUI should read columns from. Equality
    is by ``source`` so duplicate markers compare equal.
    """

    __slots__ = ("source",)

    def __init__(self, source: ColumnSource = "measurements") -> None:
        self.source: ColumnSource = source

    def __repr__(self) -> str:
        return f"_ColumnRefMarker({self.source!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ColumnRefMarker) and other.source == self.source

    def __hash__(self) -> int:
        return hash(("_ColumnRefMarker", self.source))


#: Annotated alias for a single column name parameter. Resolves to
#: ``str`` at runtime; the GUI renders a single-select dropdown.
ColumnRef = Annotated[str, _ColumnRefMarker("measurements")]

#: Annotated alias for a list-of-column-names parameter. Resolves to
#: ``list[str]`` at runtime; the GUI renders a multi-select dropdown.
ColumnRefList = Annotated[list[str], _ColumnRefMarker("measurements")]


__all__ = [
    "ColumnRef",
    "ColumnRefList",
    "ColumnSource",
    "_ColumnRefMarker",
]

"""Curated, repo-defined groupings of measurement columns across metric families.

A category is a many-to-many label: one column may carry several categories and
one category spans several metric families (``SIZE``, ``ColorLab``, …). Unlike
the kind/tier classification in :mod:`._tiers`, a category makes no trust
claim. The CLI writes one spreadsheet per category under
``deliverables/measurements_by_category/``.

Declare membership on the measurement itself::

    AREA = Entry("Area", "...", categories=CATEGORIES.STARTING_METRICS)

This module imports only the standard library at module level, keeping
``phenotypic.schema`` import-light; :meth:`CATEGORIES.members` imports the
schema package lazily.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._measurement_info import MeasurementInfo

_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
#: Word boundaries in a CamelCase label: lower/digit→Upper, and the last capital
#: of an acronym before a capitalised word ("CIELabColor" → "CIE Lab Color").
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ANCHOR_PREFIX = "measurement-category-"


@dataclass(frozen=True, slots=True)
class CategoryEntry:
    """Declarative value for a :class:`CATEGORIES` member.

    Args:
        label: CamelCase token used as the output file stem and anchor slug
            (e.g. ``"StartingMetrics"``).
        desc: Technical description of what the category groups and why.
    """

    label: str
    desc: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not _LABEL_RE.match(self.label):
            raise ValueError(
                f"CategoryEntry.label must be a CamelCase token, got {self.label!r}"
            )
        if not isinstance(self.desc, str) or not self.desc.strip():
            raise ValueError("CategoryEntry.desc must be a non-empty string")


class CATEGORIES(str, Enum):
    """Repo-defined measurement categories.

    Each member's value is its label (no metric-family prefix). Members expose
    ``label``, ``desc``, ``display_name`` and ``anchor``.
    """

    label: str
    desc: str

    @staticmethod
    def _validate_entry(entry: object) -> CategoryEntry:
        if not isinstance(entry, CategoryEntry):
            raise TypeError(
                "CATEGORIES members must be declared as CategoryEntry(...); "
                f"got {entry!r}"
            )
        return entry

    def __new__(cls, entry: CategoryEntry) -> CATEGORIES:
        entry = cls._validate_entry(entry)
        obj = str.__new__(cls, entry.label)
        obj._value_ = entry.label
        obj.label = entry.label
        obj.desc = entry.desc
        return obj

    def __str__(self) -> str:
        return self._value_

    @property
    def display_name(self) -> str:
        """Human-readable name: the label split on CamelCase boundaries."""
        return _CAMEL_BOUNDARY_RE.sub(" ", self.label)

    @property
    def anchor(self) -> str:
        """Stable Sphinx label of this category's section on the Categories page."""
        return f"{_ANCHOR_PREFIX}{self.label.lower()}"

    @classmethod
    def in_order(cls, categories: Iterable[CATEGORIES]) -> tuple[CATEGORIES, ...]:
        """Return *categories* sorted by declaration order."""
        order = list(cls)
        return tuple(sorted(set(categories), key=order.index))

    def members(self) -> tuple[MeasurementInfo, ...]:
        """Every public schema member carrying this category.

        Ordered by ``phenotypic.schema.__all__`` export order, then member
        order; compatibility aliases are deduplicated to their class.
        Deliberately **not** cached: it walks ~40 classes and is called only
        by docs and tests, and caching would hide classes registered later.
        """
        import phenotypic.schema as schema

        from ._measurement_info import MeasurementInfo

        seen: set[type] = set()
        found: list[MeasurementInfo] = []
        for name in schema.__all__:
            info = getattr(schema, name, None)
            if (
                not isinstance(info, type)
                or not issubclass(info, MeasurementInfo)
                or info is MeasurementInfo
                or info in seen
            ):
                continue
            seen.add(info)
            found.extend(member for member in info if self in member.categories)
        return tuple(found)

    STARTING_METRICS = CategoryEntry(
        "StartingMetrics",
        "Core per-colony magnitudes to examine first: the size measurements, "
        "integrated grayscale intensity, and the CIELAB medoid colour.",
    )

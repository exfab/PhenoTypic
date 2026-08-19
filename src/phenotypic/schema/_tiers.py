"""Intermediate classification base classes for MeasurementInfo enums.

These member-less bases carry the coarse ``kind()`` and (for primary
measurements) ``tier()`` for the measurement-classification framework. A
measurement enum declares its classification by subclassing the matching
base instead of ``MeasurementInfo`` directly. Straddling enums subclass the
neutral parent (``PrimaryMeasure``/``DerivedMeasure``) and tag the minority
members with ``Entry(tier=...)`` / ``Entry(derivation_type=...)``.
"""

from __future__ import annotations

from functools import cache
from collections.abc import Iterable
from typing import Any, cast

from ._measurement_info import MeasurementInfo


class IdentityInfo(MeasurementInfo):
    """Identity / design-factor columns (metadata, locators)."""

    @classmethod
    def kind(cls) -> str:
        return "identity"


class MetadataInfo(IdentityInfo):
    """Identity information owned by a concrete metadata vocabulary.

    Every concrete owner emits the shared ``Metadata_<Label>`` namespace. Class
    identity, rather than a category-string prefix, is the stable signal that an
    enum owns metadata.
    """

    @classmethod
    def category(cls) -> str:
        """Return the shared physical namespace for metadata columns."""
        return "Metadata"

    @classmethod
    def _missing_(cls, value: Any) -> MetadataInfo | None:
        """Resolve an exact previous-release header during enum deserialization.

        Enum pickles reconstruct members by calling their owner with the stored
        enum value. Previous releases therefore pass values such as
        ``MetadataGenetic_Strain`` to the renamed canonical owner. The permanent
        compatibility registry is imported lazily to keep schema import order
        acyclic while allowing those stored members to resolve by identity.

        Args:
            value: Enum value supplied by the deserializer.

        Returns:
            The matching member of ``cls``, or ``None`` for the normal Enum
            ``ValueError`` path.
        """
        if not isinstance(value, str):
            return None
        from phenotypic.sdk_._metadata_compatibility import LEGACY_HEADER_TO_MEMBER

        member = LEGACY_HEADER_TO_MEMBER.get(value)
        return member if member is not None and type(member) is cls else None

    @classmethod
    @cache
    def header_set(cls) -> frozenset[str]:
        """Return this owner's finite set of currently emitted headers."""
        members = cast(Iterable[MetadataInfo], cls)
        return frozenset(member.value for member in members)


class QualityInfo(MeasurementInfo):
    """Quality / trust columns that gate analysis."""

    @classmethod
    def kind(cls) -> str:
        return "quality"


class DerivedMeasure(MeasurementInfo):
    """Model/derived outputs; per-member tier via Entry overrides."""

    @classmethod
    def kind(cls) -> str:
        return "derived"


class PrimaryMeasure(MeasurementInfo):
    """Primary measured signal with no fixed tier (used by straddlers)."""

    @classmethod
    def kind(cls) -> str:
        return "primary"


class DirectPhenotype(PrimaryMeasure):
    """Tier 1 — semantic readout, safe to interpret a single value."""

    @classmethod
    def tier(cls) -> int:
        return 1


class DescriptiveTrait(PrimaryMeasure):
    """Tier 2 — interpretable named trait; interpret directionally."""

    @classmethod
    def tier(cls) -> int:
        return 2


class DiscriminativeFeature(PrimaryMeasure):
    """Tier 3 — agnostic fingerprint; use in aggregate for discrimination."""

    @classmethod
    def tier(cls) -> int:
        return 3

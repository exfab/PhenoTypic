"""Intermediate classification base classes for MeasurementInfo enums.

These member-less bases carry the coarse ``kind()`` and (for primary
measurements) ``tier()`` for the measurement-classification framework. A
measurement enum declares its classification by subclassing the matching
base instead of ``MeasurementInfo`` directly. Straddling enums subclass the
neutral parent (``PrimaryMeasure``/``DerivedMeasure``) and tag the minority
members with ``Entry(tier=...)`` / ``Entry(derivation_type=...)``.
"""

from ._measurement_info import MeasurementInfo


class IdentityInfo(MeasurementInfo):
    """Identity / design-factor columns (metadata, locators)."""

    @classmethod
    def kind(cls) -> str:
        return "identity"


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

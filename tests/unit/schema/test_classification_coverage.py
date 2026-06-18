"""Gate: every measurement column resolves to a valid (kind, tier)."""
import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo
from phenotypic.schema._measurement_info import _VALID_KINDS
from phenotypic.schema._tiers import (
    DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
    IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure,
)

_BASES = (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
          IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure)


def _column_enums():
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj not in _BASES and obj is not MeasurementInfo and list(obj)):
            yield obj


def test_every_member_is_classified():
    failures = []
    for enum in _column_enums():
        for m in enum:
            try:
                kind, tier = m.resolved_kind, m.resolved_tier
            except ValueError as exc:
                failures.append(f"{enum.__name__}.{m.name}: {exc}")
                continue
            if kind not in _VALID_KINDS:
                failures.append(f"{enum.__name__}.{m.name}: bad kind {kind!r}")
            if tier not in (None, 1, 2, 3):
                failures.append(f"{enum.__name__}.{m.name}: bad tier {tier!r}")
            if kind == "primary" and tier is None:
                failures.append(f"{enum.__name__}.{m.name}: primary w/o tier")
    assert not failures, "Unclassified measurement columns:\n" + "\n".join(failures)


def test_intermediate_bases_have_no_members():
    for base in _BASES:
        assert list(base) == [], f"{base.__name__} must stay member-less"

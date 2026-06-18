import pytest
from phenotypic.schema import Entry
from phenotypic.schema._tiers import (
    DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
    IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure,
)


def _make(cls, **entries):
    # Build a throwaway enum subclass for testing resolution.
    # type() bypasses EnumMeta.__prepare__ on Python 3.12+, so use EnumMeta directly.
    from enum import EnumMeta
    from phenotypic.schema import MeasurementInfo  # noqa: F401
    ns = EnumMeta.__prepare__("T", (cls,))
    ns["category"] = classmethod(lambda c: "T")
    for name, kw in entries.items():
        ns[name] = Entry(name, "d", **kw)
    return EnumMeta("T", (cls,), ns)


def test_tier_bases_are_memberless():
    for base in (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
                 IdentityInfo, QualityInfo, DerivedMeasure):
        assert list(base) == []


def test_class_level_tier_resolution():
    E = _make(DiscriminativeFeature, A={})
    assert E.A.resolved_kind == "primary"
    assert E.A.resolved_tier == 3


def test_entry_override_beats_class():
    E = _make(DescriptiveTrait, A={}, B={"tier": 1})
    assert E.A.resolved_tier == 2          # class default
    assert E.B.resolved_tier == 1          # Entry override


def test_diagnostic_resolves_to_quality():
    E = _make(DerivedMeasure, A={"derivation_type": "diagnostic"})
    assert E.A.resolved_kind == "quality"
    assert E.A.resolved_tier is None


def test_normalization_is_covered_with_deferred_tier():
    E = _make(DerivedMeasure, A={"derivation_type": "normalization", "derives_from": "SIZE"})
    assert E.A.resolved_kind == "derived"
    assert E.A.resolved_tier is None


def test_unclassified_primary_member_raises():
    E = _make(DerivedMeasure, A={})        # derived, no tier, no derivation_type
    with pytest.raises(ValueError):
        _ = E.A.resolved_tier


def test_parameterization_resolves_to_derived():
    E = _make(DerivedMeasure, A={"derivation_type": "parameterization"})
    assert E.A.resolved_kind == "derived"
    assert E.A.resolved_tier is None


def test_primary_straddler_without_tier_raises():
    E = _make(PrimaryMeasure, A={})
    with pytest.raises(ValueError):
        _ = E.A.resolved_tier


def test_primary_straddler_with_entry_tier():
    E = _make(PrimaryMeasure, A={"tier": 2})
    assert E.A.resolved_kind == "primary"
    assert E.A.resolved_tier == 2


def test_entry_validates_tier_and_derivation_type():
    with pytest.raises(ValueError):
        Entry("x", "d", tier=4)
    with pytest.raises(ValueError):
        Entry("x", "d", derivation_type="bogus")


def test_tier2_primary_enums():
    from phenotypic.schema import ColorLab, ColorHSV, RADIAL_EXPANSION, SYMMETRIC_ZONES
    for enum in (ColorLab, ColorHSV, RADIAL_EXPANSION, SYMMETRIC_ZONES):
        assert all(m.resolved_tier == 2 for m in enum), enum.__name__


def test_tier1_primary_enums():
    from phenotypic.schema import SIZE, INTENSITY
    for enum in (SIZE, INTENSITY):
        assert all(m.resolved_kind == "primary" for m in enum), enum.__name__
        assert all(m.resolved_tier == 1 for m in enum), enum.__name__


def test_quality_enums_resolve_quality():
    from phenotypic.schema import (
        QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE,
        QUALITY_TUKEY, QUALITY_ZMAX, CURATION, ErrorCategory, MODEL_METRICS,
        GRID_LINREG_STATS, GRID_SPATIAL, GRID_SPREAD,
    )
    for enum in (QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE,
                 QUALITY_TUKEY, QUALITY_ZMAX, CURATION, ErrorCategory, MODEL_METRICS,
                 GRID_LINREG_STATS, GRID_SPATIAL, GRID_SPREAD):
        assert all(m.resolved_kind == "quality" for m in enum), enum.__name__


def test_identity_enums_resolve_identity():
    from phenotypic.schema import (
        METADATA, BBOX, OBJECT, GRID,
        GENETIC_METADATA, SAMPLE_METADATA, PLATE_METADATA, CONDITION_METADATA,
        INCUBATION_METADATA, ACQUISITION_METADATA, EXPERIMENT_METADATA,
    )
    for enum in (METADATA, BBOX, OBJECT, GRID, GENETIC_METADATA, SAMPLE_METADATA,
                 PLATE_METADATA, CONDITION_METADATA, INCUBATION_METADATA,
                 ACQUISITION_METADATA, EXPERIMENT_METADATA):
        assert all(m.resolved_kind == "identity" for m in enum), enum.__name__

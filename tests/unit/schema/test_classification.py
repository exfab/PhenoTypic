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

    ns = EnumMeta.__prepare__("T", (cls,))
    ns["category"] = classmethod(lambda c: "T")
    for name, kw in entries.items():
        ns[name] = Entry(name, "d", **kw)
    return EnumMeta("T", (cls,), ns)


def test_tier_bases_are_memberless():
    for base in (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
                 IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure):
        assert list(base) == []


def test_class_level_tier_resolution():
    E = _make(DiscriminativeFeature, A={})
    assert E.A.resolved_kind == "primary"
    assert E.A.resolved_tier == 3


def test_entry_override_beats_class():
    E = _make(DescriptiveTrait, A={}, B={"tier": 1})
    assert E.A.resolved_tier == 2  # class default
    assert E.B.resolved_tier == 1  # Entry override


def test_diagnostic_resolves_to_quality():
    E = _make(DerivedMeasure, A={"derivation_type": "diagnostic"})
    assert E.A.resolved_kind == "quality"
    assert E.A.resolved_tier is None


def test_normalization_is_covered_with_deferred_tier():
    E = _make(DerivedMeasure,
              A={"derivation_type": "normalization", "derives_from": "SIZE"})
    assert E.A.resolved_kind == "derived"
    assert E.A.resolved_tier is None


def test_unclassified_primary_member_raises():
    E = _make(DerivedMeasure, A={})  # derived, no tier, no derivation_type
    with pytest.raises(ValueError):
        _ = E.A.resolved_tier


def test_parameterization_with_tier_resolves_to_derived():
    E = _make(DerivedMeasure, A={"derivation_type": "parameterization", "tier": 1})
    assert E.A.resolved_kind == "derived"
    assert E.A.resolved_tier == 1


def test_parameterization_without_tier_raises():
    # parameterization members must always carry an explicit tier; only
    # normalization legitimately defers its tier to the runtime target.
    E = _make(DerivedMeasure, A={"derivation_type": "parameterization"})
    with pytest.raises(ValueError):
        _ = E.A.resolved_tier


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


def test_derived_growth_models_and_edge_correction():
    from phenotypic.schema import (
        LOG_GROWTH_MODEL, LINEAR_LAG_MODEL, LINEAR_CAP_AND_LAG_MODEL, EDGE_CORRECTION,
    )

    # LOG_GROWTH: kinetics -> Tier 1; regularization knobs -> diagnostic/quality
    assert LOG_GROWTH_MODEL.R_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.K_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.N0_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.GROWTH_RATE.resolved_tier == 1
    for knob in (LOG_GROWTH_MODEL.LAM, LOG_GROWTH_MODEL.BETA, LOG_GROWTH_MODEL.K_MAX):
        assert knob.resolved_kind == "quality"
    # LINEAR_SOFTPLUS: v/s0/lam Tier 1; alpha Tier 2
    assert LINEAR_LAG_MODEL.v.resolved_tier == 1
    assert LINEAR_LAG_MODEL.s0.resolved_tier == 1
    assert LINEAR_LAG_MODEL.lam.resolved_tier == 1
    assert LINEAR_LAG_MODEL.alpha.resolved_tier == 2
    # DOUBLE_SOFTPLUS: v/s0/lam/smax Tier 1; alpha/beta Tier 2; mode diagnostic
    for m in (LINEAR_CAP_AND_LAG_MODEL.v, LINEAR_CAP_AND_LAG_MODEL.s0,
              LINEAR_CAP_AND_LAG_MODEL.lam, LINEAR_CAP_AND_LAG_MODEL.smax):
        assert m.resolved_tier == 1, m
    assert LINEAR_CAP_AND_LAG_MODEL.alpha.resolved_tier == 2
    assert LINEAR_CAP_AND_LAG_MODEL.beta.resolved_tier == 2
    assert LINEAR_CAP_AND_LAG_MODEL.mode.resolved_kind == "quality"
    # EDGE_CORRECTION: normalization, tier deferred to target
    for m in EDGE_CORRECTION:
        assert m.resolved_kind == "derived"
        assert m.resolved_tier is None


def test_shape_straddles_tier1_and_tier2():
    from phenotypic.schema import SHAPE

    tier1 = {SHAPE.AREA, SHAPE.CONVEX_AREA, SHAPE.MEDIAN_RADIUS, SHAPE.MEAN_RADIUS,
             SHAPE.MAX_RADIUS, SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER,
             SHAPE.MAJOR_AXIS_LENGTH, SHAPE.MINOR_AXIS_LENGTH, SHAPE.BBOX_AREA,
             SHAPE.PERIMETER}
    tier2 = {SHAPE.CIRCULARITY, SHAPE.ECCENTRICITY, SHAPE.SOLIDITY, SHAPE.EXTENT,
             SHAPE.COMPACTNESS, SHAPE.ORIENTATION}
    for m in tier1:
        assert m.resolved_tier == 1, m
    for m in tier2:
        assert m.resolved_tier == 2, m
    assert tier1 | tier2 == set(SHAPE)  # full coverage, no member missed


def test_tier3_primary_enums():
    from phenotypic.schema import TEXTURE, ColorXYZ, Colorxy, ColorComposition

    for enum in (TEXTURE, ColorXYZ, Colorxy, ColorComposition):
        assert all(m.resolved_tier == 3 for m in enum), enum.__name__


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
        GRID_LINREG_STATS, NEIGHBOR_DIST, GRID_SPREAD,
    )

    for enum in (QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE,
                 QUALITY_TUKEY, QUALITY_ZMAX, CURATION, ErrorCategory, MODEL_METRICS,
                 GRID_LINREG_STATS, NEIGHBOR_DIST, GRID_SPREAD):
        assert all(m.resolved_kind == "quality" for m in enum), enum.__name__


def test_identity_enums_resolve_identity():
    from phenotypic.schema import (
        METADATA, BBOX, OBJECT, GRID,
        GENETIC_METADATA, SAMPLE_METADATA, PLATE_METADATA, CONDITION_METADATA,
        CULTURE_METADATA, ACQUISITION_METADATA, EXPERIMENT_METADATA,
    )

    for enum in (METADATA, BBOX, OBJECT, GRID, GENETIC_METADATA, SAMPLE_METADATA,
                 PLATE_METADATA, CONDITION_METADATA, CULTURE_METADATA,
                 ACQUISITION_METADATA, EXPERIMENT_METADATA):
        assert all(m.resolved_kind == "identity" for m in enum), enum.__name__


def test_rst_table_includes_type_column_with_tier_badge():
    from phenotypic.schema import TEXTURE, METADATA

    txt = TEXTURE.rst_table()
    assert "- Type" in txt
    # Tier 3 renders the amber (warning) discriminative-feature badge.
    assert ":bdg-ref-warning:`Tier 3 · Discriminative feature <measurement-tiers>`" in txt
    # Identity enums now classify too: the Type column is present with grey
    # Identity badges (previously the column was suppressed entirely).
    meta = METADATA.rst_table()
    assert "- Type" in meta
    assert ":bdg-ref-muted:`Identity / design <measurement-classification>`" in meta


def test_derived_tier1_columns_surface_use_badge():
    from phenotypic.schema import LOG_GROWTH_MODEL

    txt = LOG_GROWTH_MODEL.rst_table()
    assert "- Type" in txt
    assert ":bdg-ref-success:`Tier 1 · Direct phenotype <measurement-tiers>`" in txt
    # The plain-text use_label contract is unchanged (badge is a separate accessor).
    assert LOG_GROWTH_MODEL.R_FIT.use_label == "Direct phenotype (Tier 1)"


def test_badge_spec_colors_are_valid_sphinx_design_semantic_colors():
    # A typo'd colour (e.g. "grey") would silently emit a bdg-ref role that
    # docutils can't resolve, surfacing only as a warning in the -W docs build.
    # Guard it here for fast feedback. Skips when sphinx-design isn't installed
    # (it lives in the optional ``docs`` dependency group).
    import pytest

    sd_shared = pytest.importorskip("sphinx_design.shared")
    from phenotypic.schema._measurement_info import _BADGE_SPECS

    valid = set(sd_shared.SEMANTIC_COLORS)
    used = {color for _text, color, _anchor in _BADGE_SPECS.values()}
    assert used <= valid, f"unknown sphinx-design colours: {sorted(used - valid)}"

import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo, REMBI_MODULE
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


def test_metadata_enums_declare_a_real_module():
    bad = []
    for enum in _column_enums():
        if not enum.category().startswith("Metadata"):
            continue
        for m in enum:
            mod = m.resolved_rembi_module
            if mod in (REMBI_MODULE.ANALYZED_DATA, REMBI_MODULE.UNCATEGORIZED):
                bad.append(f"{enum.__name__}.{m.name} -> {mod}")
    assert not bad, "metadata members must declare a real REMBI module:\n" + "\n".join(bad)


def test_resolved_module_is_total():
    for enum in _column_enums():
        for m in enum:
            assert isinstance(m.resolved_rembi_module, REMBI_MODULE)


def test_culture_time_members_are_biosample():
    from phenotypic.schema import CULTURE
    assert CULTURE.TIME.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE.TIME_UNIT.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE.TIMEPOINT.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE.FRAME_INDEX.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE.TEMPERATURE.resolved_rembi_module is REMBI_MODULE.SPECIMEN_PREP

from phenotypic.schema import Entry, REMBI_MODULE
from phenotypic.schema._tiers import IdentityInfo


class _ModEnum(IdentityInfo):
    @classmethod
    def category(cls) -> str:
        return "TestMod"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.BIOSAMPLE

    PLAIN = Entry("Plain", "uses the enum-level module")
    OVERRIDDEN = Entry("Overridden", "per-member override",
                       rembi_module=REMBI_MODULE.SPECIMEN_PREP)


class _NoModEnum(IdentityInfo):
    @classmethod
    def category(cls) -> str:
        return "TestNoMod"

    LONELY = Entry("Lonely", "no module declared -> fallback")


def test_canonical_module_order():
    assert [m.value for m in REMBI_MODULE] == [
        "Study", "Biosample", "SpecimenPreparation", "ImageAcquisition",
        "ImageData", "AnalyzedData", "Uncategorized",
    ]


def test_enum_level_module():
    assert _ModEnum.PLAIN.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE


def test_member_override_wins():
    assert _ModEnum.OVERRIDDEN.resolved_rembi_module is REMBI_MODULE.SPECIMEN_PREP


def test_fallback_is_analyzed_data():
    assert _NoModEnum.LONELY.resolved_rembi_module is REMBI_MODULE.ANALYZED_DATA


def test_entry_rejects_bad_module():
    import pytest
    with pytest.raises((ValueError, TypeError)):
        Entry("X", "bad", rembi_module="Biosample")  # not a REMBI_MODULE

"""Unit assertions for B7 decouple: ICC default routes through schema enums.

The subject_label default equals ``str(CULTURE_METADATA.TIME)`` after B7;
the rater_label conversion to BIO_REPLICATE is deferred (value would differ
from "Metadata_Replicate" today, breaking existing test_icc.py assertions).
"""

from phenotypic.schema import CULTURE_METADATA, SAMPLE_METADATA
from phenotypic.analysis.qc._icc import ICC


def test_icc_default_subject_is_culture_time() -> None:
    """subject_label default routes through the schema enum after B7."""
    assert ICC(on="Size_Area", groupby=["Plate"]).subject_label == str(CULTURE_METADATA.TIME)


def test_enum_values_unchanged_pre_b2_flip() -> None:
    """Decouple phase: enum values still equal the legacy column strings."""
    assert str(CULTURE_METADATA.TIME) == "Metadata_Time"
    assert str(SAMPLE_METADATA.SAMPLE_ID) == "Metadata_SampleID"

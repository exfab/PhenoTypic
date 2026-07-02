"""Unit assertions for the ICC default axis labels (B7 decouple + B2 flip).

Both axis defaults route through the schema enums: ``subject_label`` ==
``str(CULTURE_METADATA.TIME)`` and ``rater_label`` ==
``str(SAMPLE_METADATA.BIO_REPLICATE)``. After the B2 category flip those enum
values are the per-topic Scheme-B headers.
"""

from phenotypic.schema import CULTURE_METADATA, SAMPLE_METADATA
from phenotypic.analysis.qc._icc import ICC


def test_icc_default_subject_is_culture_time() -> None:
    """subject_label default routes through the schema enum."""
    assert ICC(on="Size_Area", groupby=["Plate"]).subject_label == str(CULTURE_METADATA.TIME)


def test_icc_default_rater_is_bio_replicate() -> None:
    """rater_label default routes through the schema enum (B-flip carry-forward)."""
    assert ICC(on="Size_Area", groupby=["Plate"]).rater_label == str(
        SAMPLE_METADATA.BIO_REPLICATE
    )


def test_enum_values_are_scheme_b_after_flip() -> None:
    """Post B2 flip: enum values are the per-topic Scheme-B headers."""
    assert str(CULTURE_METADATA.TIME) == "MetadataCulture_Time"
    assert str(SAMPLE_METADATA.SAMPLE_ID) == "MetadataSample_SampleID"
    assert str(SAMPLE_METADATA.BIO_REPLICATE) == "MetadataSample_BioReplicate"

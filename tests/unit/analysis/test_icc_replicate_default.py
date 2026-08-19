"""Unit assertions for the ICC default axis labels (B7 decouple + B2 flip).

Both axis defaults route through the canonical schema owners: ``subject_label``
== ``str(CULTURE.TIME)`` and ``rater_label`` ==
``str(SAMPLE.BIO_REPLICATE)``.
"""

from phenotypic.analysis.qc._icc import ICC
from phenotypic.schema import CULTURE, SAMPLE


def test_icc_default_subject_is_culture_time() -> None:
    """subject_label default routes through the schema enum."""
    assert ICC(on="Size_Area", groupby=["Plate"]).subject_label == str(CULTURE.TIME)


def test_icc_default_rater_is_bio_replicate() -> None:
    """rater_label default routes through the schema enum (B-flip carry-forward)."""
    assert ICC(on="Size_Area", groupby=["Plate"]).rater_label == str(
        SAMPLE.BIO_REPLICATE
    )


def test_flat_metadata_references_resolve_to_the_current_schema() -> None:
    """Flat C4-style axis settings stay accepted during the C3 transition."""
    check = ICC(
        on="Size_Area",
        groupby=["Plate"],
        subject_label="Metadata_Time",
        rater_label="Metadata_BioReplicate",
    )

    assert check.subject_label == str(CULTURE.TIME)
    assert check.rater_label == str(SAMPLE.BIO_REPLICATE)

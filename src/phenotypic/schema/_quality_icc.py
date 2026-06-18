"""Measurement info container for replicate-agreement ICC QC."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class QUALITY_ICC(QualityInfo):
    """Measurement info for the replicate-agreement ICC QC check.

    Documents the per-check extra columns produced by the intraclass
    correlation coefficient (ICC) agreement check, which models colony
    phenotypes with a two-way design to quantify how consistently replicates
    (raters) reproduce each subject's measurement. Low agreement between
    replicates often signals contamination, edge artifacts, or imaging issues
    rather than real biology, so these columns support curation decisions
    downstream of the QC pipeline.
    """

    @classmethod
    def category(cls) -> str:
        return "QC_ICC"

    NUM_SUBJECTS = Entry("NumSubjects", "Distinct subjects (e.g. timepoints) in the two-way model.")
    NUM_RATERS = Entry("NumRaters", "Replicates per subject in the two-way model.")
    NUM_MEMBERS = Entry("NumMembers", "Total measurements contributing to the ICC.")
